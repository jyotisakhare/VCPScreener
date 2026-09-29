"""Volatility Contraction Pattern (VCP) detection.

A VCP is a base in which price pulls back several times, each pullback
shallower than the one before (e.g. 25% -> 12% -> 5%), while volume dries up.
The high of the last, tightest contraction is the *pivot*: the buy point.

Contractions are found without any zig-zag threshold:

    H1 = highest high of the lookback window          (left side of the base)
    L1 = lowest low after H1
    H2 = highest high after L1,  L2 = lowest low after H2,  ...

By construction every Hk is <= the previous high and every Lk is >= the
previous low, so the sequence describes a range that keeps narrowing towards
the right edge of the chart - exactly the footprint Minervini describes.

A stock that broke out a few bars ago would make "today" the highest high and
hide the base, so detection is also attempted with the last ``t`` bars cut off;
the cut-off bars must then contain a genuine breakout: a *close* above the pivot.
An intraday poke that closes back under the pivot is still part of the base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import VCPConfig

STATUS_SETUP = "SETUP"  # below the pivot: buy when price trades through it
STATUS_BREAKOUT = "BREAKOUT"  # closed through the pivot and still inside the buy zone

ENTRY_PIVOT = "PIVOT"  # pivot in the upper third of the base: the standard VCP buy point
ENTRY_CHEAT = "CHEAT"  # middle third
ENTRY_LOW_CHEAT = "LOW_CHEAT"  # lower third


@dataclass
class Contraction:
    high_date: str
    high: float
    low_date: str
    low: float
    depth_pct: float
    bars: int


@dataclass
class VCPResult:
    status: str
    price: float  # latest close
    pivot: float
    entry_type: str
    pivot_position: float  # 0.0 = base low, 1.0 = base high
    last_contraction_low: float
    contractions: list[Contraction]
    base_start: str
    base_bars: int
    base_depth_pct: float
    volume_dry_up_ratio: float
    quietest_day_ratio: float
    pct_from_pivot: float  # negative = still below the pivot
    breakout_bars_ago: int | None = None
    breakout_volume_ratio: float | None = None
    breakout_volume_confirmed: bool | None = None

    @property
    def footprint(self) -> str:
        """Minervini's shorthand, e.g. ``9W 24/11/4 3T``."""
        depths = "/".join(f"{c.depth_pct:.0f}" for c in self.contractions)
        return f"{max(1, round(self.base_bars / 5))}W {depths} {len(self.contractions)}T"


def find_contractions(
    high: np.ndarray, low: np.ndarray, start: int, min_depth_pct: float, min_bars: int = 1
) -> list[tuple[int, int]]:
    """Return ``(high_idx, low_idx)`` pairs of successively narrower pullbacks."""
    end = len(high)
    if end - start < 2:
        return []
    pairs: list[tuple[int, int]] = []
    hi = start + int(np.argmax(high[start:end]))
    while hi + 1 < end:
        lo = hi + 1 + int(np.argmin(low[hi + 1 : end]))
        depth_pct = (high[hi] - low[lo]) / high[hi] * 100
        if depth_pct < min_depth_pct or lo - hi < min_bars:
            break
        pairs.append((hi, lo))
        if lo + 1 >= end:
            break
        hi = lo + 1 + int(np.argmax(high[lo + 1 : end]))
    return pairs


def _detect_base(df: pd.DataFrame, cfg: VCPConfig) -> VCPResult | None:
    """Look for a valid, not-yet-broken-out VCP ending at the last bar of ``df``."""
    n = len(df)
    if n < max(cfg.min_base_bars, cfg.volume_avg_window):
        return None
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    pairs = find_contractions(
        high, low, max(0, n - cfg.lookback_bars), cfg.min_contraction_depth_pct, cfg.min_contraction_bars
    )
    if not cfg.min_contractions <= len(pairs) <= cfg.max_contractions:
        return None

    depths = [(high[h] - low[l]) / high[h] * 100 for h, l in pairs]
    if depths[0] > cfg.max_base_depth_pct or depths[-1] > cfg.max_final_depth_pct:
        return None
    if any(later >= earlier for earlier, later in zip(depths, depths[1:])):
        return None
    if depths[-1] > depths[0] * cfg.max_final_to_first_ratio:
        return None

    base_bars = n - pairs[0][0]
    if base_bars < cfg.min_base_bars:
        return None

    avg_volume = volume[-cfg.volume_avg_window :].mean()
    if avg_volume <= 0:
        return None
    recent_volume = volume[-cfg.dry_up_window :]
    dry_up_ratio = float(recent_volume.mean() / avg_volume)
    quietest_day_ratio = float(recent_volume.min() / avg_volume)
    if dry_up_ratio > cfg.max_dry_up_ratio or quietest_day_ratio > cfg.max_quietest_day_ratio:
        return None

    pivot = float(high[pairs[-1][0]])
    base_high, base_low = float(high[pairs[0][0]]), float(low[pairs[0][1]])
    pivot_position = (pivot - base_low) / (base_high - base_low)
    if pivot_position >= cfg.pivot_zone_floor:
        entry_type = ENTRY_PIVOT
    elif pivot_position >= cfg.cheat_zone_floor:
        entry_type = ENTRY_CHEAT
    else:
        entry_type = ENTRY_LOW_CHEAT
    if entry_type != ENTRY_PIVOT and not cfg.allow_cheat_entries:
        return None

    price = float(df["close"].iloc[-1])
    dates = df.index
    return VCPResult(
        status=STATUS_SETUP,
        price=price,
        pivot=pivot,
        entry_type=entry_type,
        pivot_position=float(pivot_position),
        last_contraction_low=float(low[pairs[-1][1]]),
        contractions=[
            Contraction(_date(dates[h]), float(high[h]), _date(dates[l]), float(low[l]), float(d), int(l - h))
            for (h, l), d in zip(pairs, depths)
        ],
        base_start=_date(dates[pairs[0][0]]),
        base_bars=int(base_bars),
        base_depth_pct=float(depths[0]),
        volume_dry_up_ratio=dry_up_ratio,
        quietest_day_ratio=quietest_day_ratio,
        pct_from_pivot=(price / pivot - 1.0) * 100,
    )


def find_vcp_candidates(df: pd.DataFrame, cfg: VCPConfig = VCPConfig()) -> list[VCPResult]:
    """Every *actionable* reading of the chart (near the pivot or freshly broken out), best first.

    Several readings can be valid at once: once price clears a tight pivot and
    keeps rising, that contraction is absorbed into the rally and the older,
    wider pivot above becomes a "setup" too. They are ranked by the tightness of
    the final contraction - the most precise pivot and the closest logical stop.
    """
    n = len(df)
    price = float(df["close"].iloc[-1])
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    candidates: list[VCPResult] = []
    setup = _detect_base(df, cfg)
    if setup is not None and setup.pct_from_pivot >= -cfg.max_pct_below_pivot:
        candidates.append(setup)

    for age in range(1, cfg.max_breakout_age_bars + 1):
        cut = n - age
        base = _detect_base(df.iloc[:cut], cfg)
        if base is None:
            continue
        # The breakout is the first *close* above the pivot. Bars before it that only
        # poked through intraday still belong to the base.
        closes_above = np.flatnonzero(close[cut:] > base.pivot)
        if closes_above.size == 0:
            continue
        breakout_idx = cut + int(closes_above[0])
        if low[cut:].min() <= base.last_contraction_low:
            continue  # undercut the base: the pattern failed
        if not base.pivot <= price <= base.pivot * (1 + cfg.buy_zone_pct / 100):
            continue  # back under the pivot, or too extended to chase
        avg_volume = volume[max(0, breakout_idx - cfg.volume_avg_window) : breakout_idx].mean()
        ratio = float(volume[breakout_idx] / avg_volume) if avg_volume > 0 else None
        base.status = STATUS_BREAKOUT
        base.price = price
        base.pct_from_pivot = (price / base.pivot - 1.0) * 100
        base.breakout_bars_ago = n - 1 - breakout_idx
        base.breakout_volume_ratio = ratio
        base.breakout_volume_confirmed = ratio is not None and ratio >= cfg.breakout_volume_multiple
        candidates.append(base)

    candidates.sort(key=lambda c: (c.contractions[-1].depth_pct, c.breakout_bars_ago or 0))
    return candidates


def detect_vcp(df: pd.DataFrame, cfg: VCPConfig = VCPConfig()) -> VCPResult | None:
    candidates = find_vcp_candidates(df, cfg)
    return candidates[0] if candidates else None


def _date(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()
