"""Synthetic market for demo mode.

Generates a made-up universe - a few textbook VCP setups and breakouts among extended
leaders, laggards and random walks - so the dashboard and API can be explored without
market-data access. Deterministic per ticker. Nothing here resembles a real company.
"""

from __future__ import annotations

import zlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .indices import Constituent, IndexSpec

_NAMES = """Alder Amber Arrow Aster Atlas Basil Beacon Birch Bolt Cedar Cobalt Comet Coral Crest Delta Drift
Ember Falcon Fjord Flint Garnet Glade Harbor Hazel Indigo Iris Jasper Juniper Kestrel Lark Linden Lotus Maple
Meadow Nimbus Oak Onyx Opal Orchid Osprey Pearl Pine Quartz Raven Reef Sable Sage Sierra Slate Sol Summit Tern
Thistle Topaz Tundra Umber Vale Vertex Willow Zephyr""".split()
_SUFFIXES = ["Industries", "Technologies", "Finance", "Pharma", "Energy", "Motors", "Foods", "Infra"]
_SECTORS = ["Capital Goods", "Information Technology", "Financial Services", "Healthcare", "Energy", "Automobile", "FMCG", "Construction"]
_UNIVERSE_SIZE = {"NIFTY50": 50, "NIFTYNEXT50": 50, "NIFTYBANK": 14, "NIFTYIT": 10}
BARS = 1200  # ~4.8 years, enough for a 3-year backtest


def demo_constituents(spec: IndexSpec, refresh: bool = False) -> list[Constituent]:
    count = _UNIVERSE_SIZE.get(spec.key, 60)
    offset = zlib.crc32(spec.key.encode()) % len(_NAMES)
    out = []
    for i in range(count):
        word = _NAMES[(offset + i) % len(_NAMES)]
        kind = (offset + i * 3) % len(_SUFFIXES)
        symbol = f"{word.upper()}{'' if i < len(_NAMES) else i}"
        out.append(Constituent(symbol, f"{symbol}{spec.ticker_suffix}", f"{word} {_SUFFIXES[kind]} (demo)", _SECTORS[kind]))
    return out


def _path(anchors: list[tuple[int, float]], n: int) -> np.ndarray:
    return np.interp(np.arange(n), [a[0] for a in anchors], np.log([a[1] for a in anchors]))


def _vcp_anchors(rng: np.random.Generator, n: int, ending: str) -> tuple[list, list]:
    """Price and volume anchors for an advance followed by a 3-contraction base."""
    d1 = rng.uniform(0.15, 0.26)
    d2 = d1 * rng.uniform(0.38, 0.5)
    d3 = max(0.035, d2 * rng.uniform(0.4, 0.5))
    tail = {"setup": 0, "breakout": int(rng.integers(1, 4)), "extended": int(rng.integers(8, 16))}[ending]
    end = n - 1 - tail  # bar where the base is complete and price sits under the pivot
    b = end - int(rng.integers(55, 75))  # base high
    h2, h3 = 0.975, 0.965
    price = [
        (0, 22), (b - 600, 30), (b - 330, 38), (b - 200, 62), (b - 170, 55), (b, 100),
        (b + 15, 100 * (1 - d1)), (b + 30, 100 * h2), (b + 40, 100 * h2 * (1 - d2)),
        (end - 10, 100 * h3), (end - 4, 100 * h3 * (1 - d3)), (end, 100 * h3 * 0.988),
    ]
    volume = [(0, 1.0), (b, 1.25), (b + 30, 0.95), (end - 10, 0.6), (end, 0.36)]
    if ending == "breakout":
        price.append((n - 1, 100 * h3 * rng.uniform(1.012, 1.022)))
        volume += [(end + 1, 2.6), (n - 1, 1.8)]
    elif ending == "extended":
        price.append((n - 1, 100 * h3 * rng.uniform(1.12, 1.25)))
        volume += [(end + 1, 2.6), (n - 1, 1.3)]
    return price, volume


def _series(ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    n = len(index)
    kind = rng.choice(["setup", "breakout", "extended", "walk"], p=[0.14, 0.08, 0.12, 0.66])
    if kind == "walk":
        drift = rng.normal(0.0002, 0.0007)
        log_close = np.log(100) + np.cumsum(rng.normal(drift, rng.uniform(0.011, 0.022), n))
        volume_shape = np.ones(n)
        noise = 0.0
    else:
        price, volume = _vcp_anchors(rng, n, kind)
        log_close = _path(price, n)
        volume_shape = np.interp(np.arange(n), [v[0] for v in volume], [v[1] for v in volume])
        noise = 0.0022
    close = np.exp(log_close + rng.normal(0, noise, n)) * rng.uniform(0.8, 30)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0015, n))
    wick = np.abs(rng.normal(0, 0.003 if kind != "walk" else 0.008, n))
    sigma = 0.35 if kind == "walk" else 0.12
    volume = volume_shape * rng.uniform(2e5, 5e6) * rng.lognormal(0, sigma, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + wick),
            "low": np.minimum(open_, close) * (1 - wick),
            "close": close,
            "volume": volume.round(),
        },
        index=index,
    )


class DemoPriceProvider:
    def get_history(self, tickers: list[str], years: float | None = None) -> dict[str, pd.DataFrame]:
        last_session = pd.bdate_range(end=(datetime.now() - timedelta(days=1)).date(), periods=1)[0]
        index = pd.bdate_range(end=last_session, periods=BARS)
        bars = BARS if years is not None else 520
        return {t: _series(t, index).iloc[-bars:] for t in tickers}
