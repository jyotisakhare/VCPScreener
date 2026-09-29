"""Tunable thresholds for the screener.

Defaults follow the numbers Mark Minervini gives in "Trade Like a Stock Market
Wizard" / "Think & Trade Like a Champion". Everything is in one place so the
rules can be tightened or loosened without touching the detection code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendTemplateConfig:
    # 200 bars for the long SMA + 22 bars to measure its slope.
    min_history_bars: int = 222
    sma200_rising_bars: int = 22  # "200-day MA trending up for at least 1 month"
    min_pct_above_52w_low: float = 30.0
    max_pct_below_52w_high: float = 25.0
    min_rs_rating: float = 70.0


@dataclass(frozen=True)
class VCPConfig:
    lookback_bars: int = 252  # where to look for the start of the base
    min_contractions: int = 2
    max_contractions: int = 6
    min_contraction_depth_pct: float = 2.0  # smaller wiggles are noise, not contractions
    # High-to-low in fewer bars is a shakeout spike (volatility expanding), not a pullback.
    min_contraction_bars: int = 3
    max_base_depth_pct: float = 35.0  # deepest (first) contraction
    max_final_depth_pct: float = 10.0  # the last contraction must be tight
    # Volatility must actually contract: final depth <= this fraction of the first.
    max_final_to_first_ratio: float = 0.5
    min_base_bars: int = 15  # ~3 weeks
    # Where the pivot sits between the base low (0.0) and the base high (1.0).
    # Minervini's standard pivot forms in the upper third of the base; the middle
    # third is his "cheat" entry and the lower third a "low cheat" - earlier but
    # riskier entries with overhead supply, so they are opt-in.
    pivot_zone_floor: float = 2 / 3
    cheat_zone_floor: float = 1 / 3
    allow_cheat_entries: bool = False
    # Volume dry-up over the last N bars of the base, relative to the 50-bar average:
    # the N-bar average must be clearly below normal and at least one session must
    # be outright quiet (supply has stopped coming to market).
    dry_up_window: int = 10
    volume_avg_window: int = 50
    max_dry_up_ratio: float = 0.7
    max_quietest_day_ratio: float = 0.6
    # A setup is only actionable if price is close to the pivot.
    max_pct_below_pivot: float = 5.0
    buy_zone_pct: float = 5.0  # don't chase more than this above the pivot
    max_breakout_age_bars: int = 10
    breakout_volume_multiple: float = 1.4


@dataclass(frozen=True)
class TradePlanConfig:
    stop_buffer_pct: float = 0.5  # room below the last contraction low
    # Minervini's ceiling on risk per trade, measured from the price actually paid.
    # A setup whose logical stop is further than this from the pivot is skipped, not
    # given an arbitrary closer stop: a wide final contraction is not a low-risk entry.
    max_stop_pct: float = 8.0
    min_stop_pct: float = 3.0  # avoid stops that sit inside normal daily noise
    reward_to_risk: float = 3.0


@dataclass(frozen=True)
class ScreenerConfig:
    trend: TrendTemplateConfig = TrendTemplateConfig()
    vcp: VCPConfig = VCPConfig()
    plan: TradePlanConfig = TradePlanConfig()
