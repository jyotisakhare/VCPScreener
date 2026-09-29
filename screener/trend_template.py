"""Minervini's Trend Template (stage-2 uptrend filter) and relative strength."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TrendTemplateConfig

# IBD-style weighting: the latest quarter counts double.
_RS_PERIODS = ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2))


def relative_strength_score(close: pd.Series) -> float:
    """Weighted 3/6/9/12-month performance; NaN when there is under a year of data."""
    if len(close) <= _RS_PERIODS[-1][0]:
        return float("nan")
    last = close.iloc[-1]
    return float(sum(w * (last / close.iloc[-1 - bars] - 1.0) for bars, w in _RS_PERIODS))


def rs_ratings(histories: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Percentile-rank (1-99) every stock's RS score against the rest of the universe."""
    scores = pd.Series({t: relative_strength_score(df["close"]) for t, df in histories.items()}, dtype=float)
    ranked = scores.dropna().rank(pct=True) * 99
    return {t: float(max(1.0, round(r))) for t, r in ranked.items()}


@dataclass
class TrendTemplateResult:
    passed: bool
    criteria: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


def evaluate_trend_template(
    df: pd.DataFrame, rs_rating: float | None, cfg: TrendTemplateConfig = TrendTemplateConfig()
) -> TrendTemplateResult:
    if len(df) < cfg.min_history_bars:
        return TrendTemplateResult(False, reason=f"needs {cfg.min_history_bars} bars, has {len(df)}")

    close = df["close"]
    price = float(close.iloc[-1])
    sma50 = float(close.iloc[-50:].mean())
    sma150 = float(close.iloc[-150:].mean())
    sma200_series = close.rolling(200).mean()
    sma200 = float(sma200_series.iloc[-1])
    sma200_prev = float(sma200_series.iloc[-1 - cfg.sma200_rising_bars])

    year = df.iloc[-252:]
    low_52w = float(year["low"].min())
    high_52w = float(year["high"].max())
    pct_above_low = (price / low_52w - 1.0) * 100
    pct_below_high = (1.0 - price / high_52w) * 100
    has_rs = rs_rating is not None and not np.isnan(rs_rating)

    criteria = {
        "price_above_sma150_and_sma200": price > sma150 and price > sma200,
        "sma150_above_sma200": sma150 > sma200,
        "sma200_rising": sma200 > sma200_prev,
        "sma50_above_sma150_and_sma200": sma50 > sma150 and sma50 > sma200,
        "price_above_sma50": price > sma50,
        "price_30pct_above_52w_low": pct_above_low >= cfg.min_pct_above_52w_low,
        "price_within_25pct_of_52w_high": pct_below_high <= cfg.max_pct_below_52w_high,
        "rs_rating_strong": bool(has_rs and rs_rating >= cfg.min_rs_rating),
    }
    metrics = {
        "price": price,
        "sma50": sma50,
        "sma150": sma150,
        "sma200": sma200,
        "low_52w": low_52w,
        "high_52w": high_52w,
        "pct_above_52w_low": pct_above_low,
        "pct_below_52w_high": pct_below_high,
    }
    return TrendTemplateResult(all(criteria.values()), criteria, metrics)


# --- Whole-history versions, used by the backtest ---------------------------------
# They apply exactly the rules above to every bar at once. Each value depends only on
# data up to that bar, so walking forward through them involves no look-ahead.


def rs_rating_history(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """RS rating of every stock on every date (dates x tickers), ranked within the universe."""
    scores = {}
    for ticker, df in histories.items():
        close = df["close"]
        scores[ticker] = sum(w * (close / close.shift(bars) - 1.0) for bars, w in _RS_PERIODS)
    frame = pd.DataFrame(scores).sort_index()
    return (frame.rank(axis=1, pct=True) * 99).round().clip(lower=1.0)


def trend_template_mask(
    df: pd.DataFrame, rs_rating: pd.Series, cfg: TrendTemplateConfig = TrendTemplateConfig()
) -> np.ndarray:
    """Boolean per bar: does the stock pass the Trend Template at that bar's close?"""
    close = df["close"]
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    low_52w = df["low"].rolling(252, min_periods=1).min()
    high_52w = df["high"].rolling(252, min_periods=1).max()
    rs = rs_rating.reindex(df.index)

    passed = (
        (close > sma150) & (close > sma200)
        & (sma150 > sma200)
        & (sma200 > sma200.shift(cfg.sma200_rising_bars))
        & (sma50 > sma150) & (sma50 > sma200)
        & (close > sma50)
        & ((close / low_52w - 1.0) * 100 >= cfg.min_pct_above_52w_low)
        & ((1.0 - close / high_52w) * 100 <= cfg.max_pct_below_52w_high)
        & (rs >= cfg.min_rs_rating)
    ).to_numpy(dtype=bool, copy=True)
    passed[: cfg.min_history_bars - 1] = False
    return passed
