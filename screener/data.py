"""Historical price data.

``PriceProvider`` is the seam between the screener and the outside world: the
default implementation pulls daily candles from Yahoo Finance, tests plug in
synthetic data, and a broker/vendor feed can be dropped in the same way.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataUnavailableError(RuntimeError):
    pass


class PriceProvider(Protocol):
    def get_history(self, tickers: list[str], years: float | None = None) -> dict[str, pd.DataFrame]:
        """Return daily OHLCV frames (columns: open/high/low/close/volume, oldest
        first) keyed by ticker. Tickers without data are simply left out.
        ``years`` asks for a longer history than the provider's default (backtests)."""
        ...


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price frame is missing columns: {missing}")
    df = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["high", "low", "close"])
    df = df[df["close"] > 0]
    df["volume"] = df["volume"].fillna(0)
    return df.sort_index()


# Regular-session close per market. Until then (plus a margin for the vendor to
# publish the final candle) today's bar is partial: its volume looks like a dry-up
# and a breakout looks unconfirmed, so it must not be screened as a daily candle.
MARKET_CLOSE = {
    "IN": (ZoneInfo("Asia/Kolkata"), clock_time(15, 30)),
    "US": (ZoneInfo("America/New_York"), clock_time(16, 0)),
}
CLOSE_SETTLE_MARGIN = timedelta(minutes=20)


def incomplete_session(market: str, now: datetime) -> date | None:
    """The exchange-local date whose daily candle is not final yet, if any."""
    tz, close = MARKET_CLOSE[market]
    local = now.astimezone(tz)
    if local < datetime.combine(local.date(), close, tz) + CLOSE_SETTLE_MARGIN:
        return local.date()
    return None


def drop_incomplete_bar(df: pd.DataFrame, session: date | None) -> pd.DataFrame:
    if session is None or df.empty or df.index[-1].date() < session:
        return df
    return df[[ts.date() < session for ts in df.index]]


class YahooPriceProvider:
    """Daily candles from Yahoo Finance via ``yfinance``, adjusted for splits only.

    Dividend adjustment (``auto_adjust=True``) rescales history on every ex-date, so
    pivots and stops would drift away from the prices on a real chart and at the
    broker. Yahoo's raw OHLC is already split-adjusted, which is all we need.
    """

    # yfinance.download keeps its results in module-level state, so concurrent
    # calls from parallel API requests would trample each other.
    _download_lock = threading.Lock()

    def __init__(self, period: str = "2y", chunk_size: int = 50, retries: int = 2, pause: float = 1.0):
        self.period = period
        self.chunk_size = chunk_size
        self.retries = retries
        self.pause = pause

    def get_history(self, tickers: list[str], years: float | None = None) -> dict[str, pd.DataFrame]:
        start = None if years is None else (datetime.now() - timedelta(days=round(years * 365.25))).date()
        out: dict[str, pd.DataFrame] = {}
        pending = list(dict.fromkeys(tickers))
        for attempt in range(self.retries + 1):
            if not pending:
                break
            if attempt:
                time.sleep(self.pause * 2**attempt)
            for offset in range(0, len(pending), self.chunk_size):
                out.update(self._download(pending[offset : offset + self.chunk_size], start))
            pending = [t for t in pending if t not in out]
        if pending:
            logger.warning("No price data for %d tickers: %s", len(pending), ", ".join(pending[:20]))
        if tickers and not out:
            raise DataUnavailableError(
                "Yahoo Finance returned no price data (rate limited or unreachable). Try again later."
            )
        return out

    def _download(self, tickers: list[str], start: date | None = None) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        span = {"period": self.period} if start is None else {"start": start.isoformat()}
        with self._download_lock:
            try:
                raw = yf.download(
                    tickers,
                    **span,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
            except Exception as exc:  # yfinance raises a wide range of transport errors
                logger.warning("yfinance download failed for %d tickers: %s", len(tickers), exc)
                return {}
        return split_yfinance_frame(raw, tickers)


def split_yfinance_frame(raw: pd.DataFrame | None, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                frame = raw[ticker]
            elif ticker in raw.columns.get_level_values(1):
                frame = raw.xs(ticker, axis=1, level=1)
            else:
                continue
        elif len(tickers) == 1:
            frame = raw
        else:
            continue
        try:
            frame = normalize_ohlcv(frame)
        except ValueError:
            continue
        if not frame.empty:
            out[ticker] = frame
    return out
