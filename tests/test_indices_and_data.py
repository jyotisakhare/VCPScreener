import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from screener.data import YahooPriceProvider, drop_incomplete_bar, incomplete_session, split_yfinance_frame
from screener.indices import (
    UnknownIndexError,
    parse_nasdaq_api,
    parse_nse_csv,
    parse_wikipedia_constituents,
    resolve_index,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("NIFTY50", "NIFTY50"),
        ("nifty 50", "NIFTY50"),
        ("Nifty", "NIFTY50"),
        ("bank-nifty", "NIFTYBANK"),
        ("S&P 500", "SP500"),
        ("^NDX", "NASDAQ100"),
    ],
)
def test_resolve_index_accepts_loose_spelling(text, key):
    assert resolve_index(text).key == key


def test_unknown_index_lists_supported_ones():
    with pytest.raises(UnknownIndexError, match="NIFTY50"):
        resolve_index("FOOBAR")


def test_parse_nse_csv():
    stocks = parse_nse_csv((FIXTURES / "nifty50.csv").read_text())
    assert len(stocks) == 50
    reliance = next(s for s in stocks if s.symbol == "RELIANCE")
    assert reliance.ticker == "RELIANCE.NS"
    assert reliance.name.startswith("Reliance")


def test_parse_sp500_wikipedia_page():
    stocks = parse_wikipedia_constituents((FIXTURES / "sp500.html").read_text())
    assert 495 <= len(stocks) <= 510
    by_symbol = {s.symbol: s for s in stocks}
    assert by_symbol["BRK.B"].ticker == "BRK-B"  # Yahoo uses dashes for share classes
    assert by_symbol["AAPL"].sector == "Information Technology"


def test_parse_nasdaq_api():
    stocks = parse_nasdaq_api(json.loads((FIXTURES / "nasdaq100.json").read_text()))
    assert 95 <= len(stocks) <= 105
    apple = next(s for s in stocks if s.symbol == "AAPL")
    assert apple.name == "Apple Inc."


def test_split_yfinance_frame_handles_multiindex_and_missing_tickers():
    idx = pd.bdate_range("2026-01-01", periods=3)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    raw = pd.DataFrame(
        {("AAA", f): [1.0, 2.0, 3.0] for f in fields} | {("BBB", f): [float("nan")] * 3 for f in fields},
        index=idx,
    )
    out = split_yfinance_frame(raw, ["AAA", "BBB", "CCC"])
    assert list(out) == ["AAA"]  # BBB has no candles, CCC was never returned
    assert list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]


def test_prices_are_split_adjusted_only(monkeypatch):
    """Dividend-adjusted history would move pivots and stops off the real chart."""
    import yfinance

    idx = pd.bdate_range("2026-01-01", periods=2)
    raw = pd.DataFrame(
        {("AAA", "Open"): [10.0, 11], ("AAA", "High"): [10.5, 11.5], ("AAA", "Low"): [9.5, 10.5],
         ("AAA", "Close"): [10.0, 11.0], ("AAA", "Adj Close"): [9.0, 11.0], ("AAA", "Volume"): [100, 200]},
        index=idx,
    )
    seen = {}
    monkeypatch.setattr(yfinance, "download", lambda tickers, **kw: seen.update(kw) or raw)

    out = YahooPriceProvider(retries=0).get_history(["AAA"])
    assert seen["auto_adjust"] is False
    assert list(out["AAA"]["close"]) == [10.0, 11.0]  # traded prices, not "Adj Close"


@pytest.mark.parametrize(
    ("market", "now", "expected"),
    [
        ("IN", datetime(2026, 9, 16, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")), date(2026, 9, 16)),
        ("IN", datetime(2026, 9, 16, 15, 40, tzinfo=ZoneInfo("Asia/Kolkata")), date(2026, 9, 16)),  # settling
        ("IN", datetime(2026, 9, 16, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")), None),
        # 09:00 in Mumbai is still the previous evening's closed session in New York.
        ("US", datetime(2026, 9, 17, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata")), None),
        ("US", datetime(2026, 9, 16, 12, 0, tzinfo=ZoneInfo("America/New_York")), date(2026, 9, 16)),
    ],
)
def test_incomplete_session_follows_the_exchange_clock(market, now, expected):
    assert incomplete_session(market, now) == expected


def test_drop_incomplete_bar_only_touches_the_open_session():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.bdate_range(end="2026-09-16", periods=3))
    assert len(drop_incomplete_bar(df, None)) == 3
    assert len(drop_incomplete_bar(df, date(2026, 9, 17))) == 3  # today's bar has not arrived yet
    trimmed = drop_incomplete_bar(df, date(2026, 9, 16))
    assert list(trimmed["close"]) == [1.0, 2.0]
