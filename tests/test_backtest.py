from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from screener.api import app, get_backtest_service
from screener.backtest import (
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    BacktestConfig,
    Trade,
    _check_exit,
    _Order,
    _Position,
    _try_fill,
    curve_metrics,
    equal_weight_buy_and_hold,
    generate_trades,
    run_portfolio,
    simulate_ticker,
    trade_statistics,
)
from screener.config import ScreenerConfig
from screener.indices import Constituent
from screener.service import BacktestService
from screener.trend_template import evaluate_trend_template, rs_rating_history, rs_ratings, trend_template_mask

from .conftest import LAST_LOW, VCP_ANCHORS, VCP_VOLUMES, make_ohlcv
from .test_api import FakeProvider

STOP = round(LAST_LOW * 0.995, 2)
LOOSE_VOLUME = replace(ScreenerConfig(), vcp=replace(ScreenerConfig().vcp, max_dry_up_ratio=9, max_quietest_day_ratio=9))


def random_universe(tickers=14, bars=760, seed=5):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end="2026-09-16", periods=bars)
    out = {}
    for i in range(tickers):
        vol = rng.uniform(0.012, 0.025)
        close = 100 * np.exp(np.cumsum(rng.normal(rng.normal(0.0008, 0.0006), vol, bars)))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, bars))
        wick = np.abs(rng.normal(0, vol * 0.6, bars))
        out[f"R{i}"] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1 + wick),
                "low": np.minimum(open_, close) * (1 - wick),
                "close": close,
                "volume": rng.lognormal(13, 0.4, bars),
            },
            index=index,
        )
    return out


# --- the vectorised screen must agree with the one the live API uses ------------------


def test_trend_template_mask_matches_the_point_in_time_function(vcp_setup, downtrend, sideways):
    for df in (vcp_setup, downtrend, sideways):
        rs = pd.Series(np.linspace(40, 99, len(df)), index=df.index)
        mask = trend_template_mask(df, rs)
        for bar in (100, 221, 222, 250, 300, 320, 345, len(df) - 1):
            expected = evaluate_trend_template(df.iloc[: bar + 1], rs.iloc[bar]).passed
            assert mask[bar] == expected, bar
    assert trend_template_mask(vcp_setup, pd.Series(99.0, index=vcp_setup.index)).any()


def test_rs_rating_history_matches_the_point_in_time_function():
    universe = random_universe(tickers=9, bars=400)
    history = rs_rating_history(universe)
    for bar in (260, 333, 399):
        expected = rs_ratings({t: df.iloc[: bar + 1] for t, df in universe.items()})
        assert history.iloc[bar].to_dict() == expected
    assert history.iloc[:252].isna().all().all()  # no rating before a year of data exists


# --- order fills and exits ------------------------------------------------------------


def order(kind="SETUP"):
    return _Order(kind, "PIVOT", "8W 20/5 2T", 90.0, signal_bar=0, pivot=100.0, entry_high=103.0, stop=95.0)


@pytest.mark.parametrize(
    ("kind", "open_", "high", "fill"),
    [
        ("SETUP", 99.0, 101.0, 100.0),  # trades up through the pivot: buy-stop fills at the pivot
        ("SETUP", 99.0, 99.9, None),  # never reached the pivot
        ("SETUP", 101.5, 102.0, 101.5),  # gapped over the pivot: filled at the open
        ("SETUP", 103.5, 104.0, None),  # gapped beyond the buy range: no chase
        ("BREAKOUT", 101.0, 102.0, 101.0),  # bought at the open
        ("BREAKOUT", 99.5, 102.0, None),  # fell back under the pivot overnight
        ("BREAKOUT", 104.0, 105.0, None),
    ],
)
def test_try_fill(kind, open_, high, fill):
    assert _try_fill(order(kind), open_, high) == fill


def test_exit_rules_are_pessimistic():
    pos = _Position(order(), entry_bar=10, entry=100.0, target=115.0)
    check = lambda bar, o, h, l, c, last=False: _check_exit(pos, bar, o, h, l, c, last, 60)
    assert check(12, 101, 102, 100, 101) is None
    assert check(12, 101, 102, 94, 96) == (95.0, EXIT_STOP)
    assert check(12, 93, 94, 92, 93) == (93.0, EXIT_STOP)  # gap down: filled at the open, worse than the stop
    assert check(12, 110, 116, 109, 115) == (115.0, EXIT_TARGET)
    assert check(12, 117, 118, 116, 117) == (117.0, EXIT_TARGET)  # gap up: filled at the open
    assert check(12, 100, 116, 94, 100) == (95.0, EXIT_STOP)  # both inside one bar: assume the stop came first
    assert check(70, 101, 102, 100, 101.5) == (101.5, EXIT_TIME)
    assert check(12, 101, 102, 100, 101.5, True)[1] == "END_OF_DATA"
    # On the entry bar the open happened before the fill, so it cannot be a gap exit.
    assert check(10, 94, 101, 93.5, 100) == (95.0, EXIT_STOP)


# --- trades -----------------------------------------------------------------------------


def run_one(df, bt=BacktestConfig()):
    everything = np.ones(len(df), dtype=bool)
    return simulate_ticker("T", df, everything, np.full(len(df), 95.0), ScreenerConfig(), bt, start_bar=300)


def test_winning_trade_is_bought_through_the_pivot_and_sold_at_the_target():
    df = make_ohlcv(VCP_ANCHORS + [(362, 99.5), (400, 125)], VCP_VOLUMES + [(361, 3_000_000), (400, 2_000_000)])
    (trade,) = run_one(df)
    assert trade.entry_kind == "SETUP"
    assert trade.signal_date == df.index[360].date()  # seen at the close of bar 360...
    assert trade.entry_date == df.index[361].date()  # ...bought the next session
    assert trade.entry_price == pytest.approx(97.75)  # opened above the 97.48 pivot: filled at the open
    assert trade.stop_loss == STOP
    assert trade.target == pytest.approx(97.75 + 3 * (97.75 - STOP))
    assert trade.exit_reason == EXIT_TARGET
    # This synthetic series opens each bar at its close, so the target is crossed by a small
    # gap and the sale is filled at that open - never below the target.
    assert trade.target <= trade.exit_price <= trade.target * 1.002
    assert trade.r_multiple == pytest.approx(3.0, abs=0.06)  # ~3R less two lots of costs


def test_losing_trade_is_stopped_out_for_about_one_r():
    df = make_ohlcv(VCP_ANCHORS + [(362, 99.5), (370, 88)], VCP_VOLUMES + [(361, 3_000_000), (370, 2_000_000)])
    trades = run_one(df)
    assert trades[0].exit_reason == EXIT_STOP
    assert trades[0].exit_price == STOP
    assert trades[0].r_multiple == pytest.approx(-1.0, abs=0.06)
    assert trades[0].r_multiple < -1.0  # costs make a full stop slightly worse than -1R


def test_time_stop_closes_a_trade_that_goes_nowhere():
    df = make_ohlcv(VCP_ANCHORS + [(362, 99.5), (420, 100.5)], VCP_VOLUMES + [(361, 3_000_000), (420, 1_000_000)])
    (trade,) = run_one(df, BacktestConfig(max_holding_bars=20))
    assert trade.exit_reason == EXIT_TIME
    assert trade.bars_held == 20


def test_no_look_ahead_future_data_never_changes_past_trades():
    """Cutting off the future must leave every trade that was already closed untouched."""
    universe = random_universe(tickers=40, seed=1)
    bt = BacktestConfig()
    start = universe["R0"].index[300]
    full = generate_trades(universe, LOOSE_VOLUME, bt, start)
    assert len(full) >= 15, "test universe should trade"

    cut = 600
    cutoff = universe["R0"].index[cut - 1].date()
    truncated = generate_trades({t: df.iloc[:cut] for t, df in universe.items()}, LOOSE_VOLUME, bt, start)

    closed_before = lambda trades: [t for t in trades if t.exit_date < cutoff]
    assert closed_before(full) == closed_before(truncated)
    assert len(closed_before(full)) >= 10
    # and every entry comes strictly after the evening its signal was generated
    assert all(t.entry_date > t.signal_date for t in full)


# --- portfolio ----------------------------------------------------------------------------


def make_trade(ticker, entry_day, exit_day, entry, stop, exit_price, rs=90.0):
    days = pd.bdate_range("2026-01-01", periods=30)
    net = exit_price * 0.999 / (entry * 1.001) - 1
    return Trade(
        ticker, days[entry_day - 1].date(), days[entry_day].date(), "SETUP", "PIVOT", "8W 20/5 2T", rs,
        entry, stop, entry + 3 * (entry - stop), days[exit_day].date(), exit_price,
        EXIT_TARGET if exit_price > entry else EXIT_STOP, exit_day - entry_day, net * 100, net / ((entry - stop) / entry),
    )


def flat_closes(tickers, price=100.0):
    return pd.DataFrame({t: price for t in tickers}, index=pd.bdate_range("2026-01-01", periods=30))


def test_position_size_risks_a_fixed_share_of_equity():
    bt = BacktestConfig(initial_capital=1_000_000, risk_per_trade_pct=1.0, max_position_pct=50, cost_pct=0.0)
    trade = make_trade("A", 2, 10, entry=100.0, stop=95.0, exit_price=95.0)
    result = run_portfolio([trade], flat_closes(["A"]), bt)
    # 1% of 1,000,000 at risk with a 5-point stop = 2,000 shares; stopped out = -10,000
    assert result.pnl == [pytest.approx(-10_000)]
    assert result.equity.iloc[-1] == pytest.approx(990_000)
    assert result.invested_fraction.max() == pytest.approx(0.2)


def test_position_size_is_capped():
    bt = BacktestConfig(risk_per_trade_pct=1.0, max_position_pct=10, cost_pct=0.0)
    tight = make_trade("A", 2, 10, entry=100.0, stop=99.0, exit_price=103.0)  # 1-point stop wants 100% of equity
    result = run_portfolio([tight], flat_closes(["A"]), bt)
    assert result.pnl == [pytest.approx(1000 * 3.0)]  # capped at 10% = 1,000 shares


def test_position_limit_gives_the_slot_to_the_strongest_stock_and_costs_are_charged():
    bt = BacktestConfig(max_positions=1, cost_pct=0.1)
    weak = make_trade("WEAK", 2, 10, 100.0, 95.0, 115.0, rs=75)
    strong = make_trade("STRONG", 2, 10, 100.0, 95.0, 115.0, rs=98)
    later = make_trade("LATER", 12, 20, 100.0, 95.0, 95.0)
    result = run_portfolio([weak, strong, later], flat_closes(["WEAK", "STRONG", "LATER"]), bt)
    assert [t.ticker for t in result.taken] == ["STRONG", "LATER"]
    assert result.skipped_no_slot == 1
    assert result.equity.iloc[-1] == pytest.approx(1_000_000 + sum(result.pnl))
    assert result.pnl[0] < 2000 * 15  # commission + slippage came out of the gross 30,000


def test_curve_metrics_on_a_known_curve():
    index = pd.to_datetime(["2025-01-01", "2025-07-01", "2026-01-01"])
    m = curve_metrics(pd.Series([120.0, 90.0, 121.0], index=index), initial_capital=100.0)
    assert m["total_return_pct"] == 21.0
    assert m["cagr_pct"] == pytest.approx(21.0, abs=0.1)
    assert m["max_drawdown_pct"] == -25.0
    assert (m["max_drawdown_peak"], m["max_drawdown_trough"]) == ("2025-01-01", "2025-07-01")
    assert m["calmar_ratio"] == pytest.approx(0.84, abs=0.01)


def test_trade_statistics_on_known_trades():
    trades = [
        make_trade("A", 2, 5, 100, 95, 115),
        make_trade("B", 3, 6, 100, 95, 95),
        make_trade("C", 4, 7, 100, 95, 95),
        make_trade("D", 5, 9, 100, 95, 115),
    ]
    s = trade_statistics(trades)
    assert (s["total_trades"], s["winners"], s["losers"], s["win_rate_pct"]) == (4, 2, 2, 50.0)
    assert s["payoff_ratio"] == pytest.approx(14.77 / 5.19, abs=0.02)
    assert s["profit_factor"] == s["payoff_ratio"]  # equal counts of wins and losses
    assert s["expectancy_r"] == pytest.approx((2.95 * 2 - 1.04 * 2) / 4, abs=0.02)
    assert s["max_consecutive_losses"] == 2
    assert s["by_exit_reason"]["STOP"]["trades"] == 2
    assert trade_statistics([]) == {"total_trades": 0}


def test_benchmark_is_equal_weight_buy_and_hold():
    closes = pd.DataFrame({"A": [10.0, 20.0], "B": [10.0, 10.0], "LATE": [np.nan, 5.0]}, index=pd.bdate_range("2026-01-01", periods=2))
    assert list(equal_weight_buy_and_hold(closes, 1000.0)) == [1000.0, 1500.0]  # LATE was not listed on day one


# --- API ---------------------------------------------------------------------------------


@pytest.fixture
def backtest_client(downtrend, sideways):
    winner = make_ohlcv(VCP_ANCHORS + [(362, 99.5), (400, 125)], VCP_VOLUMES + [(361, 3_000_000), (400, 2_000_000)])
    loser = make_ohlcv(VCP_ANCHORS + [(362, 99.5), (370, 88), (400, 80)], VCP_VOLUMES + [(361, 3_000_000), (400, 2_000_000)])
    histories = {"WIN.NS": winner, "LOSE.NS": loser}
    for i in range(4):  # laggards, so the two leaders carry a high RS rating
        histories[f"DOWN{i}.NS"] = make_ohlcv([(0, 100 + i), (400, 55 + i)], [(0, 1e6), (400, 1e6)])
        histories[f"FLAT{i}.NS"] = make_ohlcv([(0, 50 + i), (200, 53 + i), (400, 51 + i)], [(0, 1e6), (400, 1e6)])
    constituents = [Constituent(t.removesuffix(".NS"), t, f"{t} Ltd") for t in histories]
    provider = FakeProvider(histories)
    clock = lambda: datetime(2026, 9, 16, 18, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    service = BacktestService(provider, lambda spec, refresh: constituents, clock=clock)
    app.dependency_overrides[get_backtest_service] = lambda: service
    yield TestClient(app), provider
    app.dependency_overrides.clear()


def test_backtest_endpoint_reports_the_headline_numbers(backtest_client):
    http, provider = backtest_client
    resp = http.get("/api/v1/backtest", params={"index": "nifty 50", "years": 1, "initial_capital": 500_000})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert provider.years == pytest.approx(2.5)  # test period + warm-up for the 200-day average and RS rating
    assert body["index"] == "NIFTY50"
    assert body["period"]["end"] == "2026-09-16"

    stats = body["trade_stats"]
    assert (stats["total_trades"], stats["winners"], stats["losers"]) == (2, 1, 1)
    assert stats["win_rate_pct"] == 50.0
    assert stats["expectancy_r"] == pytest.approx((2.96 - 1.04) / 2, abs=0.05)
    assert stats["profit_factor"] > 2
    assert set(stats["by_exit_reason"]) == {"STOP", "TARGET"}

    # 1% risk per trade: +~2.96R and -~1.04R on 500,000
    portfolio = body["portfolio"]
    assert portfolio["trades_taken"] == 2
    assert portfolio["net_profit"] == pytest.approx(500_000 * 0.01 * 1.92, rel=0.05)
    assert portfolio["final_equity"] == pytest.approx(500_000 + portfolio["net_profit"])
    assert portfolio["max_drawdown_pct"] < 0
    assert body["benchmark"]["total_return_pct"] is not None

    assessment = body["assessment"]
    assert assessment["profitable"] and assessment["positive_expectancy"]
    assert assessment["sample_size_adequate"] is False
    assert "too small a sample" in assessment["summary"]

    assert {t["symbol"]: t["exit_reason"] for t in body["trades"]} == {"WIN": "TARGET", "LOSE": "STOP"}
    assert all(t["taken_in_portfolio"] for t in body["trades"])
    assert body["equity_curve"][-1]["strategy"] == portfolio["final_equity"]
    assert any("Survivorship" in c for c in body["caveats"])


def test_backtest_endpoint_options_and_errors(backtest_client):
    http, provider = backtest_client
    lean = http.get("/api/v1/backtest", params={"index": "NIFTY50", "years": 1, "initial_capital": 500_000, "include_trades": False})
    assert lean.json()["trades"] is None
    assert provider.calls == 1  # same settings: served from cache

    tighter = http.get("/api/v1/backtest", params={"index": "NIFTY50", "years": 1, "max_stop_pct": 4}).json()
    assert tighter["trade_stats"]["total_trades"] == 0
    assert "no trades" in tighter["assessment"]["summary"]
    assert tighter["portfolio"]["total_return_pct"] == 0

    assert http.get("/api/v1/backtest", params={"index": "FOOBAR"}).status_code == 404
    assert http.get("/api/v1/backtest", params={"index": "NIFTY50", "years": 99}).status_code == 422
