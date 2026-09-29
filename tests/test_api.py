from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from screener.api import app, get_service
from screener.data import DataUnavailableError
from screener.indices import Constituent
from screener.service import ScreenerService

from .conftest import LAST_LOW, PIVOT, VCP_ANCHORS, VCP_VOLUMES, make_ohlcv

IST = ZoneInfo("Asia/Kolkata")
AFTER_CLOSE = lambda: datetime(2026, 9, 16, 18, 0, tzinfo=IST)  # fixtures end on 2026-09-16
MID_SESSION = lambda: datetime(2026, 9, 16, 11, 0, tzinfo=IST)


class FakeProvider:
    def __init__(self, histories):
        self.histories = histories
        self.calls = 0

    def get_history(self, tickers, years=None):
        self.calls += 1
        self.years = years
        return {t: self.histories[t] for t in tickers if t in self.histories}


class DownProvider:
    def get_history(self, tickers, years=None):
        raise DataUnavailableError("rate limited")


@pytest.fixture
def universe(vcp_setup, vcp_breakout, vcp_extended, downtrend, sideways):
    histories = {
        "SETUP.NS": vcp_setup,
        "BRKOUT.NS": vcp_breakout,
        "EXTENDED.NS": vcp_extended,
        "NODATA.NS": None,
    }
    # Pad with laggards so the leaders rank in the top RS percentiles.
    for i in range(4):
        histories[f"DOWN{i}.NS"] = downtrend * (1 + i / 100)
        histories[f"FLAT{i}.NS"] = sideways * (1 + i / 100)
    histories = {t: df for t, df in histories.items() if df is not None}
    constituents = [Constituent(t.removesuffix(".NS"), t, f"{t} Ltd", "Test") for t in [*histories, "NODATA.NS"]]
    return histories, constituents


@pytest.fixture
def client(universe):
    histories, constituents = universe
    provider = FakeProvider(histories)
    service = ScreenerService(provider, constituents_loader=lambda spec, refresh: constituents, clock=AFTER_CLOSE)
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app), provider
    app.dependency_overrides.clear()


def test_screen_returns_buyable_stocks_with_trade_plan(client):
    http, _ = client
    resp = http.get("/api/v1/screen", params={"index": "nifty 50"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["index"] == "NIFTY50"
    assert body["summary"] == {
        "constituents": 12,
        "with_price_data": 11,
        "passed_trend_template": 3,
        "actionable_vcp": 2,
    }
    assert [r["symbol"] for r in body["results"]] == ["BRKOUT", "SETUP"]  # breakouts first
    assert body["trend_template_watchlist"] == ["EXTENDED"]

    setup = body["results"][1]
    assert setup["name"] == "SETUP.NS Ltd"
    assert setup["status"] == "SETUP"
    stop = LAST_LOW * 0.995
    assert setup["entry_type"] == "PIVOT"
    assert setup["entry_range"] == {"low": round(PIVOT, 2), "high": round(stop / 0.92, 2)}
    assert setup["reference_entry"] == setup["entry_range"]["low"]
    assert setup["stop_loss"] < setup["entry_range"]["low"] < setup["target"]
    assert body["incomplete_session_excluded"] is None
    assert body["as_of"] == "2026-09-16"
    assert setup["vcp"]["footprint"] == "12W 21/10/5 3T"

    breakout = body["results"][0]
    assert breakout["status"] == "BREAKOUT"
    assert breakout["vcp"]["breakout_volume_confirmed"] is True
    assert breakout["entry_range"]["low"] <= breakout["current_price"] <= breakout["entry_range"]["high"]
    assert breakout["reference_entry"] == breakout["current_price"]
    assert breakout["risk_pct"] <= breakout["max_risk_pct"] <= 8.01


def test_query_params_shape_the_trade_plan(client):
    http, _ = client
    body = http.get("/api/v1/screen", params={"index": "NIFTY50", "reward_to_risk": 2}).json()
    for pick in body["results"]:
        assert pick["reward_pct"] == pytest.approx(2 * pick["risk_pct"], abs=0.02)

    # Both bases need a ~5.4% stop: with a 4% risk limit neither is a trade, only a watchlist name.
    strict = http.get("/api/v1/screen", params={"index": "NIFTY50", "max_stop_pct": 4}).json()
    assert strict["results"] == []
    assert {"SETUP", "BRKOUT"} <= set(strict["trend_template_watchlist"])


def test_partial_candle_is_ignored_while_the_market_is_open(universe):
    """Mid-session the last bar is a thin-volume stub that pokes through the pivot: it must
    not be read as an unconfirmed breakout - the stock is still a setup as of yesterday."""
    histories, constituents = universe
    live = make_ohlcv(VCP_ANCHORS + [(361, 98.2)], VCP_VOLUMES + [(361, 50_000)])  # stub bar is 2026-09-16
    histories = dict(histories) | {"SETUP.NS": live}

    def run(clock):
        service = ScreenerService(FakeProvider(histories), lambda spec, refresh: constituents, clock=clock)
        body = service.screen("NIFTY50")
        return body, next(p for p in body.results if p.symbol == "SETUP")

    body, pick = run(MID_SESSION)
    assert body.incomplete_session_excluded == "2026-09-16"
    assert body.as_of == "2026-09-15"
    assert pick.status == "SETUP"

    body, pick = run(AFTER_CLOSE)
    assert body.incomplete_session_excluded is None
    assert pick.status == "BREAKOUT"
    assert pick.vcp.breakout_volume_confirmed is False


def test_cheat_entries_are_opt_in(universe):
    histories, constituents = universe
    # Sharp run-up, then a pivot that forms only half-way up the base.
    deep = make_ohlcv(
        [(0, 30), (200, 50), (300, 100), (315, 80), (330, 91), (340, 84.5), (350, 90), (356, 87), (360, 89.3)],
        VCP_VOLUMES,
    )
    histories = dict(histories) | {"SETUP.NS": deep}
    service = ScreenerService(FakeProvider(histories), lambda spec, refresh: constituents, clock=AFTER_CLOSE)
    app.dependency_overrides[get_service] = lambda: service
    try:
        http = TestClient(app)
        default = http.get("/api/v1/screen", params={"index": "NIFTY50"}).json()
        opted_in = http.get("/api/v1/screen", params={"index": "NIFTY50", "include_cheat_entries": True}).json()
    finally:
        app.dependency_overrides.clear()
    assert "SETUP" not in [r["symbol"] for r in default["results"]]
    pick = next(r for r in opted_in["results"] if r["symbol"] == "SETUP")
    assert pick["entry_type"] == "CHEAT"


def test_results_are_cached_until_refresh(client):
    http, provider = client
    http.get("/api/v1/screen", params={"index": "NIFTY50"})
    http.get("/api/v1/screen", params={"index": "NIFTY50"})
    assert provider.calls == 1
    http.get("/api/v1/screen", params={"index": "NIFTY50", "refresh": True})
    assert provider.calls == 2


def test_unknown_index_is_404(client):
    http, _ = client
    resp = http.get("/api/v1/screen", params={"index": "FOOBAR"})
    assert resp.status_code == 404
    assert "NIFTY50" in resp.json()["detail"]


def test_index_is_required(client):
    http, _ = client
    assert http.get("/api/v1/screen").status_code == 422


def test_data_outage_is_503(universe):
    _, constituents = universe
    service = ScreenerService(DownProvider(), constituents_loader=lambda spec, refresh: constituents)
    app.dependency_overrides[get_service] = lambda: service
    try:
        resp = TestClient(app).get("/api/v1/screen", params={"index": "NIFTY50"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503


def test_list_indices(client):
    http, _ = client
    keys = {i["key"] for i in http.get("/api/v1/indices").json()}
    assert {"NIFTY50", "NIFTY500", "SP500", "NASDAQ100"} <= keys
