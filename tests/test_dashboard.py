from datetime import datetime, timezone

import pandas as pd
from fastapi.testclient import TestClient

from screener.api import app
from screener.demo import DemoPriceProvider, demo_constituents
from screener.indices import resolve_index
from screener.service import ScreenerService

from .test_api import client, universe  # noqa: F401  (fixtures)


def test_dashboard_is_served_at_the_root_and_is_self_contained():
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text
    assert "<title>Minervini VCP Screener</title>" in html
    assert "/api/v1/screen" in html and "/api/v1/indices" in html
    # no CDN or external assets: it has to work offline and behind proxies
    assert "https://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "innerHTML" not in html  # API strings reach the DOM through textContent only


def test_health_reports_live_mode_by_default():
    assert TestClient(app).get("/health").json() == {"status": "ok", "demo": False}


def test_chart_data_is_opt_in_and_covers_the_base(client):  # noqa: F811
    http, _ = client
    plain = http.get("/api/v1/screen", params={"index": "NIFTY50"}).json()
    assert all(pick["chart"] is None for pick in plain["results"])

    rich = http.get("/api/v1/screen", params={"index": "NIFTY50", "include_chart": True}).json()
    for pick in rich["results"]:
        chart = pick["chart"]
        n = len(chart["dates"])
        assert n >= 130
        assert all(len(chart[k]) == n for k in ("open", "high", "low", "close", "volume", "volume_avg_50"))
        assert chart["dates"][0] < pick["vcp"]["base_start"]  # some run-up is shown before the base
        assert chart["dates"][-1] == rich["as_of"]
        assert chart["close"][-1] == pick["current_price"]
        # every contraction the chart must draw is inside the window
        assert all(c["high_date"] in chart["dates"] and c["low_date"] in chart["dates"] for c in pick["vcp"]["contractions"])


def test_demo_market_is_deterministic_and_has_something_to_show():
    spec = resolve_index("SP500")
    constituents = demo_constituents(spec)
    assert len(constituents) == 60 and len({c.ticker for c in constituents}) == 60
    assert all("(demo)" in c.name for c in constituents)  # can never be mistaken for a real company

    provider = DemoPriceProvider()
    tickers = [c.ticker for c in constituents]
    first, again = provider.get_history(tickers), provider.get_history(tickers)
    assert all(first[t].equals(again[t]) for t in tickers)
    assert len(provider.get_history(tickers[:1], years=4.5)[tickers[0]]) > 1000  # long enough to backtest

    clock = lambda: datetime.now(timezone.utc) + pd.Timedelta(days=1)  # well after the last demo session closed
    result = ScreenerService(provider, demo_constituents, clock=clock).screen("SP500", include_chart=True)
    statuses = {p.status for p in result.results}
    assert {"SETUP", "BREAKOUT"} <= statuses
    assert result.trend_template_watchlist
