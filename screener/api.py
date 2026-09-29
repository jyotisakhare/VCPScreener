"""FastAPI application.

Run with:  uvicorn screener.api:app --reload
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import __version__
from .backtest import BacktestConfig
from .config import ScreenerConfig
from .data import DataUnavailableError
from .indices import INDICES, ConstituentsFetchError, UnknownIndexError
from .schemas import BacktestResponse, IndexInfo, ScreenResponse
from .service import BacktestService, ScreenerService

app = FastAPI(
    title="Minervini VCP Screener",
    version=__version__,
    description=(
        "Screens the constituents of a stock-market index with Mark Minervini's Trend Template "
        "and Volatility Contraction Pattern, and returns buyable stocks with entry range, "
        "stop-loss and target."
    ),
)

# Demo mode (``python -m screener serve --demo``): synthetic prices and made-up companies,
# for trying the dashboard without market-data access. Never mixed with real data.
DEMO_MODE = os.environ.get("SCREENER_DEMO") == "1"
DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"

if DEMO_MODE:
    from .demo import DemoPriceProvider, demo_constituents

    _demo_provider = DemoPriceProvider()
    _service = ScreenerService(_demo_provider, demo_constituents)
    _backtest_service = BacktestService(_demo_provider, demo_constituents)
else:
    _service = ScreenerService()
    _backtest_service = BacktestService()


def get_service() -> ScreenerService:
    return _service


def get_backtest_service() -> BacktestService:
    return _backtest_service


@app.get("/api/v1/indices", response_model=list[IndexInfo], summary="List supported indices")
def list_indices() -> list[IndexInfo]:
    return [IndexInfo(key=s.key, name=s.name, market=s.market) for s in INDICES.values()]


# Declared with plain ``def`` so FastAPI runs the (blocking) screen in its thread pool.
@app.get("/api/v1/screen", response_model=ScreenResponse, summary="Screen an index for buyable VCP setups")
def screen(
    service: Annotated[ScreenerService, Depends(get_service)],
    index: Annotated[str, Query(description="Index to screen, e.g. NIFTY50, NIFTY500, SP500, NASDAQ100")],
    reward_to_risk: Annotated[float, Query(ge=1, le=10, description="Target as a multiple of risk")] = 3.0,
    max_stop_pct: Annotated[
        float,
        Query(gt=3, le=15, description="Maximum risk from the purchase price, %. Setups needing a wider stop are skipped"),
    ] = 8.0,
    min_rs_rating: Annotated[float, Query(ge=0, le=99, description="Minimum relative-strength percentile")] = 70.0,
    max_dry_up_ratio: Annotated[
        float,
        Query(gt=0, le=2, description="Volume dry-up: 10-day average volume must be at most this fraction of the 50-day average"),
    ] = 0.7,
    include_cheat_entries: Annotated[
        bool, Query(description="Also return pivots in the middle/lower third of the base (earlier, riskier entries)")
    ] = False,
    include_chart: Annotated[bool, Query(description="Attach daily candles to every pick (used by the dashboard)")] = False,
    refresh: Annotated[bool, Query(description="Bypass the 15-minute result cache")] = False,
) -> ScreenResponse:
    base = ScreenerConfig()
    config = replace(
        base,
        trend=replace(base.trend, min_rs_rating=min_rs_rating),
        vcp=replace(base.vcp, allow_cheat_entries=include_cheat_entries, max_dry_up_ratio=max_dry_up_ratio),
        plan=replace(base.plan, reward_to_risk=reward_to_risk, max_stop_pct=max_stop_pct),
    )
    try:
        return service.screen(index, config, refresh=refresh, include_chart=include_chart)
    except UnknownIndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConstituentsFetchError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch index constituents: {exc}") from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/v1/backtest",
    response_model=BacktestResponse,
    summary="Backtest the strategy on an index",
    description=(
        "Walk-forward simulation: every evening the screen runs on data up to that day only, SETUPs become "
        "buy-stop orders at the pivot for the next session, BREAKOUTs are bought at the next open, and positions "
        "exit at the stop, the target or a time stop. Slow on a cold cache: a 50-stock index takes seconds, "
        "a 500-stock index several minutes. Results are cached for 6 hours."
    ),
)
def backtest(
    service: Annotated[BacktestService, Depends(get_backtest_service)],
    index: Annotated[str, Query(description="Index to backtest, e.g. NIFTY50, NIFTY200, SP500")],
    years: Annotated[float, Query(ge=0.5, le=15, description="Length of the test period")] = 3.0,
    initial_capital: Annotated[float, Query(gt=0)] = 1_000_000.0,
    risk_per_trade_pct: Annotated[float, Query(gt=0, le=10, description="Equity lost if a stop is hit")] = 1.0,
    max_positions: Annotated[int, Query(ge=1, le=100)] = 10,
    max_position_pct: Annotated[float, Query(gt=0, le=100, description="Largest single position, % of equity")] = 20.0,
    max_holding_days: Annotated[int, Query(ge=1, le=500, description="Time stop, in trading days")] = 60,
    cost_pct: Annotated[float, Query(ge=0, le=5, description="Commission + slippage per side, %")] = 0.1,
    reward_to_risk: Annotated[float, Query(ge=1, le=10)] = 3.0,
    max_stop_pct: Annotated[float, Query(gt=3, le=15)] = 8.0,
    min_rs_rating: Annotated[float, Query(ge=0, le=99)] = 70.0,
    max_dry_up_ratio: Annotated[float, Query(gt=0, le=2)] = 0.7,
    include_cheat_entries: bool = False,
    include_trades: Annotated[bool, Query(description="Return the full trade list")] = True,
    refresh: bool = False,
) -> BacktestResponse:
    base = ScreenerConfig()
    config = replace(
        base,
        trend=replace(base.trend, min_rs_rating=min_rs_rating),
        vcp=replace(base.vcp, allow_cheat_entries=include_cheat_entries, max_dry_up_ratio=max_dry_up_ratio),
        plan=replace(base.plan, reward_to_risk=reward_to_risk, max_stop_pct=max_stop_pct),
    )
    bt = BacktestConfig(
        years=years,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        max_positions=max_positions,
        max_position_pct=max_position_pct,
        max_holding_bars=max_holding_days,
        cost_pct=cost_pct,
    )
    try:
        return service.run(index, config, bt, include_trades=include_trades, refresh=refresh)
    except UnknownIndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConstituentsFetchError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch index constituents: {exc}") from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD, media_type="text/html")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str | bool]:
    return {"status": "ok", "demo": DEMO_MODE}
