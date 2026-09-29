"""Screening pipeline: index -> constituents -> Trend Template -> VCP -> trade plan."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from .backtest import (
    WARMUP_YEARS,
    BacktestConfig,
    close_matrix,
    curve_metrics,
    equal_weight_buy_and_hold,
    generate_trades,
    run_portfolio,
    trade_statistics,
    yearly_breakdown,
    finite_round,
)
from .config import ScreenerConfig
from .data import (
    DataUnavailableError,
    PriceProvider,
    YahooPriceProvider,
    drop_incomplete_bar,
    incomplete_session,
)
from .indices import Constituent, IndexSpec, get_constituents, resolve_index
from .schemas import (
    Assessment,
    BacktestResponse,
    ContractionOut,
    EntryRange,
    PriceChart,
    ScreenResponse,
    ScreenSummary,
    StockPick,
    VCPDetails,
)
from .trade_plan import plan_best_candidate
from .trend_template import evaluate_trend_template, rs_ratings
from .vcp import STATUS_BREAKOUT

logger = logging.getLogger(__name__)

RESULT_TTL_SECONDS = 15 * 60
DISCLAIMER = (
    "Algorithmic pattern screen for educational use, not investment advice. "
    "Confirm the chart, liquidity, earnings dates and overall market trend before trading."
)


class ScreenerService:
    def __init__(
        self,
        provider: PriceProvider | None = None,
        constituents_loader: Callable[[IndexSpec, bool], list[Constituent]] = get_constituents,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.provider = provider or YahooPriceProvider()
        self._load_constituents = constituents_loader
        self._clock = clock
        self._cache: dict[tuple, tuple[float, ScreenResponse]] = {}
        self._lock = threading.Lock()

    def screen(
        self,
        index: str,
        config: ScreenerConfig = ScreenerConfig(),
        refresh: bool = False,
        include_chart: bool = False,
    ) -> ScreenResponse:
        spec = resolve_index(index)
        cache_key = (spec.key, config)
        with self._lock:
            hit = self._cache.get(cache_key)
        if hit and not refresh and time.time() - hit[0] < RESULT_TTL_SECONDS:
            response = hit[1]
        else:
            response = self._run(spec, config, refresh)
            with self._lock:
                self._cache[cache_key] = (time.time(), response)
        if include_chart:
            return response
        return response.model_copy(update={"results": [p.model_copy(update={"chart": None}) for p in response.results]})

    def _run(self, spec: IndexSpec, config: ScreenerConfig, refresh: bool) -> ScreenResponse:
        constituents = self._load_constituents(spec, refresh)
        histories = self.provider.get_history([c.ticker for c in constituents])

        # Screen completed daily candles only; a half-finished session distorts volume.
        now = self._clock()
        open_session = incomplete_session(spec.market, now)
        trimmed = {t: drop_incomplete_bar(df, open_session) for t, df in histories.items()}
        session_excluded = any(len(trimmed[t]) != len(histories[t]) for t in histories)
        histories = trimmed
        ratings = rs_ratings(histories)

        picks: list[StockPick] = []
        watchlist: list[str] = []
        passed = 0
        as_of = None
        for stock in constituents:
            df = histories.get(stock.ticker)
            if df is None or df.empty:
                continue
            last_bar = df.index[-1].date().isoformat()
            as_of = max(as_of, last_bar) if as_of else last_bar

            trend = evaluate_trend_template(df, ratings.get(stock.ticker), config.trend)
            if not trend.passed:
                continue
            passed += 1

            signal = plan_best_candidate(df, config)
            if signal is None:
                watchlist.append(stock.symbol)
                continue
            vcp, plan = signal
            picks.append(
                StockPick(
                    symbol=stock.symbol,
                    ticker=stock.ticker,
                    name=stock.name,
                    sector=stock.sector,
                    status=vcp.status,
                    entry_type=vcp.entry_type,
                    current_price=round(trend.metrics["price"], 2),
                    pct_from_pivot=round(vcp.pct_from_pivot, 2),
                    entry_range=EntryRange(low=plan.entry_low, high=plan.entry_high),
                    reference_entry=plan.reference_entry,
                    stop_loss=plan.stop_loss,
                    target=plan.target,
                    risk_pct=plan.risk_pct,
                    max_risk_pct=plan.max_risk_pct,
                    reward_pct=plan.reward_pct,
                    reward_to_risk=plan.reward_to_risk,
                    stop_basis=plan.stop_basis,
                    rs_rating=ratings[stock.ticker],
                    pct_below_52w_high=round(trend.metrics["pct_below_52w_high"], 2),
                    vcp=VCPDetails(
                        footprint=vcp.footprint,
                        contractions=[
                            ContractionOut(
                                high_date=c.high_date,
                                high=round(c.high, 2),
                                low_date=c.low_date,
                                low=round(c.low, 2),
                                depth_pct=round(c.depth_pct, 2),
                            )
                            for c in vcp.contractions
                        ],
                        base_start=vcp.base_start,
                        base_weeks=round(vcp.base_bars / 5, 1),
                        pivot_position_pct=round(vcp.pivot_position * 100, 1),
                        volume_dry_up_ratio=round(vcp.volume_dry_up_ratio, 2),
                        quietest_day_ratio=round(vcp.quietest_day_ratio, 2),
                        breakout_bars_ago=vcp.breakout_bars_ago,
                        breakout_volume_ratio=(
                            round(vcp.breakout_volume_ratio, 2) if vcp.breakout_volume_ratio is not None else None
                        ),
                        breakout_volume_confirmed=vcp.breakout_volume_confirmed,
                    ),
                    chart=_price_chart(df, vcp.base_start),
                )
            )

        # Confirmed breakouts first, then the strongest relative strength.
        picks.sort(key=lambda p: (p.status != STATUS_BREAKOUT, -p.rs_rating, p.symbol))
        logger.info(
            "%s: %d constituents, %d with data, %d trend template, %d VCP",
            spec.key, len(constituents), len(histories), passed, len(picks),
        )
        return ScreenResponse(
            index=spec.key,
            index_name=spec.name,
            market=spec.market,
            as_of=as_of,
            incomplete_session_excluded=open_session.isoformat() if session_excluded else None,
            generated_at=now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            summary=ScreenSummary(
                constituents=len(constituents),
                with_price_data=len(histories),
                passed_trend_template=passed,
                actionable_vcp=len(picks),
            ),
            results=picks,
            trend_template_watchlist=watchlist,
            disclaimer=DISCLAIMER,
        )


CHART_BARS_BEFORE_BASE = 45
CHART_MIN_BARS = 130


def _price_chart(df: pd.DataFrame, base_start: str) -> PriceChart:
    first = int(df.index.searchsorted(pd.Timestamp(base_start))) - CHART_BARS_BEFORE_BASE
    window = df.iloc[max(0, min(first, len(df) - CHART_MIN_BARS)) :]
    average = df["volume"].rolling(50).mean().reindex(window.index)
    return PriceChart(
        dates=[ts.date().isoformat() for ts in window.index],
        open=window["open"].round(2).tolist(),
        high=window["high"].round(2).tolist(),
        low=window["low"].round(2).tolist(),
        close=window["close"].round(2).tolist(),
        volume=window["volume"].round().tolist(),
        volume_avg_50=[None if pd.isna(v) else round(v) for v in average],
    )


BACKTEST_TTL_SECONDS = 6 * 60 * 60
MIN_TRADES_FOR_CONFIDENCE = 30
BACKTEST_CAVEATS = [
    "Survivorship bias: the test uses today's index members. Stocks that were dropped or delisted "
    "during the period are missing, which flatters both the strategy and the benchmark.",
    "Daily candles cannot show the order of events inside a day. Gaps through the stop are filled at the open "
    "and a bar containing both stop and target counts as a loss, but real fills (circuit limits, illiquid "
    "opens, partial fills) can still be worse.",
    "Costs are a flat percentage per side; taxes and market impact are not modelled. Dividends are excluded "
    "for both the strategy and the benchmark.",
    "The RS rating ranks stocks within the index, not the whole market, and the rules ignore fundamentals, "
    "earnings dates and the general market trend that Minervini also uses.",
    "Fractional shares are allowed and one run is one sample path: a different start date, position limit "
    "or cost assumption can change the outcome. Past performance does not predict future results.",
]


class BacktestService:
    def __init__(
        self,
        provider: PriceProvider | None = None,
        constituents_loader: Callable[[IndexSpec, bool], list[Constituent]] = get_constituents,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.provider = provider or YahooPriceProvider()
        self._load_constituents = constituents_loader
        self._clock = clock
        self._cache: dict[tuple, tuple[float, BacktestResponse]] = {}
        self._lock = threading.Lock()

    def run(
        self,
        index: str,
        config: ScreenerConfig = ScreenerConfig(),
        bt: BacktestConfig = BacktestConfig(),
        include_trades: bool = True,
        refresh: bool = False,
    ) -> BacktestResponse:
        spec = resolve_index(index)
        cache_key = (spec.key, config, bt)
        with self._lock:
            hit = self._cache.get(cache_key)
        if hit and not refresh and time.time() - hit[0] < BACKTEST_TTL_SECONDS:
            response = hit[1]
        else:
            response = self._run(spec, config, bt, refresh)
            with self._lock:
                self._cache[cache_key] = (time.time(), response)
        return response if include_trades else response.model_copy(update={"trades": None})

    def _run(self, spec: IndexSpec, config: ScreenerConfig, bt: BacktestConfig, refresh: bool) -> BacktestResponse:
        constituents = self._load_constituents(spec, refresh)
        histories = self.provider.get_history([c.ticker for c in constituents], years=bt.years + WARMUP_YEARS)
        now = self._clock()
        open_session = incomplete_session(spec.market, now)
        histories = {t: drop_incomplete_bar(df, open_session) for t, df in histories.items()}
        histories = {t: df for t, df in histories.items() if not df.empty}
        if not histories:
            raise DataUnavailableError("No price history available for this index.")

        end = max(df.index[-1] for df in histories.values())
        start = end - pd.Timedelta(days=round(bt.years * 365.25))
        usable = {t: df for t, df in histories.items() if len(df) >= config.trend.min_history_bars}

        trades = generate_trades(usable, config, bt, start)
        closes = close_matrix(histories, start)
        portfolio = run_portfolio(trades, closes, bt)
        benchmark = equal_weight_buy_and_hold(closes, bt.initial_capital)

        stats = trade_statistics(trades)
        strategy_curve = curve_metrics(portfolio.equity, bt.initial_capital)
        benchmark_curve = curve_metrics(benchmark, bt.initial_capital)
        gross_profit = sum(p for p in portfolio.pnl if p > 0)
        gross_loss = -sum(p for p in portfolio.pnl if p < 0)

        symbols = {c.ticker: c.symbol for c in constituents}
        taken = {id(t) for t in portfolio.taken}
        points = pd.DataFrame({"s": portfolio.equity, "b": benchmark}).groupby(
            [portfolio.equity.index.year, portfolio.equity.index.month]
        ).tail(1)

        return BacktestResponse(
            index=spec.key,
            index_name=spec.name,
            market=spec.market,
            generated_at=now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            period={
                "start": closes.index[0].date().isoformat(),
                "end": closes.index[-1].date().isoformat(),
                "trading_days": len(closes),
                "years": round((closes.index[-1] - closes.index[0]).days / 365.25, 2),
            },
            universe={
                "constituents": len(constituents),
                "with_price_data": len(histories),
                "with_enough_history": len(usable),
            },
            settings={
                "initial_capital": bt.initial_capital,
                "risk_per_trade_pct": bt.risk_per_trade_pct,
                "max_positions": bt.max_positions,
                "max_position_pct": bt.max_position_pct,
                "max_holding_bars": bt.max_holding_bars,
                "cost_pct_per_side": bt.cost_pct,
                "reward_to_risk": config.plan.reward_to_risk,
                "max_stop_pct": config.plan.max_stop_pct,
                "min_rs_rating": config.trend.min_rs_rating,
                "max_dry_up_ratio": config.vcp.max_dry_up_ratio,
                "include_cheat_entries": config.vcp.allow_cheat_entries,
            },
            assessment=_assess(stats, strategy_curve, benchmark_curve),
            trade_stats=stats,
            portfolio={
                **strategy_curve,
                "initial_capital": bt.initial_capital,
                "trades_taken": len(portfolio.taken),
                "trades_skipped_no_free_slot": portfolio.skipped_no_slot,
                "trades_skipped_no_cash": portfolio.skipped_no_cash,
                "net_profit": finite_round(portfolio.equity.iloc[-1] - bt.initial_capital),
                "profit_factor": finite_round(gross_profit / gross_loss) if gross_loss > 0 else None,
                "avg_exposure_pct": finite_round(portfolio.invested_fraction.mean() * 100),
                "max_exposure_pct": finite_round(portfolio.invested_fraction.max() * 100),
            },
            benchmark={**benchmark_curve, "name": f"Equal-weight buy & hold of the current {spec.name} constituents"},
            yearly_returns=yearly_breakdown(portfolio.equity, benchmark, bt.initial_capital, portfolio.taken),
            equity_curve=[
                {"date": ts.date().isoformat(), "strategy": round(row.s, 2), "benchmark": round(row.b, 2)}
                for ts, row in points.iterrows()
            ],
            trades=[
                {
                    "symbol": symbols.get(t.ticker, t.ticker),
                    "ticker": t.ticker,
                    "signal_date": t.signal_date.isoformat(),
                    "entry_date": t.entry_date.isoformat(),
                    "entry_kind": t.entry_kind,
                    "entry_type": t.entry_type,
                    "footprint": t.footprint,
                    "rs_rating": t.rs_rating,
                    "entry_price": round(t.entry_price, 2),
                    "stop_loss": round(t.stop_loss, 2),
                    "target": round(t.target, 2),
                    "exit_date": t.exit_date.isoformat(),
                    "exit_price": round(t.exit_price, 2),
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                    "return_pct": round(t.return_pct, 2),
                    "r_multiple": round(t.r_multiple, 2),
                    "taken_in_portfolio": id(t) in taken,
                }
                for t in trades
            ],
            caveats=BACKTEST_CAVEATS,
        )


def _assess(stats: dict, strategy: dict, benchmark: dict) -> Assessment:
    n = stats["total_trades"]
    expectancy = stats.get("expectancy_r") or 0.0
    total = strategy.get("total_return_pct") or 0.0
    cagr, bench_cagr = strategy.get("cagr_pct") or 0.0, benchmark.get("cagr_pct") or 0.0
    dd, bench_dd = strategy.get("max_drawdown_pct") or 0.0, benchmark.get("max_drawdown_pct") or 0.0
    adequate = n >= MIN_TRADES_FOR_CONFIDENCE

    if n == 0:
        summary = "The rules produced no trades in this period, so there is nothing to evaluate. Try a broader index, a longer period or looser filters."
    else:
        edge = (
            f"an average trade made {expectancy:+.2f}R after costs (win rate {stats['win_rate_pct']:.0f}%, "
            f"payoff ratio {stats['payoff_ratio'] if stats.get('payoff_ratio') is not None else 'n/a'})"
        )
        summary = (
            f"{n} trades: {edge}. The portfolio returned {total:+.1f}% ({cagr:+.1f}% a year) with a worst drawdown of "
            f"{dd:.1f}%, versus {bench_cagr:+.1f}% a year and {bench_dd:.1f}% for buy & hold of the same stocks."
        )
        if not adequate:
            summary += f" With fewer than {MIN_TRADES_FOR_CONFIDENCE} trades this is too small a sample to call the strategy profitable or not."
    return Assessment(
        profitable=total > 0,
        positive_expectancy=expectancy > 0,
        beats_benchmark=n > 0 and cagr > bench_cagr,
        lower_drawdown_than_benchmark=dd > bench_dd,
        sample_size_adequate=adequate,
        summary=summary,
    )
