"""CLI: ``python -m screener screen NIFTY50``, ``... backtest NIFTY50 --years 3`` or ``... serve``."""

from __future__ import annotations

import argparse
import logging
import sys

from .data import DataUnavailableError
from .indices import INDICES, ConstituentsFetchError, UnknownIndexError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minervini-screener", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--demo", action="store_true", help="serve synthetic demo data instead of live market data")

    screen = sub.add_parser("screen", help="screen an index and print the picks")
    screen.add_argument("index", help=f"one of: {', '.join(INDICES)}")
    screen.add_argument("--json", action="store_true", help="print the full API response as JSON")

    backtest = sub.add_parser("backtest", help="backtest the strategy on an index")
    backtest.add_argument("index", help=f"one of: {', '.join(INDICES)}")
    backtest.add_argument("--years", type=float, default=3.0)
    backtest.add_argument("--capital", type=float, default=1_000_000.0)
    backtest.add_argument("--risk-pct", type=float, default=1.0, help="equity risked per trade, %%")
    backtest.add_argument("--max-positions", type=int, default=10)
    backtest.add_argument("--json", action="store_true", help="print the full API response as JSON")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "serve":
        import os

        import uvicorn

        if args.demo:
            os.environ["SCREENER_DEMO"] = "1"
        print(f"Dashboard: http://{args.host}:{args.port}/   API docs: http://{args.host}:{args.port}/docs")
        uvicorn.run("screener.api:app", host=args.host, port=args.port)
        return 0

    if args.command == "backtest":
        return _backtest(args)

    from .service import ScreenerService

    try:
        result = ScreenerService().screen(args.index)
    except (UnknownIndexError, ConstituentsFetchError, DataUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    s = result.summary
    print(
        f"{result.index_name} as of {result.as_of}: {s.constituents} stocks -> "
        f"{s.passed_trend_template} pass Trend Template -> {s.actionable_vcp} actionable VCP"
    )
    if result.incomplete_session_excluded:
        print(f"Market still open: ignoring the partial candle of {result.incomplete_session_excluded}")
    if result.results:
        header = f"{'SYMBOL':<14}{'STATUS':<10}{'PRICE':>10}{'ENTRY RANGE':>24}{'SL':>10}{'TARGET':>10}{'RISK%':>7}{'RS':>4}  VCP"
        print(header)
        for p in result.results:
            entry = f"{p.entry_range.low:.2f} - {p.entry_range.high:.2f}"
            print(
                f"{p.symbol:<14}{p.status:<10}{p.current_price:>10.2f}{entry:>24}{p.stop_loss:>10.2f}"
                f"{p.target:>10.2f}{p.risk_pct:>7.1f}{p.rs_rating:>4.0f}  {p.vcp.footprint}"
            )
    if result.trend_template_watchlist:
        print("Trend Template watchlist (no pivot yet):", ", ".join(result.trend_template_watchlist))
    return 0


def _backtest(args: argparse.Namespace) -> int:
    from .backtest import BacktestConfig
    from .service import BacktestService

    bt = BacktestConfig(
        years=args.years,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk_pct,
        max_positions=args.max_positions,
    )
    try:
        result = BacktestService().run(args.index, bt=bt)
    except (UnknownIndexError, ConstituentsFetchError, DataUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    def fmt(value, suffix=""):
        return "n/a" if value is None else f"{value:,.2f}{suffix}"

    t, p, b = result.trade_stats, result.portfolio, result.benchmark
    print(f"{result.index_name}: {result.period.start} to {result.period.end} ({result.period.years} years)")
    print(f"\n{result.assessment.summary}\n")
    rows = [
        ("Trades", f"{t.total_trades} ({t.winners} won / {t.losers} lost)"),
        ("Win rate", fmt(t.win_rate_pct, "%")),
        ("Average win / loss", f"{fmt(t.avg_win_pct, '%')} / {fmt(t.avg_loss_pct, '%')}"),
        ("Payoff ratio", fmt(t.payoff_ratio)),
        ("Expectancy per trade", f"{fmt(t.expectancy_r, 'R')} ({fmt(t.expectancy_pct, '%')})"),
        ("Profit factor", fmt(t.profit_factor)),
        ("Max consecutive losses", str(t.max_consecutive_losses)),
        ("Total return", f"{fmt(p.total_return_pct, '%')}   benchmark {fmt(b.total_return_pct, '%')}"),
        ("CAGR", f"{fmt(p.cagr_pct, '%')}   benchmark {fmt(b.cagr_pct, '%')}"),
        ("Max drawdown", f"{fmt(p.max_drawdown_pct, '%')}   benchmark {fmt(b.max_drawdown_pct, '%')}"),
        ("Sharpe / Sortino", f"{fmt(p.sharpe_ratio)} / {fmt(p.sortino_ratio)}"),
        ("Average exposure", fmt(p.avg_exposure_pct, "%")),
        ("Final equity", fmt(p.final_equity)),
    ]
    for label, value in rows:
        print(f"  {label:<24}{value}")
    print("\n  Year   Strategy  Benchmark  Trades")
    for y in result.yearly_returns:
        print(f"  {y.year}  {fmt(y.strategy_return_pct, '%'):>9}  {fmt(y.benchmark_return_pct, '%'):>9}  {y.trades_closed:>6}")
    print("\nCaveats:")
    for caveat in result.caveats:
        print(f"  - {caveat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
