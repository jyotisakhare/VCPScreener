"""Walk-forward backtest of the Trend Template + VCP strategy.

The simulation replays what a trader using the screener every evening would do:

1. After the close of day *t* the screen runs on data up to *t* only.
2. A ``SETUP`` becomes a buy-stop order at the pivot for day *t+1*; a ``BREAKOUT``
   becomes a buy at the open of *t+1*. Orders live for one session and are rebuilt
   from the next evening's screen, and nothing is bought above the entry range.
3. A position is closed at the stop-loss, the target, or after ``max_holding_bars``.

Fills are deliberately pessimistic where daily candles are ambiguous: a gap through
the stop is filled at the (worse) open, and if stop and target are both inside one
bar the stop is assumed to have been hit first.

The engine has two layers. ``simulate_ticker`` produces every trade the rules would
have taken in one stock - the raw edge, independent of account size. ``run_portfolio``
then feeds those trades through a cash account with risk-based position sizing and a
limit on open positions to get an equity curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import ScreenerConfig
from .trade_plan import plan_best_candidate
from .trend_template import rs_rating_history, trend_template_mask
from .vcp import STATUS_BREAKOUT

EXIT_STOP = "STOP"
EXIT_TARGET = "TARGET"
EXIT_TIME = "TIME"
EXIT_END_OF_DATA = "END_OF_DATA"  # still open when the data ended; marked to the last close

# Bars needed before the first signal is possible: a year for the RS rating plus a margin.
WARMUP_YEARS = 1.5


@dataclass(frozen=True)
class BacktestConfig:
    years: float = 3.0
    initial_capital: float = 1_000_000.0
    risk_per_trade_pct: float = 1.0  # share of equity lost if the stop is hit
    max_positions: int = 10
    max_position_pct: float = 20.0  # cap on a single position, % of equity
    max_holding_bars: int = 60  # time stop
    cost_pct: float = 0.1  # commission + slippage, charged on entry and again on exit


@dataclass
class Trade:
    ticker: str
    signal_date: date
    entry_date: date
    entry_kind: str  # SETUP (buy-stop at pivot) | BREAKOUT (bought next open)
    entry_type: str  # PIVOT | CHEAT | LOW_CHEAT
    footprint: str
    rs_rating: float
    entry_price: float
    stop_loss: float
    target: float
    exit_date: date
    exit_price: float
    exit_reason: str
    bars_held: int
    return_pct: float  # net of costs
    r_multiple: float  # net return in units of initial risk


@dataclass
class _Order:
    kind: str
    entry_type: str
    footprint: str
    rs_rating: float
    signal_bar: int
    pivot: float
    entry_high: float
    stop: float


@dataclass
class _Position:
    order: _Order
    entry_bar: int
    entry: float
    target: float


def _try_fill(order: _Order, open_: float, high: float) -> float | None:
    if open_ > order.entry_high:
        return None  # gapped beyond the buyable range: do not chase
    if order.kind == STATUS_BREAKOUT:
        return open_ if open_ >= order.pivot else None  # slipped back under the pivot overnight
    if open_ >= order.pivot:
        return open_  # gapped through the pivot: a buy-stop fills at the open
    return order.pivot if high >= order.pivot else None


def _check_exit(
    pos: _Position, bar: int, open_: float, high: float, low: float, close: float, is_last_bar: bool, max_bars: int
) -> tuple[float, str] | None:
    stop, target = pos.order.stop, pos.target
    on_entry_bar = bar == pos.entry_bar
    # The open of the entry bar came before (or at) the fill, so it cannot gap through anything.
    if not on_entry_bar and open_ <= stop:
        return open_, EXIT_STOP
    if low <= stop:  # checked before the target: when both are in the bar, assume the worst
        return stop, EXIT_STOP
    if not on_entry_bar and open_ >= target:
        return open_, EXIT_TARGET
    if high >= target:
        return target, EXIT_TARGET
    if bar - pos.entry_bar >= max_bars:
        return close, EXIT_TIME
    if is_last_bar:
        return close, EXIT_END_OF_DATA
    return None


def simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    tradable: np.ndarray,
    rs_rating: np.ndarray,
    config: ScreenerConfig,
    bt: BacktestConfig,
    start_bar: int = 0,
) -> list[Trade]:
    """All trades the rules take in one stock from ``start_bar`` on, one position at a time.

    ``tradable[i]`` says whether the stock passes the Trend Template at the close of bar ``i``.
    """
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    dates = df.index
    n = len(df)
    # The detector never looks further back than this, so a bounded slice gives identical
    # signals to the full history - and provably cannot see anything after bar i.
    lookback = config.vcp.lookback_bars + config.vcp.max_breakout_age_bars + 5
    cost = bt.cost_pct / 100

    trades: list[Trade] = []
    order: _Order | None = None
    pos: _Position | None = None
    for i in range(start_bar, n):
        if pos is None and order is not None:
            fill = _try_fill(order, open_[i], high[i])
            if fill is not None:
                target = fill + config.plan.reward_to_risk * (fill - order.stop)
                pos = _Position(order, i, fill, target)
        order = None

        if pos is not None:
            exit_ = _check_exit(pos, i, open_[i], high[i], low[i], close[i], i == n - 1, bt.max_holding_bars)
            if exit_ is not None:
                price, reason = exit_
                net = (price * (1 - cost)) / (pos.entry * (1 + cost)) - 1.0
                risk = (pos.entry - pos.order.stop) / pos.entry
                trades.append(
                    Trade(
                        ticker=ticker,
                        signal_date=dates[pos.order.signal_bar].date(),
                        entry_date=dates[pos.entry_bar].date(),
                        entry_kind=pos.order.kind,
                        entry_type=pos.order.entry_type,
                        footprint=pos.order.footprint,
                        rs_rating=pos.order.rs_rating,
                        entry_price=pos.entry,
                        stop_loss=pos.order.stop,
                        target=pos.target,
                        exit_date=dates[i].date(),
                        exit_price=price,
                        exit_reason=reason,
                        bars_held=i - pos.entry_bar,
                        return_pct=net * 100,
                        r_multiple=net / risk,
                    )
                )
                pos = None

        if pos is None and tradable[i] and i < n - 1:
            signal = plan_best_candidate(df.iloc[max(0, i + 1 - lookback) : i + 1], config)
            if signal is not None:
                vcp, plan = signal
                order = _Order(
                    kind=vcp.status,
                    entry_type=vcp.entry_type,
                    footprint=vcp.footprint,
                    rs_rating=float(rs_rating[i]),
                    signal_bar=i,
                    pivot=plan.entry_low,
                    entry_high=plan.entry_high,
                    stop=plan.stop_loss,
                )
    return trades


def generate_trades(
    histories: dict[str, pd.DataFrame], config: ScreenerConfig, bt: BacktestConfig, start: pd.Timestamp
) -> list[Trade]:
    ratings = rs_rating_history(histories)
    trades: list[Trade] = []
    for ticker, df in histories.items():
        if len(df) < config.trend.min_history_bars:
            continue
        rs = ratings[ticker].reindex(df.index)
        tradable = trend_template_mask(df, rs, config.trend)
        start_bar = int(df.index.searchsorted(start))
        trades.extend(simulate_ticker(ticker, df, tradable, rs.to_numpy(), config, bt, start_bar))
    trades.sort(key=lambda t: (t.entry_date, -t.rs_rating, t.ticker))
    return trades


# --- Portfolio layer -----------------------------------------------------------------


@dataclass
class PortfolioResult:
    equity: pd.Series
    invested_fraction: pd.Series
    taken: list[Trade]
    pnl: list[float]  # currency P&L per taken trade, same order as ``taken``
    skipped_no_slot: int
    skipped_no_cash: int


def run_portfolio(trades: list[Trade], closes: pd.DataFrame, bt: BacktestConfig) -> PortfolioResult:
    """Run the trades through a cash account, marked to market on every date of ``closes``.

    Size = the number of shares that loses ``risk_per_trade_pct`` of equity at the stop,
    capped at ``max_position_pct`` of equity and by available cash. When several trades
    start on one day the strongest RS rating gets the free slot first. Fractional shares
    are allowed so results do not depend on the account currency.
    """
    cost = bt.cost_pct / 100
    by_entry: dict[date, list[Trade]] = {}
    for trade in trades:
        by_entry.setdefault(trade.entry_date, []).append(trade)

    cash = equity = bt.initial_capital
    open_: list[tuple[Trade, float, float]] = []  # trade, shares, cash paid
    taken: list[Trade] = []
    pnl: list[float] = []
    skipped_no_slot = skipped_no_cash = 0
    equity_curve, invested_curve = [], []

    def close_position(trade: Trade, shares: float, paid: float) -> None:
        nonlocal cash
        proceeds = shares * trade.exit_price * (1 - cost)
        cash += proceeds
        taken.append(trade)
        pnl.append(proceeds - paid)

    for ts in closes.index:
        today = ts.date()
        still_open = []
        for trade, shares, paid in open_:
            if trade.exit_date <= today:
                close_position(trade, shares, paid)
            else:
                still_open.append((trade, shares, paid))
        open_ = still_open

        for trade in sorted(by_entry.get(today, []), key=lambda t: -t.rs_rating):
            if len(open_) >= bt.max_positions:
                skipped_no_slot += 1
                continue
            unit_cost = trade.entry_price * (1 + cost)
            shares = equity * bt.risk_per_trade_pct / 100 / (trade.entry_price - trade.stop_loss)
            shares = min(shares, equity * bt.max_position_pct / 100 / unit_cost, cash / unit_cost)
            if shares * unit_cost < equity * 0.005:  # not worth a position: effectively out of cash
                skipped_no_cash += 1
                continue
            paid = shares * unit_cost
            cash -= paid
            if trade.exit_date <= today:  # stopped out on the entry day
                close_position(trade, shares, paid)
            else:
                open_.append((trade, shares, paid))

        invested = sum(shares * closes.at[ts, trade.ticker] for trade, shares, _ in open_)
        equity = cash + invested
        equity_curve.append(equity)
        invested_curve.append(invested / equity if equity > 0 else 0.0)

    return PortfolioResult(
        equity=pd.Series(equity_curve, index=closes.index, dtype=float),
        invested_fraction=pd.Series(invested_curve, index=closes.index, dtype=float),
        taken=taken,
        pnl=pnl,
        skipped_no_slot=skipped_no_slot,
        skipped_no_cash=skipped_no_cash,
    )


def close_matrix(histories: dict[str, pd.DataFrame], start: pd.Timestamp) -> pd.DataFrame:
    """Dates x tickers closes from ``start`` on; gaps (halts, holidays) carry the last price."""
    frame = pd.DataFrame({t: df["close"] for t, df in histories.items()}).sort_index().ffill()
    return frame[frame.index >= start]


def equal_weight_buy_and_hold(closes: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Benchmark: the same capital split equally across the universe on day one and held."""
    listed = closes.loc[:, closes.iloc[0].notna()]
    return (listed / listed.iloc[0]).mean(axis=1) * initial_capital


# --- Metrics -------------------------------------------------------------------------


def finite_round(value: float | None, digits: int = 2) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def curve_metrics(equity: pd.Series, initial_capital: float) -> dict:
    """Return / risk figures of an equity curve. Sharpe and Sortino use a 0% risk-free rate."""
    if equity.empty:
        return {}
    curve = pd.concat([pd.Series([initial_capital], index=[equity.index[0] - pd.Timedelta(days=1)]), equity])
    returns = curve.pct_change().dropna()
    total = curve.iloc[-1] / initial_capital - 1.0
    years = max((equity.index[-1] - equity.index[0]).days, 1) / 365.25
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0

    drawdown = curve / curve.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = curve.loc[:trough].idxmax()
    max_dd = float(drawdown.min())

    std = returns.std()
    downside = math.sqrt((returns.clip(upper=0) ** 2).mean()) if len(returns) else 0.0
    monthly = curve.resample("ME").last().pct_change().dropna()
    active = monthly[monthly != 0]  # months spent entirely in cash are neither good nor bad
    return {
        "final_equity": finite_round(curve.iloc[-1]),
        "total_return_pct": finite_round(total * 100),
        "cagr_pct": finite_round(cagr * 100),
        "max_drawdown_pct": finite_round(max_dd * 100),
        "max_drawdown_peak": peak.date().isoformat() if max_dd < 0 else None,
        "max_drawdown_trough": trough.date().isoformat() if max_dd < 0 else None,
        "annual_volatility_pct": finite_round(std * math.sqrt(252) * 100) if len(returns) > 1 else None,
        "sharpe_ratio": finite_round(returns.mean() / std * math.sqrt(252)) if len(returns) > 1 and std > 0 else None,
        "sortino_ratio": finite_round(returns.mean() / downside * math.sqrt(252)) if downside > 0 else None,
        "calmar_ratio": finite_round(cagr / abs(max_dd)) if max_dd < 0 else None,
        "best_month_pct": finite_round(monthly.max() * 100) if len(monthly) else None,
        "worst_month_pct": finite_round(monthly.min() * 100) if len(monthly) else None,
        "profitable_months_pct": finite_round((active > 0).mean() * 100) if len(active) else None,
    }


def trade_statistics(trades: list[Trade]) -> dict:
    """Edge per trade, every trade weighted equally (independent of position sizing)."""
    stats: dict = {"total_trades": len(trades)}
    if not trades:
        return stats
    returns = np.array([t.return_pct for t in trades])
    r_multiples = np.array([t.r_multiple for t in trades])
    wins, losses = returns[returns > 0], returns[returns <= 0]

    longest = streak = 0
    for t in sorted(trades, key=lambda t: (t.exit_date, t.entry_date)):
        streak = streak + 1 if t.return_pct <= 0 else 0
        longest = max(longest, streak)

    def group(items: list[Trade]) -> dict:
        r = np.array([t.r_multiple for t in items])
        return {
            "trades": len(items),
            "win_rate_pct": finite_round((r > 0).mean() * 100),
            "avg_return_pct": finite_round(np.mean([t.return_pct for t in items])),
            "expectancy_r": finite_round(r.mean()),
        }

    def breakdown(key) -> dict:
        groups: dict[str, list[Trade]] = {}
        for t in trades:
            groups.setdefault(key(t), []).append(t)
        return {name: group(items) for name, items in sorted(groups.items())}

    stats.update(
        {
            "winners": int(len(wins)),
            "losers": int(len(losses)),
            "win_rate_pct": finite_round(len(wins) / len(trades) * 100),
            "avg_win_pct": finite_round(wins.mean()) if len(wins) else None,
            "avg_loss_pct": finite_round(losses.mean()) if len(losses) else None,
            "payoff_ratio": finite_round(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() < 0 else None,
            "expectancy_pct": finite_round(returns.mean()),
            "expectancy_r": finite_round(r_multiples.mean()),
            "profit_factor": finite_round(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else None,
            "median_return_pct": finite_round(float(np.median(returns))),
            "best_trade_pct": finite_round(returns.max()),
            "worst_trade_pct": finite_round(returns.min()),
            "avg_bars_held": finite_round(np.mean([t.bars_held for t in trades]), 1),
            "avg_bars_held_winners": finite_round(np.mean([t.bars_held for t in trades if t.return_pct > 0]), 1) if len(wins) else None,
            "avg_bars_held_losers": finite_round(np.mean([t.bars_held for t in trades if t.return_pct <= 0]), 1) if len(losses) else None,
            "max_consecutive_losses": longest,
            "by_exit_reason": breakdown(lambda t: t.exit_reason),
            "by_entry_kind": breakdown(lambda t: t.entry_kind),
        }
    )
    return stats


def yearly_breakdown(equity: pd.Series, benchmark: pd.Series, initial_capital: float, taken: list[Trade]) -> list[dict]:
    def by_year(curve: pd.Series) -> pd.Series:
        year_end = curve.groupby(curve.index.year).last()
        previous = year_end.shift(1)
        previous.iloc[0] = initial_capital
        return (year_end / previous - 1.0) * 100

    strategy, bench = by_year(equity), by_year(benchmark)
    return [
        {
            "year": int(year),
            "strategy_return_pct": finite_round(strategy[year]),
            "benchmark_return_pct": finite_round(bench[year]),
            "trades_closed": sum(t.exit_date.year == year for t in taken),
        }
        for year in strategy.index
    ]
