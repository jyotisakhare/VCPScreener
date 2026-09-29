"""Response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntryRange(BaseModel):
    low: float = Field(description="Pivot price: buy as price trades through this level")
    high: float = Field(
        description="Highest buyable price: the lower of the buy zone and the price where the stop "
        "would exceed the maximum risk"
    )


class ContractionOut(BaseModel):
    high_date: str
    high: float
    low_date: str
    low: float
    depth_pct: float


class VCPDetails(BaseModel):
    footprint: str = Field(description="Minervini shorthand: <weeks>W <depths> <contractions>T")
    contractions: list[ContractionOut]
    base_start: str
    base_weeks: float
    pivot_position_pct: float = Field(description="Pivot height within the base: 0 = base low, 100 = base high")
    volume_dry_up_ratio: float = Field(description="Avg volume of last 10 bars of the base / 50-bar avg")
    quietest_day_ratio: float = Field(description="Lowest-volume session of those 10 bars / 50-bar avg")
    breakout_bars_ago: int | None = Field(default=None, description="Sessions since the first close above the pivot")
    breakout_volume_ratio: float | None = None
    breakout_volume_confirmed: bool | None = None


class PriceChart(BaseModel):
    """Daily candles from shortly before the base to today, for drawing the setup."""

    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    volume_avg_50: list[float | None]


class StockPick(BaseModel):
    symbol: str
    ticker: str
    name: str
    sector: str | None = None
    status: str = Field(description="SETUP = waiting below pivot, BREAKOUT = closed above pivot, still buyable")
    entry_type: str = Field(
        description="PIVOT = standard buy point in the upper third of the base; CHEAT / LOW_CHEAT = "
        "earlier, riskier entries lower in the base (only with include_cheat_entries)"
    )
    current_price: float
    pct_from_pivot: float
    entry_range: EntryRange
    reference_entry: float = Field(
        description="Price the risk/target figures assume: the pivot for a SETUP, the current price for a BREAKOUT"
    )
    stop_loss: float
    target: float
    risk_pct: float = Field(description="Stop distance from reference_entry")
    max_risk_pct: float = Field(description="Stop distance if bought at the top of the entry range")
    reward_pct: float
    reward_to_risk: float
    stop_basis: str
    rs_rating: float
    pct_below_52w_high: float
    vcp: VCPDetails
    chart: PriceChart | None = Field(default=None, description="Only with include_chart=true")


class ScreenSummary(BaseModel):
    constituents: int
    with_price_data: int
    passed_trend_template: int
    actionable_vcp: int


class ScreenResponse(BaseModel):
    index: str
    index_name: str
    market: str
    as_of: str | None = Field(description="Date of the latest completed daily candle used")
    incomplete_session_excluded: str | None = Field(
        default=None,
        description="Set when the market had not closed yet: that day's partial candle was ignored",
    )
    generated_at: str
    summary: ScreenSummary
    results: list[StockPick]
    trend_template_watchlist: list[str] = Field(
        description="Stage-2 stocks that pass the Trend Template but have no VCP that is buyable "
        "within the risk limit today"
    )
    disclaimer: str


class IndexInfo(BaseModel):
    key: str
    name: str
    market: str


# --- Backtest ------------------------------------------------------------------------


class ScreenSummaryUniverse(BaseModel):
    constituents: int
    with_price_data: int
    with_enough_history: int


class BacktestPeriod(BaseModel):
    start: str
    end: str
    trading_days: int
    years: float


class BacktestSettings(BaseModel):
    initial_capital: float
    risk_per_trade_pct: float
    max_positions: int
    max_position_pct: float
    max_holding_bars: int
    cost_pct_per_side: float
    reward_to_risk: float
    max_stop_pct: float
    min_rs_rating: float
    max_dry_up_ratio: float
    include_cheat_entries: bool


class TradeGroupStats(BaseModel):
    trades: int
    win_rate_pct: float | None
    avg_return_pct: float | None
    expectancy_r: float | None


class TradeStats(BaseModel):
    """Every signal the rules produced, each weighted equally: the raw edge of the strategy."""

    total_trades: int
    winners: int = 0
    losers: int = 0
    win_rate_pct: float | None = None
    avg_win_pct: float | None = None
    avg_loss_pct: float | None = None
    payoff_ratio: float | None = Field(default=None, description="Average win / average loss")
    expectancy_pct: float | None = Field(default=None, description="Average net return per trade")
    expectancy_r: float | None = Field(
        default=None, description="Average net result per trade in units of initial risk (R). > 0 means an edge"
    )
    profit_factor: float | None = Field(default=None, description="Sum of winning returns / sum of losing returns")
    median_return_pct: float | None = None
    best_trade_pct: float | None = None
    worst_trade_pct: float | None = None
    avg_bars_held: float | None = None
    avg_bars_held_winners: float | None = None
    avg_bars_held_losers: float | None = None
    max_consecutive_losses: int = 0
    by_exit_reason: dict[str, TradeGroupStats] = {}
    by_entry_kind: dict[str, TradeGroupStats] = {}


class CurveStats(BaseModel):
    final_equity: float | None = None
    total_return_pct: float | None = None
    cagr_pct: float | None = None
    max_drawdown_pct: float | None = None
    max_drawdown_peak: str | None = None
    max_drawdown_trough: str | None = None
    annual_volatility_pct: float | None = None
    sharpe_ratio: float | None = Field(default=None, description="Annualised, 0% risk-free rate")
    sortino_ratio: float | None = None
    calmar_ratio: float | None = Field(default=None, description="CAGR / max drawdown")
    best_month_pct: float | None = None
    worst_month_pct: float | None = None
    profitable_months_pct: float | None = Field(default=None, description="Of the months with any P&L")


class PortfolioStats(CurveStats):
    """The same trades run through one cash account with position sizing and a position limit."""

    initial_capital: float
    trades_taken: int
    trades_skipped_no_free_slot: int
    trades_skipped_no_cash: int
    net_profit: float | None = None
    profit_factor: float | None = Field(default=None, description="Gross profit / gross loss in currency")
    avg_exposure_pct: float | None = Field(default=None, description="Average share of equity invested")
    max_exposure_pct: float | None = None


class BenchmarkStats(CurveStats):
    name: str


class YearlyReturn(BaseModel):
    year: int
    strategy_return_pct: float | None
    benchmark_return_pct: float | None
    trades_closed: int


class Assessment(BaseModel):
    profitable: bool = Field(description="Portfolio ended above its starting capital")
    positive_expectancy: bool = Field(description="Average trade made money after costs (expectancy_r > 0)")
    beats_benchmark: bool = Field(description="Higher CAGR than equal-weight buy & hold of the same stocks")
    lower_drawdown_than_benchmark: bool
    sample_size_adequate: bool = Field(description="At least 30 trades; fewer says little either way")
    summary: str


class EquityPoint(BaseModel):
    date: str
    strategy: float
    benchmark: float


class TradeOut(BaseModel):
    symbol: str
    ticker: str
    signal_date: str
    entry_date: str
    entry_kind: str
    entry_type: str
    footprint: str
    rs_rating: float
    entry_price: float
    stop_loss: float
    target: float
    exit_date: str
    exit_price: float
    exit_reason: str
    bars_held: int
    return_pct: float
    r_multiple: float
    taken_in_portfolio: bool


class BacktestResponse(BaseModel):
    index: str
    index_name: str
    market: str
    generated_at: str
    period: BacktestPeriod
    universe: ScreenSummaryUniverse
    settings: BacktestSettings
    assessment: Assessment
    trade_stats: TradeStats
    portfolio: PortfolioStats
    benchmark: BenchmarkStats
    yearly_returns: list[YearlyReturn]
    equity_curve: list[EquityPoint] = Field(description="Month-end values")
    trades: list[TradeOut] | None = None
    caveats: list[str]
