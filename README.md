# Minervini VCP Screener

An API that takes a stock-market index, pulls its constituents, and screens them with
Mark Minervini's method (SEPA): the **Trend Template** to find stage-2 leaders, then the
**Volatility Contraction Pattern (VCP)** to find a low-risk buy point. For every buyable
stock it returns the name, entry price range, stop-loss and target.

## Quick start

```bash
uv sync                                   # or: pip install -e .
uv run python -m screener serve           # dashboard on http://127.0.0.1:8000/  (API docs at /docs)
uv run python -m screener serve --demo    # same, on synthetic demo data - no market-data access needed
curl "http://127.0.0.1:8000/api/v1/screen?index=NIFTY500"

uv run python -m screener screen NIFTY50  # same thing as a CLI table (--json for the full payload)
uv run python -m screener backtest NIFTY50 --years 3   # is the strategy profitable on this index?
uv run pytest                             # tests run offline on synthetic data
```

## Dashboard

Open `http://127.0.0.1:8000/` once the server is running.

- **Pick an index** in the filter row (India / United States groups) and the screen runs; the rule settings next
  to it (target multiple, max risk, min RS, volume dry-up, cheat entries) apply to everything below. Your last
  index is remembered. `Refresh data` bypasses the 15-minute cache.
- **Funnel tiles**: buyable setups today, index constituents, stocks with data, Trend Template passers.
- **Buyable setups table**: status, price, entry range, stop-loss, target, risk, RS rating, VCP footprint, and a
  mini stop -> entry -> target scale with the current price on it.
- **Select a stock** for its trade-plan tiles and a price chart with the contractions marked, the entry zone,
  stop and target drawn to the right of the pivot, and volume (dry-up window highlighted, 50-day average) on its
  own panel underneath. Hover or use the arrow keys for per-day values; `Show data table` lists the candles.
  Beside it: contraction-depth bars and meters for volume dry-up, RS rating and pivot height against their limits.
- **Trend Template watchlist**: stage-2 stocks with no buyable pivot yet.

It is one self-contained file (`screener/static/dashboard.html`, plain JS + inline SVG, no CDN or build step),
follows the OS light / dark setting (`Theme` button or `?theme=dark` to override) and accepts `?index=NIFTY500`.

`--demo` serves a made-up market - invented companies marked "(demo)", synthetic prices with a few textbook
setups - for exploring the dashboard and API offline. The header shows a "Demo data" badge; it is never mixed
with real data, and a backtest run in demo mode says nothing about real profitability.

## API

### `GET /api/v1/screen?index=<INDEX>`

| Param | Default | Meaning |
|---|---|---|
| `index` | required | `NIFTY50`, `NIFTY500`, `SP500`, `NASDAQ100`, ... Loose spelling works (`nifty 50`, `S&P 500`, `^NDX`). |
| `reward_to_risk` | `3` | Target = entry + N x risk |
| `max_stop_pct` | `8` | Maximum risk from the price you pay. Setups whose logical stop is further away are skipped |
| `min_rs_rating` | `70` | Minimum relative-strength percentile |
| `max_dry_up_ratio` | `0.7` | Volume dry-up: 10-day average volume as a fraction of the 50-day average |
| `include_cheat_entries` | `false` | Also return pivots in the middle / lower third of the base |
| `include_chart` | `false` | Attach daily candles (from before the base to today) to each pick; the dashboard uses this |
| `refresh` | `false` | Bypass the 15-minute result cache |

```json
{
  "index": "NIFTY50",
  "as_of": "2026-09-16",
  "summary": {"constituents": 50, "with_price_data": 50, "passed_trend_template": 9, "actionable_vcp": 1},
  "results": [
    {
      "symbol": "EXAMPLE",
      "name": "Example Industries Ltd.",
      "status": "SETUP",
      "entry_type": "PIVOT",
      "current_price": 96.0,
      "pct_from_pivot": -1.52,
      "entry_range": {"low": 97.49, "high": 100.29},
      "reference_entry": 97.49,
      "stop_loss": 92.27,
      "target": 113.13,
      "risk_pct": 5.35,
      "max_risk_pct": 8.0,
      "reward_pct": 16.05,
      "reward_to_risk": 3.0,
      "stop_basis": "below low of last contraction",
      "rs_rating": 94.0,
      "vcp": {"footprint": "12W 21/10/5 3T", "pivot_position_pct": 85.6, "volume_dry_up_ratio": 0.62, "contractions": ["..."]}
    }
  ],
  "trend_template_watchlist": ["ABC", "XYZ"]
}
```

- `status: "SETUP"` - price is just under the pivot. Buy **when it trades through `entry_range.low`**, ideally on volume >= 1.4x average.
- `status: "BREAKOUT"` - **closed** above the pivot within the last 10 sessions and still buyable. An intraday poke that closes back under the pivot does not count. `vcp.breakout_volume_confirmed` says whether the breakout session traded >= 1.4x average volume.
- `reference_entry` - the price the risk and target figures assume you pay: the pivot for a `SETUP`, the current price for a `BREAKOUT`. `max_risk_pct` is the risk if you buy at the top of the range.
- `trend_template_watchlist` - stage-2 stocks with no pivot that is buyable within the risk limit today; candidates for the coming weeks.
- `incomplete_session_excluded` - set when you screen while the market is open (see Data).
- Errors: `404` unknown index, `502` constituents source unreachable, `503` price data unavailable.

### `GET /api/v1/backtest?index=<INDEX>`

Replays the strategy over history and reports whether it made money.

| Param | Default | Meaning |
|---|---|---|
| `index` | required | As above |
| `years` | `3` | Length of the test period (an extra 1.5 years is downloaded as warm-up) |
| `initial_capital` | `1000000` | Starting account |
| `risk_per_trade_pct` | `1` | Equity lost if a stop is hit; position size follows from the stop distance |
| `max_positions` | `10` | Open-position limit; when signals compete, the highest RS rating wins the slot |
| `max_position_pct` | `20` | Cap on one position, % of equity |
| `max_holding_days` | `60` | Time stop, trading days |
| `cost_pct` | `0.1` | Commission + slippage **per side**, % |
| `reward_to_risk`, `max_stop_pct`, `min_rs_rating`, `max_dry_up_ratio`, `include_cheat_entries` | as `/screen` | Strategy rules under test |
| `include_trades` | `true` | Include every trade in the response |

What comes back:

- `assessment` - the verdict in plain words plus flags: `profitable`, `positive_expectancy`, `beats_benchmark`,
  `lower_drawdown_than_benchmark`, `sample_size_adequate` (>= 30 trades).
- `trade_stats` - the raw edge, every signal weighted equally: trades, win rate, average win / loss, payoff ratio,
  **expectancy in R and %**, profit factor, best / worst trade, holding time, longest losing streak, and
  breakdowns by exit reason and by entry kind (`SETUP` vs `BREAKOUT`).
- `portfolio` - those trades through one cash account: final equity, total return, **CAGR**, **max drawdown**
  (with dates), volatility, **Sharpe**, Sortino, Calmar, best / worst month, exposure, and how many signals were
  skipped for lack of a free slot or cash.
- `benchmark` - the same figures for equal-weight buy & hold of the same stocks, `yearly_returns` side by side,
  a month-end `equity_curve`, and the `trades` list.

How it trades, so the numbers can be trusted as far as they go:

- **No look-ahead.** Each evening the same screen as `/screen` runs on data up to that day only. A `SETUP` becomes a
  buy-stop at the pivot for the next session, a `BREAKOUT` is bought at the next open, nothing is bought above the
  entry range, and the target is set from the actual fill. A test asserts that deleting future data never changes a
  past trade.
- **Pessimistic fills.** A gap through the stop fills at the worse open; a bar containing both stop and target
  counts as a loss; costs are charged on entry and exit.
- **Reading it.** Look at `expectancy_r` first (is there an edge per trade after costs?), then `max_drawdown_pct`
  (could you sit through it?), then the benchmark (was the effort worth it?). In a strong bull market buy & hold of
  today's winners is very hard to beat; the strategy's case is usually lower drawdown and lower exposure.
- **Limits** are returned in `caveats`. The important one is survivorship bias: today's index members are used for
  the whole period.

A cold run takes seconds for a 50-stock index and several minutes for a 500-stock index; results are cached for 6 hours.

### `GET /api/v1/indices`

Lists supported indices: the Nifty family (50, Next 50, 100, 200, 500, Midcap 50/100/150,
Smallcap 50/100/250, Bank, IT, Auto, Pharma, FMCG, Metal), `SP500` and `NASDAQ100`.
Adding one is a single line in `screener/indices.py`.

## How the method is implemented

**1. Trend Template** (`trend_template.py`) - all eight must hold:

1. Price above the 150- and 200-day SMA
2. 150-day SMA above the 200-day SMA
3. 200-day SMA rising for at least a month
4. 50-day SMA above the 150- and 200-day SMA
5. Price above the 50-day SMA
6. Price at least 30% above its 52-week low
7. Price within 25% of its 52-week high
8. RS rating >= 70

The RS rating is an IBD-style score (40% last quarter, 20% each of the three before),
percentile-ranked 1-99 **against the other stocks in the requested index**. On a narrow
index (e.g. `NIFTYBANK`, 12 stocks) that ranking is coarse; screen a broad index for best results.

**2. VCP** (`vcp.py`) - contractions are found without any zig-zag threshold: H1 = highest
high of the last year, L1 = lowest low after it, H2 = highest high after L1, and so on.
Each high is lower and each low higher than the one before, so the pairs trace the narrowing
range. The base qualifies when:

- it has 2-6 contractions, each shallower than the last, each lasting >= 3 bars
- the first is <= 35% deep, the last <= 10% and at most half as deep as the first
- the base is >= 3 weeks long
- the pivot sits in the **upper third** of the base. Pivots in the middle or lower third are
  Minervini's "cheat" / "low cheat" entries - earlier but with overhead supply - and are only
  returned, labelled, with `include_cheat_entries=true`
- volume has dried up: the 10-day average is <= 0.7x the 50-day average **and** at least one
  of those sessions is <= 0.6x
- price is within 5% below the pivot, or *closed* above it <= 10 bars ago and is <= 5% above it

The 0.7 / 0.6 volume thresholds are judgement calls that have not been calibrated on real charts yet.
If a broad index returns next to nothing, loosen `max_dry_up_ratio` (e.g. `0.85`) before anything else.

**3. Trade plan** (`trade_plan.py`)

- **Stop-loss**: 0.5% under the low of the final contraction (never closer than 3% to the pivot). If that
  logical stop is more than `max_stop_pct` (8%) below the pivot, the stock is **skipped** rather than given an
  arbitrary closer stop: a loose final contraction is not a low-risk entry.
- **Entry range**: from the pivot up to 5% above it, *or* the price at which the stop would be 8% away,
  whichever is lower - so nowhere in the range breaks Minervini's maximum-risk rule. A breakout already
  trading above that is reported on the watchlist, not as a buy.
- **Target**: `reference_entry` + 3 x risk.

Every threshold lives in `screener/config.py`.

## Data

- Constituents: NSE Indices CSVs, Wikipedia (S&P 500), Nasdaq's public API (Nasdaq-100); cached for 24h.
- Prices: two years of daily candles from Yahoo Finance via `yfinance`, downloaded in batches. They are
  adjusted for **splits only**: dividend adjustment rescales history on every ex-date, which would move
  pivots and stops away from the prices on your chart and at your broker.
  Yahoo is unofficial and rate-limits aggressively; a 500-stock index takes a minute or so on a cold cache.
  To use a broker or vendor feed, implement `PriceProvider.get_history` (`screener/data.py`) and pass it to `ScreenerService`.
- Only completed sessions are screened. Until 20 minutes after the close (15:30 IST / 16:00 ET) today's
  partial candle is dropped - its volume would look like a dry-up and a breakout would look unconfirmed -
  and the response names the ignored date in `incomplete_session_excluded`. Exchange holidays and
  half-days are not modelled; the worst case is that a finished half-day candle is ignored until the normal close.
- Behind a TLS-inspecting corporate proxy, Python will reject the proxy's certificate. Point
  `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` at a CA bundle your IT team provides.

## Disclaimer

This is a mechanical pattern screen for education and idea generation, not investment advice.
Minervini's own process adds fundamentals (earnings and sales acceleration), a read of the general
market, and discretionary chart reading that code cannot replace. Check the chart before you trade.
