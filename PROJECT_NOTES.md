# Trading Bot — Project Notes

## Active Strategy

**SPY / QQQ Mean Reversion** — the only strategy currently enabled in the backtest and live bot.

BTC/USD Momentum Breakout and GLD/USO Trend Following have been disabled. See [Why the other strategies were removed](#why-the-other-strategies-were-removed).

---

## Strategy Parameters

| Parameter | Value | Location |
|---|---|---|
| Symbols | SPY, QQQ | `config.py → MEAN_REVERSION_SYMBOLS` |
| Timeframe | 15-minute bars | `config.py → MEAN_REVERSION_TIMEFRAME` |
| SMA / STD lookback | 20 periods | `config.py → MEAN_REVERSION_PERIOD` |
| Entry band — SPY | SMA ± 1.5 × σ | `config.py → STD_DEV_THRESHOLDS["SPY"]` |
| Entry band — QQQ | SMA ± 1.8 × σ | `config.py → STD_DEV_THRESHOLDS["QQQ"]` |
| ATR period | 14 bars | `config.py → ATR_PERIOD` |
| Risk per trade | 1% of equity per 1 ATR move | `config.py → RISK_PER_TRADE` |
| Capital per symbol | $100,000 | `--equity` arg, default |

**Entry logic:** When the 15-min close crosses below the lower band (long) or above the upper band (short), a signal is generated at that bar's close. The order fills at the **next bar's open** to avoid look-ahead bias.

**Exit logic:**
- *Profit target:* close returns to the 20-period SMA; fills at that bar's close.
- *Hard stop:* 1 ATR from entry price; fills at the stop price if the bar's low/high touches it.

---

## Position Sizing

### Formula

```
raw_size  = floor( allocated_capital × RISK_PER_TRADE / bar_ATR )
cap_size  = floor( allocated_capital / entry_price )
final_size = min( raw_size, cap_size )
```

### Why there is a cap

The ATR-based formula divides 1% of capital by the 15-minute ATR. On low-volatility bars, ATR can be small enough to produce position sizes with notional value many times the allocated capital (effectively 5–10× leverage). The cap enforces that **no single position can exceed the capital allocated to that instrument**.

For SPY at ~$545 with $100k allocated:
- Maximum shares = floor(100,000 / 545) = **183 shares**
- Maximum notional = **~$99,900** (≤ $100k ✓)

### Capital allocation

Each symbol runs as an independent allocation. Total capital = $100k × number of symbols.

| Allocation | Amount |
|---|---|
| SPY | $100,000 |
| QQQ | $100,000 |
| **Total** | **$200,000** |

---

## Backtest Results

**Period:** 2024-06-23 → 2026-06-20 (≈ 2 years)  
**Starting capital:** $200,000 ($100k per symbol)  
**Data source:** Alpaca IEX feed, 15-minute bars, raw (unadjusted)

### Performance summary

| Metric | Value |
|---|---|
| CAGR | +20.60% |
| Sharpe ratio (ann.) | 3.07 |
| Sortino ratio (ann.) | 10.26 |
| Calmar ratio | 4.00 |
| Profit factor | 1.27× |
| Max drawdown | −5.15% |
| Total return | +45.18% |
| Total P&L | +$90,364 |
| Trades | 2,611 |
| Win rate | 33.6% |
| Avg win | $483 |
| Avg loss | −$193 |
| Worst losing streak | 19 consecutive losses (−$2,307 total) |

### Monthly returns

```
  Year     Jan     Feb     Mar     Apr     May     Jun     Jul     Aug     Sep     Oct     Nov     Dec   Annual
  2024      --      --      --      --      --   +1.2%   +1.7%   +3.3%   +2.5%   +4.3%   +0.6%   +2.7%   +17.5%
  2025   +1.9%   +0.7%   +1.8%   -1.6%   +4.4%   +1.7%   +2.0%   -0.6%   +1.5%   +2.1%   -3.5%   +4.7%   +16.0%
  2026   +1.0%   -0.4%   +1.0%   +2.3%   +1.0%   +1.4%                                                    +6.5%
```

4 losing months out of 24. Worst single month: November 2025 (−3.5%).

### Equity curve shape

Steady monotonic climb with shallow pullbacks. The −5.15% max drawdown occurred intra-trade (measured trade-by-trade), not at a monthly boundary. The monthly curve shows no month worse than −3.5%.

---

## Known Limitations

### 1. No transaction costs or slippage
The backtest assumes zero commissions and fills at exact quoted prices. With 2,611 trades over two years, even $1 commission + $0.01/share slippage on ~100-share average positions would add roughly $30–$50k in costs, reducing total P&L by 33–55%. CAGR impact: roughly −5 to −8 percentage points. This is the most material gap between backtest and live performance.

### 2. SMA exit fills at bar close, not next open
When the close crosses back to the SMA, the trade exits at that bar's close price. In live trading, a market order at close would fill at the next available tick, typically 1–5 cents away on SPY/QQQ. Across many trades this creates a small systematic upward bias in backtest returns (a few basis points per trade).

### 3. Gap-through stop risk not modelled
Stop orders fill at the stop price. If a bar opens below (for longs) or above (for shorts) the stop price due to an overnight gap or news event, the live fill would be at the open price, which could be significantly worse. The backtest assumes stop fills are always at the stop price.

### 4. IEX feed, not consolidated tape
Historical bars are fetched from Alpaca's IEX feed. IEX captures roughly 2–3% of total US equity volume. Price data is generally accurate, but volume figures are not representative of total market volume. The volume-based filters in the disabled strategies (momentum) would be particularly unreliable on IEX data; the mean reversion strategy does not use volume, so this has minimal impact.

### 5. Short selling requires margin approval
The strategy takes short positions on SPY and QQQ. In a live Alpaca paper account, short selling requires the account type to support it. In a live cash account it is not permitted at all. Ensure the account has margin enabled before going live.

### 6. Backtest covers a predominantly bull market
The 2024–2026 window includes a broadly rising equity market. Mean reversion strategies can underperform or generate excessive whipsaw in strongly trending environments; this period may not stress-test the strategy against a prolonged bear market or sideways chop.

### 7. No market regime filter
The strategy runs unconditionally. It does not check whether the market is trending strongly (where mean reversion signals are likely to fail) versus ranging (where they are likely to succeed). Adding a regime filter (e.g., suppress signals when the 50-period SMA slope exceeds a threshold) is a candidate future improvement.

### 8. Per-trade Sharpe, not daily Sharpe
The Sharpe and Sortino ratios are computed per trade (annualised by trade frequency), not per calendar day. This is appropriate for a strategy with many trades per day but produces higher ratios than the standard daily-return Sharpe used by most benchmarks. Numbers are not directly comparable to fund-reported Sharpe ratios.

### 9. Position sizing still allows moderate leverage on very calm days
The notional cap prevents positions exceeding $100k, but the ATR-based formula can still reach the cap on very low-volatility bars. On a typical day the formula produces 50–150 shares ($27k–$82k notional), well within the cap. On quiet Fridays or post-holiday sessions, the cap may bind. This is expected and acceptable behaviour.

---

## Telegram Notifications

The bot sends structured Telegram messages for all significant events. All
notifications are fire-and-forget — failures are logged at WARNING level and
never affect bot execution.

**Setup:** Add to `.env` (see `TELEGRAM_SETUP.md` for full walkthrough):
```
TELEGRAM_BOT_TOKEN=<token from @BotFather>
TELEGRAM_CHAT_ID=<your chat ID>
TELEGRAM_ENABLED=true        # set to "false" to silence without removing tokens
ENVIRONMENT=paper            # paper | live | local | dry_run | test
```

**Connectivity test** (always sends, bypasses suppression):
```bash
python -m bot.telegram_notifier
```

### Suppression rules

Notifications are suppressed — and no HTTP request is made — when **any** of:
- `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` missing from `.env`
- `TELEGRAM_ENABLED=false`
- `ENVIRONMENT=test` or `ENVIRONMENT=dry_run`
- Running under pytest (`PYTEST_CURRENT_TEST` is detected at send time)

Tests are also guarded by an autouse `telegram_mock` fixture in
`tests/conftest.py` that patches `notifier.send` before every test — a
belt-and-suspenders defence against any test accidentally reaching the live API.

### Event coverage

| Event | Trigger | Message type |
|---|---|---|
| Bot startup | `main()` after Alpaca connects | Health-check: account, equity, paper/live mode, symbols |
| Bot shutdown | `KeyboardInterrupt` | Reason + timestamp |
| Order submitted | Market order accepted by broker | Symbol, side, qty, est. price, notional (shows "market order" if price unavailable) |
| Trade entered | Fill confirmed + stop order placed | Symbol, direction, qty, fill price, notional, stop, risk $ |
| Stop order placed | Stop GTC order accepted | Symbol, side, qty, stop price |
| Stop-loss triggered | `trailing_stop_triggered()` → close | Symbol, direction, qty, price, P&L |
| SMA exit | Price returns to SMA → close | Symbol, direction, qty, price, P&L |
| Critical error | Stop order fails after fill / unhandled loop error | Context + error detail |
| Daily P&L | First tick of each UTC day | Date, daily P&L, total equity |
| Weekly summary | First tick of Monday UTC | Week range, weekly P&L, equity, trade count, win rate |

### Implementation

| File | Role |
|---|---|
| `bot/telegram_notifier.py` | `TelegramNotifier` class + module-level `notifier` singleton |
| `bot/main.py` | Startup, shutdown, critical errors, weekly summary |
| `bot/portfolio.py` | Daily P&L notification inside `log_daily_pnl()` |
| `bot/strategies/mean_reversion.py` | All trade lifecycle events |
| `tests/conftest.py` | Autouse fixture — patches `notifier.send` in every test |

---

## Failure-Mode Fixes (applied before paper trading)

Five critical/high blockers identified in the failure-mode review were fixed. All
fixes are covered by `tests/test_failure_modes.py` (31 tests, all passing).

### Fix 1 — PositionFetchError: never treat a failed API call as a flat book

**Was:** `Portfolio._live_positions()` caught exceptions and returned `{}`.
`is_long()` and `is_short()` both returned `False`, making the bot believe it was
flat during any Alpaca outage. This caused phantom duplicate entries.

**Now:** `_live_positions()` raises `PositionFetchError` on failure.
`mean_reversion.run()` calls the new `position_state()` method (one API call) inside
a `try/except PositionFetchError` block and returns immediately on failure.

### Fix 2 — Emergency close if stop order fails after market fill

**Was:** Both the market order and the stop order were submitted inside a single
`try/except`. If the market order filled but the stop order failed, the position
was left open with no protection and no in-memory tracking.

**Now:** The two order submissions are separated into distinct `try/except` blocks.
If the stop order fails, `logger.critical(...)` is logged and `api.close_position()`
is called immediately. `record_entry()` is only reached if both orders succeed.
The same fix is applied to `trend_following.py` and `momentum_breakout.py`.

### Fix 3 — Partial fills: stop order uses actual filled quantity

**Was:** The stop order quantity was always the requested size, not the actual filled
size. If a partial fill occurred, the stop order would attempt to sell more shares
than owned, causing Alpaca to reject it and falling into the Fix 2 scenario.

**Now:** A `_filled_qty(order, requested)` helper reads `order.filled_qty` when
available and positive. The stop order quantity and `record_entry()` both use the
actual filled quantity. Falls back to requested size if `filled_qty` is absent.

### Fix 4 — Position state persists to disk; reconciled on restart

**Was:** All five portfolio dicts (`entry_prices`, `entry_sizes`, `entry_directions`,
`trailing_stops`, `stop_order_ids`) were plain in-memory dicts. A bot restart
lost all state: P&L records showed $0 on size 0, stop tracking was gone, and the
orphaned broker stop order was not cancelled before the SMA exit.

**Now:** `Portfolio` persists state to `position_state.json` (`POSITION_STATE_FILE`
in `config.py`) on every `record_entry()`, `clear_position()`, and stop-order
replace. On `__init__`, `_load_state()` reads the file and `_reconcile_with_broker()`
cross-checks against live Alpaca positions:
- Symbol in state but not in Alpaca → position closed offline → state cleared
- Direction mismatch → logged as CRITICAL → state cleared
- API unreachable at startup → state retained from disk; reconciliation deferred

### Fix 5 — 60-minute stop-out cooldown prevents immediate re-entry

**Was:** After a stop-loss exit, if price remained below the lower band, the bot
re-entered on the very next tick — continuing to add to a losing direction.

**Now:** `portfolio.record_stop_out(symbol)` is called after every stop-triggered
close. `in_stop_cooldown(symbol)` returns `True` for `STOP_COOLDOWN_SECONDS`
(60 minutes, configurable in `config.py`). The cooldown timestamp is stored in
`position_state.json` and survives restarts.

### Remaining medium/low risks (not blocking for paper trading)

| Risk | Impact | Status |
|---|---|---|
| `_fill_price` returns 0 on double API failure | Corrupt entry price in trades.csv | Low probability on liquid ETFs |
| Connection drop mid-order-submission | Unknown order state for 1 tick; self-heals | Acceptable for paper trading |
| `close_position()` fires while orphaned stop is pending | Potential double-close error (caught and logged) | Mitigated by Fix 4 restart reconciliation |
| No circuit breaker / daily loss limit | Bot trades through drawdowns | Document; add before going live |
| No duplicate-process guard | Two instances could double-enter | Operational discipline |

---

## Why the Other Strategies Were Removed

### BTC/USD Momentum Breakout — removed, no demonstrated edge

Backtest results over 2 years (437 trades):

| Metric | Value |
|---|---|
| Win rate | 28.6% |
| Profit factor | 0.49× |
| Ann. return | −69% |
| Max drawdown | −92.7% |
| Sharpe | −4.16 |

A profit factor below 1.0 across 437 trades is statistically significant — this is a negative-expectation strategy, not noise. The trailing-stop exit (2× ATR) pairs badly with Bitcoin's high intraday volatility: stops are hit frequently before any breakout can develop, producing a long string of small losses that compound into capital destruction. The $100k BTC allocation would have been nearly wiped out.

The position sizing fix (capping notional to allocated capital) had no material effect on BTC results because Bitcoin's large ATR already produced small fractional sizes that were within the cap. The strategy's poor performance is structural, not a sizing artefact.

### GLD/USO Trend Following — removed, insufficient signal frequency

Backtest results over 2 years (8 trades total):

| Metric | Value |
|---|---|
| Trades | 8 (4 per symbol) |
| Ann. return | −0.4% |
| Max drawdown | −3.0% |
| Sharpe | −0.11 |

A 50/200 EMA crossover on 4-hour bars produces roughly 2 crossovers per year per symbol in low-volatility commodities like GLD and USO. This is too infrequent to generate meaningful returns or to evaluate statistical edge. With only 8 trades in 2 years, the results have no statistical significance in either direction.

The strategy is not inherently wrong, but the chosen timeframe and instruments are mismatched. To be viable, the strategy would need either faster EMAs (e.g., 10/50 on 1-hour bars) or more volatile instruments.

---

## Today's Progress (2026-06-23)

- **Backtest finalized** — 2-year run (2024-06-23 → 2026-06-20) on SPY/QQQ mean reversion confirmed positive edge: 20.60% CAGR, 3.07 Sharpe, −5.15% max drawdown.
- **Position sizing bug fixed in `bot/backtest.py`** — ATR-based sizing was uncapped, producing 5–10× leverage on low-volatility bars and inflating CAGR to 237%. Added `cap_size = floor(allocated_capital / entry_price)` and take `min(raw_size, cap_size)`. Backtest numbers in this document reflect the fixed version.
- **BTC and trend following disabled** — BTC momentum had a 0.49 profit factor across 437 trades (negative expectation); GLD/USO trend following produced only 8 trades in 2 years (insufficient frequency). Both strategies are removed from `main.py` but code remains.
- **Five failure-mode fixes applied and tested** — PositionFetchError propagation, emergency close on stop-order failure, partial-fill handling, position state persistence to disk, 60-minute stop-out cooldown. All covered by 31 tests in `tests/test_failure_modes.py`.
- **Telegram notifications integrated** — full event coverage (startup, entries, exits, stop-losses, daily P&L, weekly summary) with suppression logic for test/dry-run environments.

---

## Next Steps

### Before paper trading (blockers)

1. **Fix position sizing cap in `bot/risk_manager.py`** — the live bot has the same uncapped ATR sizing bug fixed in the backtest. Apply the same `min(raw_size, floor(capital / price))` cap. This is the most critical gap between the validated backtest and the live code path.

2. **Add daily loss circuit breaker** — currently the bot trades through any drawdown with no daily loss limit. Add a configurable `MAX_DAILY_LOSS` threshold (e.g., −2% of equity) that halts all new entries for the rest of the day when breached.

### After paper trading starts

3. **Add slippage/commission estimate to backtest** — model $1 commission + $0.01/share slippage (≈ $30–50k on 2,611 trades) to get a realistic CAGR floor (~13–15%). This is the most material unmodeled cost.

4. **Add duplicate-process guard** — a stale PID file or lock at startup prevents two bot instances from running simultaneously and double-entering positions.

5. **Verify short selling account type** — confirm the Alpaca paper account has margin enabled before the bot attempts its first short position on SPY/QQQ.

### Longer-term improvements (post live deployment)

6. **Add market regime filter** — suppress mean-reversion signals when the 50-period SMA slope exceeds a threshold (strong trend → avoid). Candidate for reducing whipsaw losses in trending months.

7. **Re-evaluate trend following on faster timeframe** — if a momentum strategy is desired, test a 10/50 EMA crossover on 1-hour bars for more liquid instruments (e.g., SPY, QQQ, or sector ETFs). The current 50/200 on 4H is too slow for these instruments.

8. **Build walk-forward validation** — split backtest into in-sample (2024–2025) and out-of-sample (2026) windows to confirm the edge is not curve-fitted to the 2-year period.
