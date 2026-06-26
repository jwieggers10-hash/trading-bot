# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
# Install dependencies (inside venv)
pip install -r requirements.txt

# Start the bot
python -m bot.main
```

The bot runs as a blocking loop. Stop it with `Ctrl-C`; it writes a final daily P&L entry on exit.

## Environment

Copy `.env` and fill in real values before running:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # switch to live URL for production
```

## Architecture

```
config.py           — all tunable constants (thresholds, timeframes, intervals, paths)
bot/
  main.py           — event loop; calls each strategy on its own interval timer
  risk_manager.py   — ATR calculation, position sizing (1 ATR = 1% equity), hard stop price
  portfolio.py      — live position queries, trailing stop ratcheting, broker stop orders,
                      trades.csv and daily_pnl.csv writers
  strategies/
    mean_reversion.py     — SPY/QQQ, 15-min candles, SMA ± N·σ bands
    momentum_breakout.py  — BTC/USD, 1-hour candles, 20-period channel + volume filter
    trend_following.py    — GLD/USO, 4-hour candles (resampled from 1H), 50/200 EMA crossover
```

### Data flow

1. `main.py` fires a strategy's `run(symbol)` on the configured interval.
2. The strategy fetches OHLCV bars from Alpaca's data API v2 and computes indicators.
3. `RiskManager.calculate_atr()` + `calculate_position_size()` size every trade so 1 ATR loss = 1% equity.
4. On entry: a market order is submitted, then a separate `stop` order is placed with Alpaca (broker-side hard stop).
5. `Portfolio.record_entry()` stores the entry price, size, direction, stop price, and stop-order ID.
6. On subsequent ticks: `Portfolio.update_trailing_stop()` ratchets the stop and replaces the broker stop order if it moves.
7. On exit: the pending stop order is cancelled before `close_position()` to prevent double-fill.
8. `Portfolio.log_trade()` writes one row to `trades.csv`.

### Correlation filter

`Portfolio.blocks_new_long("BTC/USD")` returns `True` when both SPY and QQQ are simultaneously long, preventing new risk-on BTC/USD longs.

### Crypto vs equity handling

- BTC/USD uses `time_in_force="gtc"` and `api.get_crypto_bars()`; the strategy falls back to `api.get_bars("BTCUSD", ...)` automatically if `get_crypto_bars` is unavailable.
- Equity strategies check `api.get_clock().is_open` before running; BTC/USD always runs.

### 4-hour bar resampling

`trend_following.py` fetches 1-hour bars going back 400 days and resamples to 4H in pandas (offset `"30min"` to align to 9:30 ET open). This avoids depending on broker-side 4H candle support.

## Key constants (config.py)

| Constant | Default | Purpose |
|---|---|---|
| `STD_DEV_THRESHOLDS` | SPY=1.5, QQQ=1.8 | Entry band width for mean reversion |
| `MOMENTUM_TRAILING_ATR` | 2.0 | Trailing stop multiplier for BTC/USD |
| `TREND_TRAILING_ATR` | 3.0 | Trailing stop multiplier for GLD/USO |
| `RISK_PER_TRADE` | 0.01 | Fraction of equity risked per trade |
| `ATR_PERIOD` | 14 | Lookback for ATR calculation |

## Output files

| File | Contents |
|---|---|
| `trades.csv` | timestamp, instrument, direction, entry_price, exit_price, profit_loss, position_size |
| `daily_pnl.csv` | date, daily_pnl, total_equity |
| `trading_bot.log` | timestamped INFO/ERROR log of all signals, orders, and errors |
