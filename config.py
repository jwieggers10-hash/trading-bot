import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Deployment environment — controls Telegram suppression and future env-specific logic.
# Valid values: paper | live | local | dry_run | test
ENVIRONMENT = os.getenv("ENVIRONMENT", "paper").lower()

# Set to "false" to silence Telegram without removing the token from .env.
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

# Strategy selector — set STRATEGY_VERSION=2 in .env to run Strategy 2.0
# 1 = original SPY/QQQ mean reversion (2 symbols, fixed cap)
# 2 = pooled SPY/QQQ/GLD/USO mean reversion (4 symbols, dynamic cap, 0.25% risk)
STRATEGY_VERSION = int(os.getenv("STRATEGY_VERSION", "1"))

# Strategy symbol assignments
MEAN_REVERSION_SYMBOLS = ["SPY", "QQQ"]
MOMENTUM_BREAKOUT_SYMBOLS = ["BTC/USD"]
TREND_FOLLOWING_SYMBOLS = ["GLD", "USO"]
CRYPTO_SYMBOLS = ["BTC/USD"]

# Mean Reversion — 15-minute candles, 20-period SMA
MEAN_REVERSION_TIMEFRAME = "15Min"
MEAN_REVERSION_PERIOD = 20
STD_DEV_THRESHOLDS = {"SPY": 1.5, "QQQ": 1.8}

# Strategy 2.0 — Pooled-capital dynamic mean reversion across 4 symbols
# Thresholds and risk % derived from grid-search backtest (2020-2026, 5.9 years).
# Dynamic cap: each symbol is allocated equity/4 of the live account, compounding.
# Risk per trade: 0.25% of total portfolio equity risked on a 1-ATR adverse move.
MR_V2_SYMBOLS    = ["SPY", "QQQ", "GLD", "USO"]
MR_V2_THRESHOLDS = {"SPY": 0.25, "QQQ": 0.30, "GLD": 0.25, "USO": 0.25}
MR_V2_RISK_PCT   = 0.0025   # 0.25% of total portfolio equity per trade

# Momentum Breakout — 1-hour candles, 20-period channel
MOMENTUM_TIMEFRAME = "1Hour"
MOMENTUM_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
MOMENTUM_TRAILING_ATR = 2.0

# Trend Following — 4-hour candles (fetched as 1H and resampled), 50/200 EMA
TREND_FAST_EMA = 50
TREND_SLOW_EMA = 200
TREND_TRAILING_ATR = 3.0

# Risk management
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01       # 1% of equity per 1 ATR move
MAX_STOP_LOSS_PCT = 0.01    # hard stop at 1% of equity
CAPITAL_PER_SYMBOL = 100_000  # notional cap per instrument; mirrors backtest --equity default

# Strategy check intervals in seconds
MEAN_REVERSION_INTERVAL = 15 * 60
MOMENTUM_INTERVAL = 60 * 60
TREND_INTERVAL = 4 * 60 * 60

# Logging
TRADES_LOG = "trades.csv"
DAILY_PNL_LOG = "daily_pnl.csv"
BOT_LOG = "trading_bot.log"

# State persistence
POSITION_STATE_FILE = "position_state.json"
STOP_COOLDOWN_SECONDS = 60 * 60  # block re-entry for 60 minutes after a stop-out

# Daily alive heartbeat — sent via Telegram once per day at/after this hour (America/New_York)
HEARTBEAT_HOUR_ET = 9

# Daily P&L Telegram notification schedule (Europe/Amsterdam hours, 24-hour clock)
PNL_NOTIFY_HOURS_AMS = frozenset({9, 22})
