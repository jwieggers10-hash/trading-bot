"""Trading bot main loop.

Run with:
    python -m bot.main

Intervals:
  Mean Reversion  — every 15 min, equity market hours only
  Momentum        — every 1 hour,  24/7 (crypto)
  Trend Following — every 4 hours, equity market hours only

Daily P&L CSV is written once per UTC calendar day.
Daily P&L Telegram notifications fire at 09:00 and 22:00 Europe/Amsterdam.
"""
import csv as _csv
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bot import alpaca_client as tradeapi

from config import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
    BOT_LOG,
    DAILY_PNL_LOG,
    HEARTBEAT_HOUR_ET,
    MEAN_REVERSION_INTERVAL,
    MEAN_REVERSION_SYMBOLS,
    MOMENTUM_BREAKOUT_SYMBOLS,
    MOMENTUM_INTERVAL,
    PNL_NOTIFY_HOURS_AMS,
    TRADES_LOG,
    TREND_FOLLOWING_SYMBOLS,
    TREND_INTERVAL,
)
from bot.portfolio import Portfolio
from bot.risk_manager import RiskManager
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from bot.strategies.trend_following import TrendFollowingStrategy
from bot.telegram_notifier import notifier

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(BOT_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

_ET = ZoneInfo("America/New_York")
_AMS = ZoneInfo("Europe/Amsterdam")

# ------------------------------------------------------------------
# Market hours helper
# ------------------------------------------------------------------

def _market_is_open(api) -> bool:
    """Return True if the US equity market is currently open."""
    try:
        return bool(api.get_clock().is_open)
    except Exception as exc:
        logger.warning("Could not read market clock: %s — assuming closed", exc)
        return False


def _should_run(symbol: str, api) -> bool:
    """Crypto runs 24/7; equities only when the market is open."""
    from config import CRYPTO_SYMBOLS
    if symbol in CRYPTO_SYMBOLS:
        return True
    return _market_is_open(api)


def _pnl_notify_due(now_ams: datetime, notified: set) -> bool:
    """Return True when a P&L Telegram notification should fire.

    Fires at each hour in PNL_NOTIFY_HOURS_AMS (Europe/Amsterdam) but only
    once per (date, hour) pair so repeated loop ticks within the same minute
    do not trigger duplicate sends.
    """
    return (
        now_ams.hour in PNL_NOTIFY_HOURS_AMS
        and (now_ams.date(), now_ams.hour) not in notified
    )


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------

def _send_heartbeat(api, symbols: list[str]) -> None:
    et_now = datetime.now(_ET)
    check_time = et_now.strftime("%H:%M")
    try:
        account = api.get_account()
        equity = float(account.equity)
        broker_ok = True
    except Exception as exc:
        logger.warning("Heartbeat: cannot reach Alpaca: %s", exc)
        equity = 0.0
        broker_ok = False
    notifier.heartbeat(equity, ALPACA_BASE_URL, symbols, check_time, broker_ok)
    logger.info("Heartbeat sent — equity=$%.0f broker_ok=%s", equity, broker_ok)


def _connect() -> tradeapi.REST:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.critical("ALPACA_API_KEY or ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


def _weekly_stats(today: date) -> tuple[str, str, float, float, int, float]:
    """Read the past 7 days from daily_pnl.csv and trades.csv.

    Returns (week_start, week_end, weekly_pnl, equity, trade_count, win_rate).
    Falls back to zeros on any read error so the summary still sends.
    """
    week_end = (today - timedelta(days=1)).isoformat()
    week_start = (today - timedelta(days=7)).isoformat()

    weekly_pnl = 0.0
    equity = 0.0
    try:
        with open(DAILY_PNL_LOG, newline="") as f:
            rows = list(_csv.DictReader(f))
        recent = [r for r in rows if r["date"] >= week_start]
        if recent:
            weekly_pnl = sum(float(r["daily_pnl"]) for r in recent)
            equity = float(recent[-1]["total_equity"])
    except Exception as exc:
        logger.warning("Weekly summary: could not read %s: %s", DAILY_PNL_LOG, exc)

    trade_count = 0
    win_count = 0
    try:
        with open(TRADES_LOG, newline="") as f:
            rows = list(_csv.DictReader(f))
        recent_trades = [r for r in rows if r["timestamp"][:10] >= week_start]
        trade_count = len(recent_trades)
        win_count = sum(1 for r in recent_trades if float(r["profit_loss"]) > 0)
    except Exception as exc:
        logger.warning("Weekly summary: could not read %s: %s", TRADES_LOG, exc)

    win_rate = win_count / trade_count if trade_count > 0 else 0.0
    return week_start, week_end, weekly_pnl, equity, trade_count, win_rate


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    api = _connect()
    try:
        account = api.get_account()
        logger.info(
            "Connected to Alpaca | Account %s | Equity $%.2f | Status %s",
            account.id, float(account.equity), account.status,
        )
    except Exception as exc:
        logger.critical("Cannot reach Alpaca API: %s", exc)
        sys.exit(1)

    notifier.startup(
        account_id=account.id,
        equity=float(account.equity),
        account_status=account.status,
        base_url=ALPACA_BASE_URL,
        symbols=MEAN_REVERSION_SYMBOLS,
    )

    risk_manager = RiskManager(api)
    portfolio = Portfolio(api)

    mean_rev = MeanReversionStrategy(api, risk_manager, portfolio)
    momentum = MomentumBreakoutStrategy(api, risk_manager, portfolio)
    trend = TrendFollowingStrategy(api, risk_manager, portfolio)

    # Timestamps of last run for each strategy group
    last_mean_rev: float = 0.0
    last_momentum: float = 0.0
    last_trend: float = 0.0
    last_pnl_date: date | None = None
    last_weekly_date: date | None = None
    last_heartbeat_date: date | None = None
    # Tracks (Amsterdam date, hour) pairs for which the P&L notification was sent.
    pnl_notified: set[tuple[date, int]] = set()

    logger.info("Bot started — checking signals every 60 s")

    while True:
        try:
            now = time.monotonic()
            today = datetime.now(timezone.utc).date()

            # Daily P&L snapshot (once per UTC day)
            if last_pnl_date != today:
                portfolio.log_daily_pnl()
                last_pnl_date = today

                # Weekly summary every Monday UTC (covers the previous Mon–Sun)
                if today.weekday() == 0 and last_weekly_date != today:
                    ws, we, wpnl, eq, tc, wr = _weekly_stats(today)
                    notifier.weekly_summary(ws, we, wpnl, eq, tc, wr)
                    last_weekly_date = today

            # Daily heartbeat at/after 09:00 ET (once per ET calendar day)
            et_now = datetime.now(_ET)
            if last_heartbeat_date != et_now.date() and et_now.hour >= HEARTBEAT_HOUR_ET:
                _send_heartbeat(api, MEAN_REVERSION_SYMBOLS)
                last_heartbeat_date = et_now.date()

            # Daily P&L Telegram notification at 09:00 and 22:00 Europe/Amsterdam
            ams_now = datetime.now(_AMS)
            if _pnl_notify_due(ams_now, pnl_notified):
                portfolio.send_pnl_notification()
                pnl_notified.add((ams_now.date(), ams_now.hour))

            # Mean Reversion — SPY, QQQ — every 15 min
            if now - last_mean_rev >= MEAN_REVERSION_INTERVAL:
                for sym in MEAN_REVERSION_SYMBOLS:
                    if _should_run(sym, api):
                        logger.info("--- Mean Reversion: %s ---", sym)
                        mean_rev.run(sym)
                    else:
                        logger.debug("%s: market closed, skipping mean reversion", sym)
                last_mean_rev = now

            # Momentum Breakout (BTC/USD) — DISABLED: negative expectation in backtest
            # Trend Following (GLD/USO)   — DISABLED: too few signals (8 trades / 2 years)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
            portfolio.log_daily_pnl()
            notifier.shutdown("Keyboard interrupt")
            break
        except Exception as exc:
            logger.error("Unhandled error in main loop: %s", exc, exc_info=True)
            notifier.critical_error("Unhandled error in main loop", str(exc))

        time.sleep(60)


if __name__ == "__main__":
    main()
