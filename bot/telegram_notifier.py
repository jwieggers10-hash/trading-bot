"""Telegram notification helper for the trading bot.

Configuration (.env):
  TELEGRAM_BOT_TOKEN  — token issued by @BotFather
  TELEGRAM_CHAT_ID    — your personal or group chat ID
  TELEGRAM_ENABLED    — "false" to silence all notifications without removing tokens
  ENVIRONMENT         — "paper" | "live" | "local" | "dry_run" | "test"
                        Notifications are suppressed when set to "test" or "dry_run".

When the two credential vars are absent every call is a silent no-op.
Notifications are also automatically suppressed while running under pytest
(detected via the PYTEST_CURRENT_TEST env var that pytest sets per-test).

Usage:
  from bot.telegram_notifier import notifier
  notifier.startup(account_id, equity, status, base_url, symbols)
  notifier.trade_entry_filled("SPY", "long", 136, 521.40, 518.00)

Stand-alone connectivity test (bypasses environment suppression):
  python -m bot.telegram_notifier
"""
import html as _html
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")
_TIMEOUT_SECONDS = 10
_MAX_LENGTH = 4096  # Telegram hard cap per message

# Non-production environments where notifications are suppressed by default.
_SILENT_ENVIRONMENTS = {"test", "dry_run"}

logger = logging.getLogger(__name__)


def _esc(value) -> str:
    return _html.escape(str(value))


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _mode_label(base_url: str) -> str:
    return "📝 Paper" if "paper" in base_url.lower() else "⚡ Live"


class TelegramNotifier:
    """Send formatted trade notifications to a Telegram chat.

    Instantiate once as a module-level singleton; all methods return bool and
    never raise — notification failures must not affect the bot.
    """

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self._token = token or _TOKEN
        self._chat_id = chat_id or _CHAT_ID
        # True when credentials are present; does NOT mean sends are enabled.
        self._configured = bool(self._token and self._chat_id)

    # ------------------------------------------------------------------
    # Suppression check (evaluated at call time, not at construction time)
    # ------------------------------------------------------------------

    def _should_send(self) -> bool:
        """Return True only when all suppression conditions are clear."""
        if not self._configured:
            return False
        # TELEGRAM_ENABLED=false disables all sends without removing tokens.
        if os.getenv("TELEGRAM_ENABLED", "true").lower() == "false":
            return False
        # Non-production environments.
        if os.getenv("ENVIRONMENT", "paper").lower() in _SILENT_ENVIRONMENTS:
            return False
        # pytest sets PYTEST_CURRENT_TEST before every test function.
        if os.getenv("PYTEST_CURRENT_TEST"):
            return False
        return True

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def send(self, text: str) -> bool:
        """Send an HTML-formatted message. Returns True on success."""
        if not self._should_send():
            logger.debug(
                "Telegram suppressed (env=%s, enabled=%s, pytest=%s)",
                os.getenv("ENVIRONMENT", "paper"),
                os.getenv("TELEGRAM_ENABLED", "true"),
                bool(os.getenv("PYTEST_CURRENT_TEST")),
            )
            return False

        if len(text) > _MAX_LENGTH:
            text = text[: _MAX_LENGTH - 20] + "\n...[truncated]"

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("ok"):
                logger.debug("Telegram: sent %d chars", len(text))
                return True
            logger.warning("Telegram API error: %s", body.get("description", body))
            return False
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("description", "")
            except Exception:
                detail = str(exc)
            logger.warning("Telegram HTTP %d: %s", exc.code, detail)
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def startup(
        self,
        account_id: str,
        equity: float,
        account_status: str,
        base_url: str,
        symbols: list[str],
    ) -> bool:
        mode = _mode_label(base_url)
        text = (
            "🤖 <b>Trading Bot Started</b>\n"
            f"Account: <code>{_esc(account_id)}</code> | Status: {_esc(account_status)}\n"
            f"Equity: ${equity:,.2f}\n"
            f"Mode: {mode}\n"
            f"Symbols: {_esc(', '.join(symbols))}\n"
            "Telegram: ✅ Connected\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def shutdown(self, reason: str = "Keyboard interrupt") -> bool:
        text = (
            "🛑 <b>Trading Bot Stopped</b>\n"
            f"Reason: {_esc(reason)}\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def heartbeat(
        self,
        equity: float,
        base_url: str,
        symbols: list[str],
        check_time_et: str,
        broker_ok: bool = True,
    ) -> bool:
        broker_line = "Broker: Connected" if broker_ok else "Broker: ❌ Unreachable"
        mode = "Paper Trading" if "paper" in base_url.lower() else "Live Trading"
        equity_line = f"${equity:,.0f}" if broker_ok else "N/A"
        symbol_lines = "\n".join(f"• {_esc(s)}" for s in symbols)
        text = (
            "🟢 <b>Trading Bot Alive</b>\n"
            "\n"
            f"Server: Online\n"
            f"{broker_line}\n"
            f"{mode}\n"
            f"Equity: {equity_line}\n"
            "\n"
            f"Watching:\n"
            f"{symbol_lines}\n"
            "\n"
            f"Last check:\n"
            f"{_esc(check_time_et)} ET"
        )
        return self.send(text)

    # ------------------------------------------------------------------
    # Trade lifecycle events
    # ------------------------------------------------------------------

    def trade_entry_submitted(
        self,
        symbol: str,
        side: str,
        qty: int | float,
        est_price: float,
    ) -> bool:
        if est_price > 0:
            price_detail = (
                f"Qty: {qty} | Est. price: ${est_price:,.4f}\n"
                f"Notional: ~${qty * est_price:,.0f}"
            )
        else:
            price_detail = f"Qty: {qty} | Market order (price pending)"
        text = (
            "📋 <b>Order Submitted</b>\n"
            f"Symbol: {_esc(symbol)} | Side: {side.upper()}\n"
            f"{price_detail}\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def trade_entry_filled(
        self,
        symbol: str,
        direction: str,
        qty: int | float,
        fill_price: float,
        stop: float,
    ) -> bool:
        notional = qty * fill_price
        stop_risk = abs(fill_price - stop) * qty
        arrow = "▲ LONG" if direction == "long" else "▼ SHORT"
        text = (
            "✅ <b>Trade Entered</b>\n"
            f"{_esc(symbol)} {arrow} × {qty} @ ${fill_price:,.4f}\n"
            f"Notional: ${notional:,.0f}\n"
            f"Stop: ${stop:,.4f} (risk ${stop_risk:,.2f})\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def stop_order_submitted(
        self,
        symbol: str,
        side: str,
        qty: int | float,
        stop_price: float,
    ) -> bool:
        text = (
            "🛡 <b>Stop Order Placed</b>\n"
            f"{_esc(symbol)} {side.upper()} × {qty} @ ${stop_price:,.4f}\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def stop_loss_triggered(
        self,
        symbol: str,
        direction: str,
        price: float,
        qty: int | float,
        pnl: float,
    ) -> bool:
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"−${abs(pnl):,.2f}"
        side_label = "LONG" if direction == "long" else "SHORT"
        text = (
            "🔴 <b>Stop-Loss Hit</b>\n"
            f"{_esc(symbol)} {side_label} × {qty} | Price: ${price:,.4f}\n"
            f"P&amp;L: {pnl_str}\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def sma_exit(
        self,
        symbol: str,
        direction: str,
        price: float,
        qty: int | float,
        pnl: float,
    ) -> bool:
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"−${abs(pnl):,.2f}"
        side_label = "LONG" if direction == "long" else "SHORT"
        text = (
            "🟢 <b>SMA Exit</b>\n"
            f"{_esc(symbol)} {side_label} × {qty} | Price: ${price:,.4f}\n"
            f"P&amp;L: {pnl_str}\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def critical_error(self, context: str, detail: str = "") -> bool:
        text = (
            "🚨 <b>CRITICAL ERROR</b>\n"
            f"{_esc(context)}"
            + (f"\n{_esc(detail)}" if detail else "")
            + f"\nTime: {_ts()}"
        )
        return self.send(text)

    # ------------------------------------------------------------------
    # Summary notifications
    # ------------------------------------------------------------------

    def daily_pnl(self, date_str: str, pnl: float, equity: float) -> bool:
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"−${abs(pnl):,.2f}"
        text = (
            "📊 <b>Daily P&amp;L</b>\n"
            f"Date: {_esc(date_str)}\n"
            f"P&amp;L: {pnl_str}\n"
            f"Equity: ${equity:,.2f}\n"
            f"Time: {_ts()}"
        )
        return self.send(text)

    def weekly_summary(
        self,
        week_start: str,
        week_end: str,
        weekly_pnl: float,
        equity: float,
        trade_count: int = 0,
        win_rate: float = 0.0,
    ) -> bool:
        pnl_str = f"+${weekly_pnl:,.2f}" if weekly_pnl >= 0 else f"−${abs(weekly_pnl):,.2f}"
        lines = [
            "📈 <b>Weekly Summary</b>",
            f"Week: {_esc(week_start)} → {_esc(week_end)}",
            f"Weekly P&amp;L: {pnl_str}",
            f"Equity: ${equity:,.2f}",
        ]
        if trade_count > 0:
            lines.append(f"Trades: {trade_count} | Win rate: {win_rate:.1%}")
        lines.append(f"Time: {_ts()}")
        return self.send("\n".join(lines))


# Module-level singleton — import and use from any bot module.
notifier = TelegramNotifier()


def send_telegram(message: str) -> bool:
    """Convenience wrapper — delegates to the module-level notifier."""
    return notifier.send(message)


# ------------------------------------------------------------------
# Connectivity test: python -m bot.telegram_notifier
# Bypasses ENVIRONMENT and TELEGRAM_ENABLED suppression — this is
# an explicit tool for verifying the Telegram connection.
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(name)s | %(message)s")

    print(f"  TELEGRAM_BOT_TOKEN : {'set' if _TOKEN else 'NOT SET'}")
    print(f"  TELEGRAM_CHAT_ID   : {'set' if _CHAT_ID else 'NOT SET'}")
    print(f"  TELEGRAM_ENABLED   : {os.getenv('TELEGRAM_ENABLED', 'true')}")
    print(f"  ENVIRONMENT        : {os.getenv('ENVIRONMENT', 'paper')}")
    print()

    if not _TOKEN or not _CHAT_ID:
        print("[FAIL] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing from .env")
        print("       See TELEGRAM_SETUP.md for setup instructions.")
        sys.exit(1)

    # Override suppression — this tool always sends regardless of ENVIRONMENT or
    # TELEGRAM_ENABLED, because its sole purpose is to verify the connection.
    os.environ["TELEGRAM_ENABLED"] = "true"
    os.environ.pop("PYTEST_CURRENT_TEST", None)
    if os.environ.get("ENVIRONMENT", "paper").lower() in _SILENT_ENVIRONMENTS:
        os.environ["ENVIRONMENT"] = "paper"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ok = notifier.send(
        f"Trading bot — connectivity test\n{ts}\n"
        "If you see this, Telegram notifications are working correctly."
    )
    if ok:
        print("[PASS] Test message delivered. Check your Telegram app.")
    else:
        print("[FAIL] Delivery failed. Check trading_bot.log for details.")
        print("       Common causes: wrong token, wrong chat ID, bot not started.")
        sys.exit(1)
