import csv
import json
import logging
import os
from datetime import datetime, timezone

from config import (
    DAILY_PNL_LOG,
    POSITION_STATE_FILE,
    STOP_COOLDOWN_SECONDS,
    TRADES_LOG,
)
from bot.telegram_notifier import notifier

logger = logging.getLogger(__name__)


class PositionFetchError(Exception):
    """Raised when Alpaca position data cannot be fetched.

    Callers must catch this and skip the tick rather than assume a flat book,
    otherwise a failed list_positions() call would look identical to "no open
    positions" and trigger a phantom duplicate entry.
    """


class Portfolio:
    """Tracks open positions, trailing stops, stop-order IDs, and writes audit logs.

    Position state is persisted to POSITION_STATE_FILE so that a bot restart
    can recover stop prices, entry prices, and stop-out cooldowns without losing
    track of live broker positions.
    """

    def __init__(self, api, state_file: str | None = None):
        self.api = api
        self._state_file = state_file if state_file is not None else POSITION_STATE_FILE

        # symbol -> entry price
        self.entry_prices: dict[str, float] = {}
        # symbol -> actual filled position size (shares or fractional units)
        self.entry_sizes: dict[str, float] = {}
        # symbol -> entry direction ("long" | "short")
        self.entry_directions: dict[str, str] = {}
        # symbol -> current trailing stop price
        self.trailing_stops: dict[str, float] = {}
        # symbol -> Alpaca stop-order ID (broker-side hard stop)
        self.stop_order_ids: dict[str, str] = {}
        # symbol -> UTC datetime of last stop-out (for re-entry cooldown)
        self._stop_out_times: dict[str, datetime] = {}

        self._load_state()
        self._reconcile_with_broker()
        self._init_log_files()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sym, pos in data.get("positions", {}).items():
                self.entry_prices[sym] = float(pos["entry_price"])
                self.entry_sizes[sym] = float(pos["entry_size"])
                self.entry_directions[sym] = pos["direction"]
                self.trailing_stops[sym] = float(pos["trailing_stop"])
                self.stop_order_ids[sym] = pos["stop_order_id"]

            for sym, ts in data.get("stop_cooldowns", {}).items():
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                self._stop_out_times[sym] = dt

            logger.info(
                "Loaded position state from %s: %d open position(s), %d cooldown(s)",
                self._state_file, len(self.entry_prices), len(self._stop_out_times),
            )
        except Exception as exc:
            logger.error("Failed to load position state from %s: %s — starting fresh",
                         self._state_file, exc)

    def _save_state(self):
        try:
            data = {
                "positions": {
                    sym: {
                        "entry_price": self.entry_prices[sym],
                        "entry_size": self.entry_sizes[sym],
                        "direction": self.entry_directions[sym],
                        "trailing_stop": self.trailing_stops[sym],
                        "stop_order_id": self.stop_order_ids[sym],
                    }
                    for sym in self.entry_prices
                },
                "stop_cooldowns": {
                    sym: dt.isoformat()
                    for sym, dt in self._stop_out_times.items()
                },
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save position state to %s: %s", self._state_file, exc)

    def _reconcile_with_broker(self):
        """On startup, sync in-memory state with live Alpaca positions.

        Clears local state for positions that closed while the bot was offline,
        and logs a warning for broker positions that have no local state record
        (orphaned positions still exit correctly via SMA logic, but lack stop
        tracking until the bot is manually restarted with a clean state).
        """
        if not self.entry_prices:
            return

        try:
            live = {p.symbol: p for p in self.api.list_positions()}
        except Exception as exc:
            logger.warning(
                "Startup reconciliation skipped (cannot reach Alpaca): %s — "
                "local state retained from disk; will reconcile on first successful tick.",
                exc,
            )
            return

        for sym in list(self.entry_prices.keys()):
            if sym not in live:
                logger.warning(
                    "%s: In local state but absent from Alpaca — "
                    "closed while bot was offline. Clearing local state.", sym,
                )
                self.clear_position(sym)
            else:
                local_dir = self.entry_directions.get(sym)
                live_qty = float(live[sym].qty)
                live_dir = "long" if live_qty > 0 else "short"
                if local_dir != live_dir:
                    logger.critical(
                        "%s: Direction mismatch — local=%s Alpaca=%s. Clearing state.",
                        sym, local_dir, live_dir,
                    )
                    self.clear_position(sym)
                else:
                    logger.info(
                        "%s: Reconciled — %s qty=%.6f stop=%.4f",
                        sym, live_dir, live_qty, self.trailing_stops.get(sym, 0),
                    )

        for sym in live:
            if sym not in self.entry_prices:
                logger.warning(
                    "%s: Live position on Alpaca but no local state — "
                    "SMA exits will work; stop-price tracking is absent until restart.",
                    sym,
                )

    # ------------------------------------------------------------------
    # Log file setup
    # ------------------------------------------------------------------

    def _init_log_files(self):
        if not os.path.exists(TRADES_LOG):
            with open(TRADES_LOG, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "instrument", "direction", "entry_price",
                     "exit_price", "profit_loss", "position_size"]
                )
        if not os.path.exists(DAILY_PNL_LOG):
            with open(DAILY_PNL_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["date", "daily_pnl", "total_equity"])

    # ------------------------------------------------------------------
    # Position queries (always reads live from Alpaca)
    # ------------------------------------------------------------------

    def _live_positions(self) -> dict[str, object]:
        """Fetch live positions from Alpaca.

        Raises PositionFetchError on any API failure so callers cannot
        accidentally treat a failed query as an empty book.
        """
        try:
            return {p.symbol: p for p in self.api.list_positions()}
        except Exception as exc:
            logger.error("Failed to fetch live positions: %s", exc)
            raise PositionFetchError(str(exc)) from exc

    def position_state(self, symbol: str) -> str:
        """Return 'long', 'short', or 'flat' for *symbol* via a single API call.

        Raises PositionFetchError when positions cannot be queried.  Callers
        must handle this by skipping the current tick.
        """
        positions = self._live_positions()
        if symbol not in positions:
            return "flat"
        qty = float(positions[symbol].qty)
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return "flat"

    def is_long(self, symbol: str) -> bool:
        return self.position_state(symbol) == "long"

    def is_short(self, symbol: str) -> bool:
        return self.position_state(symbol) == "short"

    def has_position(self, symbol: str) -> bool:
        return self.position_state(symbol) != "flat"

    def current_qty(self, symbol: str) -> float:
        pos = self._live_positions()
        return float(pos[symbol].qty) if symbol in pos else 0.0

    # ------------------------------------------------------------------
    # Stop-out cooldown
    # ------------------------------------------------------------------

    def record_stop_out(self, symbol: str):
        """Record the time of a stop-out to enforce STOP_COOLDOWN_SECONDS cooldown."""
        self._stop_out_times[symbol] = datetime.now(timezone.utc)
        self._save_state()
        logger.info(
            "%s: Stop-out recorded — re-entry blocked for %d minutes.",
            symbol, STOP_COOLDOWN_SECONDS // 60,
        )

    def in_stop_cooldown(self, symbol: str) -> bool:
        """Return True if *symbol* is within the post-stop-out cooldown window."""
        dt = self._stop_out_times.get(symbol)
        if dt is None:
            return False
        elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
        return elapsed < STOP_COOLDOWN_SECONDS

    # ------------------------------------------------------------------
    # Correlation filter
    # ------------------------------------------------------------------

    def blocks_new_long(self, symbol: str) -> bool:
        """Return True if the correlation filter prevents a new long on *symbol*.

        Rule: if SPY and QQQ are both long, do not open new BTC/USD longs.
        Fails safe: returns True (blocks) when positions cannot be fetched.
        """
        if symbol != "BTC/USD":
            return False
        try:
            positions = self._live_positions()
            spy_long = "SPY" in positions and float(positions["SPY"].qty) > 0
            qqq_long = "QQQ" in positions and float(positions["QQQ"].qty) > 0
            if spy_long and qqq_long:
                logger.info("Correlation filter: SPY+QQQ both long — blocking BTC/USD long")
                return True
        except PositionFetchError:
            logger.warning("Correlation filter: position fetch failed — blocking BTC/USD long")
            return True
        return False

    # ------------------------------------------------------------------
    # Position state management
    # ------------------------------------------------------------------

    def record_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_order_id: str,
        initial_stop: float,
    ):
        self.entry_prices[symbol] = entry_price
        self.entry_sizes[symbol] = size
        self.entry_directions[symbol] = direction
        self.trailing_stops[symbol] = initial_stop
        self.stop_order_ids[symbol] = stop_order_id
        self._save_state()

    def clear_position(self, symbol: str):
        for store in (
            self.entry_prices,
            self.entry_sizes,
            self.entry_directions,
            self.trailing_stops,
            self.stop_order_ids,
        ):
            store.pop(symbol, None)
        self._save_state()

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        atr_multiplier: float,
        direction: str,
    ):
        """Ratchet stop in the direction of profit. Replaces broker stop order if moved."""
        if symbol not in self.trailing_stops:
            return

        if direction == "long":
            candidate = round(current_price - atr_multiplier * atr, 4)
            if candidate <= self.trailing_stops[symbol]:
                return
        else:
            candidate = round(current_price + atr_multiplier * atr, 4)
            if candidate >= self.trailing_stops[symbol]:
                return

        old_stop = self.trailing_stops[symbol]
        self.trailing_stops[symbol] = candidate
        logger.info("%s: Trailing stop moved %.4f -> %.4f", symbol, old_stop, candidate)

        self._replace_stop_order(symbol, candidate, direction)
        self._save_state()

    def _replace_stop_order(self, symbol: str, new_stop: float, direction: str):
        old_id = self.stop_order_ids.get(symbol)
        if old_id:
            try:
                self.api.cancel_order(old_id)
            except Exception as exc:
                logger.warning("Could not cancel old stop order %s: %s", old_id, exc)

        size = abs(self.entry_sizes.get(symbol, 0))
        if size == 0:
            return

        stop_side = "sell" if direction == "long" else "buy"
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=str(round(size, 6)),
                side=stop_side,
                type="stop",
                time_in_force="gtc",
                stop_price=str(round(new_stop, 4)),
            )
            self.stop_order_ids[symbol] = order.id
            self._save_state()
        except Exception as exc:
            logger.error("Failed to replace stop order for %s: %s", symbol, exc)

    def trailing_stop_triggered(self, symbol: str, current_price: float) -> bool:
        if symbol not in self.trailing_stops:
            return False
        direction = self.entry_directions.get(symbol)
        stop = self.trailing_stops[symbol]
        if direction == "long" and current_price <= stop:
            return True
        if direction == "short" and current_price >= stop:
            return True
        return False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
    ) -> float:
        multiplier = 1.0 if direction == "long" else -1.0
        pnl = round((exit_price - entry_price) * size * multiplier, 2)

        with open(TRADES_LOG, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.utcnow().isoformat(),
                    symbol,
                    direction,
                    round(entry_price, 4),
                    round(exit_price, 4),
                    pnl,
                    round(size, 6),
                ]
            )
        logger.info(
            "TRADE %s %s | entry=%.4f exit=%.4f pnl=$%.2f size=%.6f",
            symbol, direction, entry_price, exit_price, pnl, size,
        )
        return pnl

    def log_daily_pnl(self):
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_pnl = round(equity - last_equity, 2)
            date_str = datetime.utcnow().date().isoformat()

            with open(DAILY_PNL_LOG, "a", newline="") as f:
                csv.writer(f).writerow([date_str, daily_pnl, round(equity, 2)])
            logger.info("Daily P&L: $%.2f | Equity: $%.2f", daily_pnl, equity)
            notifier.daily_pnl(date_str, daily_pnl, equity)
        except Exception as exc:
            logger.error("Failed to log daily P&L: %s", exc)
