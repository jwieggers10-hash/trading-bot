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
from bot.risk_manager import round_stop_price
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
        # symbol -> "candle_ts|direction" of the most recent entry attempt,
        # so a restart cannot re-process the same candle's signal (see
        # entry_signal_already_processed / record_entry_signal below).
        self.last_entry_signal: dict[str, str] = {}

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

            self.last_entry_signal = dict(data.get("last_entry_signal", {}))

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
                "last_entry_signal": dict(self.last_entry_signal),
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
    # Restart-safe entry dedup
    #
    # main.py's per-strategy interval timer (last_mean_rev etc.) is an
    # in-memory float that resets to 0 on every process start, so a restart
    # immediately re-evaluates every symbol's latest candle regardless of how
    # much of that candle's interval had already elapsed. position_state()
    # is a live broker query, so a restart alone cannot duplicate an entry
    # once the position is visible on Alpaca — but there is a real window
    # between submit_order() returning and that fill becoming visible via
    # list_positions(). A crash inside that window followed by an immediate
    # restart would otherwise see the symbol as still "flat" and could submit
    # a second market order for the same signal, doubling the position.
    #
    # record_entry_signal() must be called BEFORE submit_order() so the flag
    # is durable on disk before the network call — that's what closes the
    # race rather than just narrowing it.
    # ------------------------------------------------------------------

    def entry_signal_already_processed(self, symbol: str, candle_ts, direction: str) -> bool:
        """Return True if an entry was already attempted for this exact
        (candle, direction) on *symbol*, including in a prior process
        lifetime (this is loaded from disk)."""
        return self.last_entry_signal.get(symbol) == f"{candle_ts}|{direction}"

    def record_entry_signal(self, symbol: str, candle_ts, direction: str) -> None:
        """Persist that an entry is about to be attempted for (candle, direction)."""
        self.last_entry_signal[symbol] = f"{candle_ts}|{direction}"
        self._save_state()

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

    def get_locked_notional(self, exclude_symbol: str | None = None) -> float:
        """Return the total entry-notional locked in all tracked open positions.

        exclude_symbol: skip this symbol (use when sizing a new entry for it,
        so you're not counting the position you're about to replace).
        """
        return sum(
            self.entry_prices[sym] * abs(self.entry_sizes.get(sym, 0.0))
            for sym in self.entry_prices
            if sym != exclude_symbol
        )

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
            candidate = round_stop_price(current_price - atr_multiplier * atr, "sell", symbol)
            if candidate <= self.trailing_stops[symbol]:
                return
        else:
            candidate = round_stop_price(current_price + atr_multiplier * atr, "buy", symbol)
            if candidate >= self.trailing_stops[symbol]:
                return

        old_stop = self.trailing_stops[symbol]
        self.trailing_stops[symbol] = candidate
        logger.info("%s: Trailing stop moved %.4f -> %.4f", symbol, old_stop, candidate)

        self._replace_stop_order(symbol, candidate, direction)
        self._save_state()

    # Orders in these statuses are done and cannot be cancelled. Everything
    # else — including "held", the status Alpaca gives an OTO/bracket
    # stop_loss leg while it waits for its parent to fill — is still live.
    _TERMINAL_ORDER_STATUSES = frozenset({
        "filled", "canceled", "expired", "rejected", "replaced", "done_for_day",
    })

    def cancel_open_stop_orders(self, symbol: str, side: str) -> int:
        """Cancel all live stop/stop_limit orders for *symbol* on *side*.

        Returns the number of orders cancelled. Called before placing any new
        protective stop to prevent Alpaca error 40310000 (wash-trade detection)
        from a lingering opposite-side order that was not cleaned up — for
        example when a prior cancel_order call failed silently, or the bot
        crashed between cancellation and position close.

        Queries status="all" rather than "open": a stop_loss leg of an
        OTO/bracket order sits in status "held" while waiting on its parent
        to fill, and Alpaca's "open" filter does not include "held" orders —
        so a sweep scoped to "open" would silently miss it.
        """
        try:
            candidate_orders = self.api.list_orders(status="all", symbols=[symbol])
        except Exception as exc:
            logger.error(
                "%s: Cannot list orders for pre-placement sweep (side=%s): %s",
                symbol, side, exc,
            )
            return 0

        cancelled = 0
        for order in candidate_orders:
            raw_status = getattr(order, "status", "")
            status = (raw_status.value if hasattr(raw_status, "value") else str(raw_status)).lower()
            if status in self._TERMINAL_ORDER_STATUSES:
                continue
            raw_type = getattr(order, "type", "")
            raw_side = getattr(order, "side", "")
            order_type = (raw_type.value if hasattr(raw_type, "value") else str(raw_type)).lower()
            order_side = (raw_side.value if hasattr(raw_side, "value") else str(raw_side)).lower()
            if order_type in ("stop", "stop_limit") and order_side == side:
                try:
                    self.api.cancel_order(order.id)
                    cancelled += 1
                    logger.info(
                        "%s: Pre-placement sweep — cancelled existing %s stop %s",
                        symbol, side, order.id,
                    )
                    if self.stop_order_ids.get(symbol) == order.id:
                        self.stop_order_ids.pop(symbol, None)
                except Exception as exc:
                    logger.warning(
                        "%s: Could not cancel stop %s during pre-placement sweep: %s",
                        symbol, order.id, exc,
                    )

        return cancelled

    def _replace_stop_order(self, symbol: str, new_stop: float, direction: str):
        stop_side = "sell" if direction == "long" else "buy"

        old_id = self.stop_order_ids.get(symbol)
        if old_id:
            try:
                self.api.cancel_order(old_id)
            except Exception as exc:
                logger.warning("Could not cancel tracked stop order %s: %s", old_id, exc)

        # Sweep for any untracked stops to prevent 40310000 on the replacement.
        self.cancel_open_stop_orders(symbol, stop_side)

        size = abs(self.entry_sizes.get(symbol, 0))
        if size == 0:
            return

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=str(round(size, 6)),
                side=stop_side,
                type="stop",
                time_in_force="gtc",
                stop_price=str(round_stop_price(new_stop, stop_side, symbol)),
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
        """Write daily P&L snapshot to CSV. Called once per UTC calendar day."""
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_pnl = round(equity - last_equity, 2)
            date_str = datetime.utcnow().date().isoformat()

            with open(DAILY_PNL_LOG, "a", newline="") as f:
                csv.writer(f).writerow([date_str, daily_pnl, round(equity, 2)])
            logger.info("Daily P&L: $%.2f | Equity: $%.2f", daily_pnl, equity)
        except Exception as exc:
            logger.error("Failed to log daily P&L: %s", exc)

    def send_pnl_notification(self):
        """Fetch live P&L from Alpaca and send a Telegram Daily P&L message.

        Called on the scheduled Amsterdam-time notification ticks (09:00 and 22:00).
        Failures are logged and never propagated so the bot loop continues.
        """
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_pnl = round(equity - last_equity, 2)
            date_str = datetime.utcnow().date().isoformat()
            notifier.daily_pnl(date_str, daily_pnl, equity)
        except Exception as exc:
            logger.error("Failed to send P&L notification: %s", exc)
