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
        self._init_log_files()
        self._reconcile_with_broker()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Parse into local variables first and only commit to instance
            # state once the whole file has parsed cleanly — a malformed
            # entry partway through must not leave self.entry_prices etc.
            # half-populated.
            entry_prices, entry_sizes, entry_directions = {}, {}, {}
            trailing_stops, stop_order_ids = {}, {}
            for sym, pos in data.get("positions", {}).items():
                entry_prices[sym] = float(pos["entry_price"])
                entry_sizes[sym] = float(pos["entry_size"])
                entry_directions[sym] = pos["direction"]
                trailing_stops[sym] = float(pos["trailing_stop"])
                stop_order_ids[sym] = pos["stop_order_id"]

            stop_out_times = {}
            for sym, ts in data.get("stop_cooldowns", {}).items():
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                stop_out_times[sym] = dt

            last_entry_signal = dict(data.get("last_entry_signal", {}))

            self.entry_prices = entry_prices
            self.entry_sizes = entry_sizes
            self.entry_directions = entry_directions
            self.trailing_stops = trailing_stops
            self.stop_order_ids = stop_order_ids
            self._stop_out_times = stop_out_times
            self.last_entry_signal = last_entry_signal

            logger.info(
                "Loaded position state from %s: %d open position(s), %d cooldown(s)",
                self._state_file, len(self.entry_prices), len(self._stop_out_times),
            )
        except Exception as exc:
            logger.error("Failed to load position state from %s: %s — starting fresh",
                         self._state_file, exc)
            self._quarantine_corrupt_state_file()

    def _quarantine_corrupt_state_file(self):
        """Rename a malformed state file aside with a timestamp so it isn't
        silently overwritten (kept for forensics) and the next _save_state()
        call starts a fresh, valid file rather than repeatedly failing to
        parse the same corruption."""
        if not os.path.exists(self._state_file):
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = f"{self._state_file}.{ts}.corrupt"
        try:
            os.replace(self._state_file, backup_path)
            logger.warning(
                "Quarantined malformed state file to %s — starting with a clean "
                "in-memory state; the next successful save writes a fresh file.",
                backup_path,
            )
        except OSError as exc:
            logger.error(
                "Failed to quarantine malformed state file %s: %s — it will be "
                "overwritten on the next successful save instead.",
                self._state_file, exc,
            )

    def _save_state(self):
        """Atomically persist state: write to a temp file, fsync, then
        os.replace() onto the real path. If serialization fails partway
        (e.g. a non-JSON-native object slips into the data), the temp file
        is discarded and the last good on-disk state file is left untouched
        — the old code opened the real file in "w" mode directly, so a
        mid-write crash (or exception) truncated it in place with no way
        back; that's exactly how position_state.json ended up permanently
        corrupt in production.
        """
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
        tmp_path = f"{self._state_file}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # default=str is defense in depth (e.g. a UUID slipping in from
                # somewhere else) — the primary fix is casting stop_order_id to
                # str at its source in _find_child_stop_id().
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._state_file)
        except Exception as exc:
            logger.error(
                "Failed to save position state to %s: %s — previous state file "
                "left untouched",
                self._state_file, exc,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _reconcile_with_broker(self):
        """On startup, sync in-memory state with live Alpaca positions.

        This is read-only against the broker: it only ever calls
        list_positions()/list_orders() (via _find_open_stop_order) to look,
        and clear_position()/reconcile_and_correct()/_save_state() to fix up
        *local* bookkeeping. It never submits, cancels, or replaces an order
        — any existing live OTO protective stop is left exactly as-is.

        Three cases per symbol:
          1. Tracked locally but absent from the broker — closed while the
             bot was offline; recovered as an external exit (existing
             behavior, unchanged).
          2. Tracked locally AND present on the broker — direction is
             re-checked (a mismatch is still cleared, unchanged); if
             direction matches, qty/avg_price are compared against the
             broker via reconcile_and_correct() and corrected if they drifted
             (e.g. a fill that continued past a client-side poll timeout —
             see _submit_protected_entry in mean_reversion.py).
          3. Present on the broker but not tracked locally at all — rebuilt
             from broker truth via _rebuild_single_symbol_from_broker(),
             whether local state was entirely empty (fresh/quarantined state
             file) or just missing this one symbol.
        """
        try:
            live = {p.symbol: p for p in self.api.list_positions()}
        except Exception as exc:
            logger.warning(
                "Startup reconciliation skipped (cannot reach Alpaca): %s — "
                "local state retained from disk as-is; will reconcile on "
                "first successful tick instead.",
                exc,
            )
            return

        for sym in list(self.entry_prices.keys()):
            if sym not in live:
                logger.warning(
                    "%s: In local state but absent from Alpaca — "
                    "closed while bot was offline. Recovering exit price "
                    "and logging the trade.", sym,
                )
                self.recover_external_exit(sym, fallback_price=self.entry_prices[sym])
                continue

            local_dir = self.entry_directions.get(sym)
            broker_state = self._position_to_broker_state(live[sym])
            if local_dir != broker_state["direction"]:
                logger.critical(
                    "%s: Direction mismatch — local=%s Alpaca=%s. Clearing state.",
                    sym, local_dir, broker_state["direction"],
                )
                self.clear_position(sym)
                continue

            corrected = self.reconcile_and_correct(sym, broker_state=broker_state)
            if corrected:
                logger.warning(
                    "%s: Startup reconciliation corrected local state to broker "
                    "truth (qty=%.6f avg_price=%.4f) — the live protective stop "
                    "order on Alpaca was not touched.",
                    sym, broker_state["qty"], broker_state["avg_price"],
                )
            else:
                logger.info(
                    "%s: Reconciled — %s qty=%.6f avg_price=%.4f stop=%.4f "
                    "(matches broker, no correction needed)",
                    sym, broker_state["direction"], broker_state["qty"],
                    broker_state["avg_price"], self.trailing_stops.get(sym, 0),
                )

        rebuilt_any = False
        for sym, pos in live.items():
            if sym not in self.entry_prices:
                logger.warning(
                    "%s: Live position on Alpaca but no local state — "
                    "rebuilding local tracking from broker truth.",
                    sym,
                )
                self._rebuild_single_symbol_from_broker(sym, pos)
                rebuilt_any = True

        if rebuilt_any:
            self._save_state()

    def _rebuild_single_symbol_from_broker(self, sym: str, pos) -> None:
        """Rebuild local tracking for one symbol that's open on the broker
        but has no local record at all (empty local state, or just this one
        symbol missing).

        Recovered with certainty, directly from the broker: symbol, side
        (direction), quantity, and average entry price.

        NOT recoverable with certainty: the bot's *intended* trailing-stop
        level (that's derived from ATR at entry time, which isn't stored on
        the broker) and, in the rare case no live stop order is found, the
        stop-order id. This recovers the live protective stop order's id and
        price directly off the broker when one exists (every entry from
        _submit_protected_entry() attaches one via OTO, so this is the common
        case). When no matching stop order is found, it falls back to the
        position's own average entry price as a placeholder trailing_stop and
        logs an explicit CRITICAL — that placeholder is a "needs manual
        verification" marker, not a real risk-managed stop, and must not be
        mistaken for one. Does not call _save_state() itself — callers batch
        that after processing every symbol.
        """
        broker_state = self._position_to_broker_state(pos)
        qty, avg_price, direction = (
            broker_state["qty"], broker_state["avg_price"], broker_state["direction"],
        )

        stop_order_id, stop_price = self._find_open_stop_order(sym, direction)
        if stop_order_id is None:
            stop_price = avg_price
            stop_order_id = ""
            logger.critical(
                "%s: Rebuilt qty=%.6f avg_price=%.4f from broker, but no live "
                "protective stop order was found for it — trailing_stop is a "
                "placeholder (=entry price), NOT a real risk-managed level. "
                "Manually confirm a stop order exists on Alpaca for this position.",
                sym, qty, avg_price,
            )
        else:
            if stop_price is None:
                stop_price = avg_price
            logger.info(
                "%s: Rebuilt qty=%.6f avg_price=%.4f, recovered live stop "
                "order %s at %.4f",
                sym, qty, avg_price, stop_order_id, stop_price,
            )

        self.entry_prices[sym] = avg_price
        self.entry_sizes[sym] = qty
        self.entry_directions[sym] = direction
        self.trailing_stops[sym] = stop_price
        self.stop_order_ids[sym] = stop_order_id

    def _find_open_stop_order(self, symbol: str, direction: str):
        """Look for a live (non-terminal) stop/stop_limit order on the
        protective side for *symbol*. Returns (order_id, stop_price) as
        (str, float), or (None, None) if none is found or the query fails."""
        stop_side = "sell" if direction == "long" else "buy"
        try:
            orders = self.api.list_orders(status="all", symbols=[symbol])
        except Exception as exc:
            logger.warning(
                "%s: Could not list orders while rebuilding stop tracking: %s",
                symbol, exc,
            )
            return None, None

        for order in orders:
            raw_status = getattr(order, "status", "")
            status = (raw_status.value if hasattr(raw_status, "value") else str(raw_status)).lower()
            if status in self._TERMINAL_ORDER_STATUSES:
                continue
            raw_type = getattr(order, "type", "")
            raw_side = getattr(order, "side", "")
            order_type = (raw_type.value if hasattr(raw_type, "value") else str(raw_type)).lower()
            order_side = (raw_side.value if hasattr(raw_side, "value") else str(raw_side)).lower()
            if order_type in ("stop", "stop_limit") and order_side == stop_side:
                raw_stop_price = getattr(order, "stop_price", None)
                stop_price = float(raw_stop_price) if raw_stop_price is not None else None
                return str(order.id), stop_price

        return None, None

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
    # Broker reconciliation
    #
    # Both methods below query the broker directly and never submit orders —
    # they only read live.Alpaca state and, in reconcile_and_correct(), write
    # to local bookkeeping. Used when an order's fill can't be trusted from
    # its own polled snapshot (see _wait_for_fill/_submit_protected_entry in
    # mean_reversion.py) — the broker's own position record is the one
    # source of truth that can't go stale from a client-side timeout.
    # ------------------------------------------------------------------

    @staticmethod
    def _position_to_broker_state(pos) -> dict:
        """Normalize a raw Alpaca position object into the
        {"qty", "avg_price", "direction"} shape used throughout reconciliation."""
        qty = float(pos.qty)
        return {
            "qty": abs(qty),
            "avg_price": float(pos.avg_entry_price),
            "direction": "long" if qty > 0 else "short",
        }

    def reconcile_position_with_broker(self, symbol: str) -> dict | None:
        """Query the broker directly for *symbol*'s true current position.

        Returns {"qty": float, "avg_price": float, "direction": "long"|"short"}
        or None if the broker shows no open position for *symbol* (e.g. an
        order that hasn't actually filled yet) or the query failed.
        """
        try:
            live = self._live_positions()
        except PositionFetchError as exc:
            logger.error("%s: Could not reconcile with broker: %s", symbol, exc)
            return None
        if symbol not in live:
            return None
        return self._position_to_broker_state(live[symbol])

    def reconcile_and_correct(
        self, symbol: str, tolerance: float = 1e-6, broker_state: dict | None = None,
    ) -> bool:
        """Compare broker truth for *symbol* against locally tracked
        entry price/size/direction and correct local state if they differ.

        Only entry_price/entry_size/entry_direction are corrected here —
        trailing_stop and stop_order_id are left as-is, since the broker
        doesn't expose "the bot's intended trailing stop," only whatever
        order happens to be live. Never places, cancels, or replaces any
        broker order — a live protective stop order is left exactly as-is.
        (Recovering trailing_stop/stop_order_id from scratch when local state
        is entirely missing for a symbol is handled separately by
        _rebuild_single_symbol_from_broker() at startup.)

        Safe for both long and short positions — direction comes straight
        from the broker's signed qty, not assumed.

        *broker_state*, if given, is used as-is instead of querying the
        broker again — callers that already fetched live positions (e.g.
        _reconcile_with_broker() at startup) pass it in to avoid a redundant
        API call per symbol. Returns True if a correction was made.
        """
        broker = broker_state if broker_state is not None else self.reconcile_position_with_broker(symbol)
        if broker is None:
            return False

        local_qty = self.entry_sizes.get(symbol)
        local_price = self.entry_prices.get(symbol)
        local_direction = self.entry_directions.get(symbol)

        mismatch = (
            local_qty is None
            or local_price is None
            or abs(local_qty - broker["qty"]) > tolerance
            or abs(local_price - broker["avg_price"]) > tolerance
            or local_direction != broker["direction"]
        )
        if not mismatch:
            return False

        logger.warning(
            "%s: Local state drift detected — local qty=%s price=%s direction=%s "
            "vs broker qty=%.6f price=%.4f direction=%s — correcting local state "
            "to broker truth.",
            symbol, local_qty, local_price, local_direction,
            broker["qty"], broker["avg_price"], broker["direction"],
        )
        self.entry_prices[symbol] = broker["avg_price"]
        self.entry_sizes[symbol] = broker["qty"]
        self.entry_directions[symbol] = broker["direction"]
        self._save_state()
        return True

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
    # External-exit recovery
    # ------------------------------------------------------------------

    def recover_external_exit(self, symbol: str, fallback_price: float) -> bool:
        """Log a position that closed on Alpaca without the bot's own _close()
        path running, and clear its local state.

        The broker-side hard stop sits live on Alpaca and fills the instant
        price crosses it — that almost always beats the bot's own poll-interval
        price check. When that happens, position_state(symbol) already reads
        "flat" by the time the bot notices, so the strategy's stop-loss/SMA-exit
        branches never fire and log_trade() never runs, silently dropping the
        fill from trades.csv. Callers must only invoke this after confirming
        (via a live position_state() query) that Alpaca now shows *symbol* flat.

        Returns True if a trade was recovered and logged, False if there was
        no local state to reconcile (nothing to do).
        """
        if symbol not in self.entry_prices:
            return False

        direction = self.entry_directions.get(symbol)
        entry_price = self.entry_prices.get(symbol, fallback_price)
        size = self.entry_sizes.get(symbol, 0)
        exit_price = fallback_price

        stop_id = self.stop_order_ids.get(symbol)
        if stop_id:
            try:
                order = self.api.get_order(stop_id)
                status = getattr(order, "status", "")
                status = status.value if hasattr(status, "value") else str(status)
                if status.lower() == "filled" and order.filled_avg_price is not None:
                    exit_price = float(order.filled_avg_price)
            except Exception as exc:
                logger.warning(
                    "%s: Could not fetch stop order %s to recover exact exit "
                    "price, using fallback %.4f: %s", symbol, stop_id, fallback_price, exc,
                )

        pnl = self.log_trade(symbol, direction, entry_price, exit_price, size)
        logger.warning(
            "%s: Recovered externally-closed position as stop-out — "
            "exit=%.4f pnl=$%.2f", symbol, exit_price, pnl,
        )
        self.record_stop_out(symbol)
        self.clear_position(symbol)
        return True

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
