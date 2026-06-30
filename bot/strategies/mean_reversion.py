"""Mean Reversion strategy for SPY and QQQ on 15-minute candles.

Entry: price crosses more than N standard deviations from the 20-period SMA
  SPY threshold = 1.5 std devs
  QQQ threshold = 1.8 std devs
Exit: price returns to the SMA
Stop: hard stop at 1 ATR below (long) or above (short) entry price
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.portfolio import PositionFetchError
from bot.risk_manager import round_stop_price
from bot.telegram_notifier import notifier
from config import (
    MEAN_REVERSION_PERIOD,
    MEAN_REVERSION_TIMEFRAME,
    STD_DEV_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    def __init__(self, api, risk_manager, portfolio):
        self.api = api
        self.rm = risk_manager
        self.portfolio = portfolio

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_bars(self, symbol: str) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        # 8 hours covers a full regular session plus pre-market buffer
        start = end - timedelta(hours=8)
        bars = self.api.get_bars(
            symbol,
            MEAN_REVERSION_TIMEFRAME,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=60,
            feed="iex",
            adjustment="raw",
        ).df
        return _flatten(bars, symbol)

    # ------------------------------------------------------------------
    # Signal calculation
    # ------------------------------------------------------------------

    def _signals(self, bars: pd.DataFrame, symbol: str) -> dict:
        closes = bars["close"]
        sma = closes.rolling(MEAN_REVERSION_PERIOD).mean()
        std = closes.rolling(MEAN_REVERSION_PERIOD).std()
        threshold = STD_DEV_THRESHOLDS.get(symbol, 1.5)
        return {
            "price": float(closes.iloc[-1]),
            "sma": float(sma.iloc[-1]),
            "upper": float(sma.iloc[-1] + threshold * std.iloc[-1]),
            "lower": float(sma.iloc[-1] - threshold * std.iloc[-1]),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, symbol: str):
        try:
            bars = self._get_bars(symbol)
        except Exception as exc:
            logger.error("%s: Failed to fetch bars: %s", symbol, exc)
            return

        if len(bars) < MEAN_REVERSION_PERIOD + 2:
            logger.warning("%s: Only %d bars available, need %d",
                           symbol, len(bars), MEAN_REVERSION_PERIOD + 2)
            return

        try:
            sig = self._signals(bars, symbol)
            atr = self.rm.calculate_atr(bars)
        except Exception as exc:
            logger.error("%s: Indicator calculation failed: %s", symbol, exc)
            return

        price = sig["price"]
        sma = sig["sma"]

        # Fetch position state once; bail out if Alpaca is unreachable rather
        # than assuming flat (which would cause a phantom duplicate entry).
        try:
            pos_state = self.portfolio.position_state(symbol)
        except PositionFetchError:
            logger.error(
                "%s: Cannot determine position state — skipping tick to prevent phantom entry",
                symbol,
            )
            return

        is_long = pos_state == "long"
        is_short = pos_state == "short"

        # --- Check trailing/hard stop first ---
        if (is_long or is_short) and self.portfolio.trailing_stop_triggered(symbol, price):
            direction = "long" if is_long else "short"
            logger.info("%s: Hard stop triggered at %.4f", symbol, price)
            self._close(symbol, direction, price, reason="stop_loss")
            self.portfolio.record_stop_out(symbol)
            return

        # --- Exit on mean reversion ---
        if is_long and price >= sma:
            logger.info("%s: Exit long — price %.4f returned to SMA %.4f", symbol, price, sma)
            self._close(symbol, "long", price, reason="sma_exit")
            return

        if is_short and price <= sma:
            logger.info("%s: Exit short — price %.4f returned to SMA %.4f", symbol, price, sma)
            self._close(symbol, "short", price, reason="sma_exit")
            return

        # --- New entries (only when flat) ---
        if not is_long and not is_short:
            if self.portfolio.in_stop_cooldown(symbol):
                logger.info("%s: In %d-min stop-out cooldown — skipping entry", symbol,
                            __import__("config").STOP_COOLDOWN_SECONDS // 60)
                return

            equity = self.rm.get_account_equity()
            size = self.rm.integer_position_size(atr, equity, price)

            if size <= 0:
                logger.warning("%s: Sizing returned 0 — skipping entry (ATR=%.4f equity=%.2f)",
                               symbol, atr, equity)
                return

            if price < sig["lower"]:
                logger.info(
                    "%s: Long signal — %.4f < lower band %.4f (SMA %.4f, ATR %.4f, size %d)",
                    symbol, price, sig["lower"], sma, atr, size,
                )
                stop = self.rm.stop_price("long", price, atr)
                self._enter(symbol, "buy", "long", size, stop, est_price=price)

            elif price > sig["upper"]:
                logger.info(
                    "%s: Short signal — %.4f > upper band %.4f (SMA %.4f, ATR %.4f, size %d)",
                    symbol, price, sig["upper"], sma, atr, size,
                )
                stop = self.rm.stop_price("short", price, atr)
                self._enter(symbol, "sell", "short", size, stop, est_price=price)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _enter(
        self, symbol: str, side: str, direction: str, size: int, stop: float,
        est_price: float = 0.0,
    ):
        # Step 1: submit market order
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=size,
                side=side,
                type="market",
                time_in_force="day",
            )
        except Exception as exc:
            logger.error("%s: Market order failed (%s %d): %s", symbol, side, size, exc)
            return

        notifier.trade_entry_submitted(symbol, side, size, est_price)

        filled = _filled_qty(order, size)
        entry_price = _fill_price(order, self.api, symbol)
        if entry_price <= 0:
            # Alpaca paper-trading returns orders before they settle; filled_avg_price is
            # sometimes None and get_latest_trade may also fail.  Fall back to the signal
            # price so that record_entry() stores a non-zero cost basis, preventing
            # get_locked_notional() from undercounting deployed capital for later symbols.
            entry_price = est_price
            logger.warning(
                "%s: fill price unavailable from broker — using signal price %.4f "
                "for locked-capital accounting",
                symbol, est_price,
            )

        # Step 2: place broker-side stop for the actual filled quantity.
        # If this fails, the position is open with no protection — close it immediately.
        stop_side = "sell" if direction == "long" else "buy"
        rounded_stop = round_stop_price(stop, stop_side, symbol)

        # Cancel any lingering stop orders before placing the new one.
        # Prevents Alpaca error 40310000 (wash-trade) from a prior position's
        # stop that was not cleaned up due to a silent cancel failure or bot crash.
        self.portfolio.cancel_open_stop_orders(symbol, stop_side)

        try:
            stop_order = self.api.submit_order(
                symbol=symbol,
                qty=filled,
                side=stop_side,
                type="stop",
                time_in_force="gtc",
                stop_price=str(rounded_stop),
            )
        except Exception as exc:
            logger.critical(
                "%s: STOP ORDER FAILED after market fill — closing position immediately. "
                "market_order=%s filled=%d error=%s",
                symbol, order.id, filled, exc,
            )
            notifier.critical_error(
                f"{symbol}: Stop order failed after market fill",
                f"order={order.id} filled={filled} error={exc}",
            )
            try:
                self.api.close_position(symbol)
            except Exception as close_exc:
                logger.critical("%s: Emergency close also failed: %s", symbol, close_exc)
            return

        self.portfolio.record_entry(symbol, direction, entry_price, filled, stop_order.id, rounded_stop)
        notifier.trade_entry_filled(symbol, direction, filled, entry_price, stop)
        notifier.stop_order_submitted(symbol, stop_side, filled, stop)
        logger.info(
            "%s: Entered %s x%d at %.4f (requested %d), stop at %.4f",
            symbol, direction, filled, entry_price, size, stop,
        )

    def _close(self, symbol: str, direction: str, current_price: float, reason: str = "sma_exit"):
        try:
            stop_side = "sell" if direction == "long" else "buy"
            # Cancel tracked stop order before closing to avoid double-fill.
            stop_id = self.portfolio.stop_order_ids.get(symbol)
            if stop_id:
                try:
                    self.api.cancel_order(stop_id)
                except Exception as exc:
                    logger.warning(
                        "%s: Could not cancel stop order %s before close: %s",
                        symbol, stop_id, exc,
                    )
            # Sweep for any untracked stops so they cannot fill against the flat position.
            self.portfolio.cancel_open_stop_orders(symbol, stop_side)

            self.api.close_position(symbol)
            entry_price = self.portfolio.entry_prices.get(symbol, current_price)
            size = self.portfolio.entry_sizes.get(symbol, 0)
            pnl = self.portfolio.log_trade(symbol, direction, entry_price, current_price, size)
            self.portfolio.clear_position(symbol)

            if reason == "stop_loss":
                notifier.stop_loss_triggered(symbol, direction, current_price, size, pnl)
            else:
                notifier.sma_exit(symbol, direction, current_price, size, pnl)
        except Exception as exc:
            logger.error("%s: Failed to close position: %s", symbol, exc)


# ------------------------------------------------------------------
# Shared helpers (imported by trend_following and momentum_breakout)
# ------------------------------------------------------------------

def _flatten(bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Handle both flat and MultiIndex DataFrames from alpaca-trade-api."""
    if isinstance(bars_df.index, pd.MultiIndex):
        return bars_df.xs(symbol, level=0)
    return bars_df


def _fill_price(order, api, symbol: str) -> float:
    """Return filled average price; fall back to latest trade price."""
    if order.filled_avg_price and float(order.filled_avg_price) > 0:
        return float(order.filled_avg_price)
    try:
        trade = api.get_latest_trade(symbol)
        return float(trade.price)
    except Exception:
        return float(order.limit_price or 0)


def _filled_qty(order, requested):
    """Return actual filled quantity from the order; fall back to requested size.

    Uses filled_qty when positive so that a partial fill on a market order
    causes the stop to be placed for the number of shares actually owned,
    not the number originally requested.
    """
    try:
        fq = getattr(order, "filled_qty", None)
        if fq is not None:
            fq_val = float(str(fq))
            if fq_val > 0:
                return int(fq_val) if isinstance(requested, int) else round(fq_val, 6)
    except (TypeError, ValueError):
        pass
    return requested
