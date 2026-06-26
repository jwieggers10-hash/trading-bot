"""Momentum Breakout strategy for BTC/USD on 1-hour candles.

Entry: price breaks above the 20-period high (long) or below the 20-period low (short)
       with volume at least 1.5x the 20-period average volume.
Exit:  trailing stop at 2x ATR, or break in the opposite direction.
Stop:  initial stop at 2x ATR from entry; ratcheted as price moves in our favour.
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import (
    MOMENTUM_PERIOD,
    MOMENTUM_TIMEFRAME,
    MOMENTUM_TRAILING_ATR,
    VOLUME_MULTIPLIER,
)
from bot.strategies.mean_reversion import _fill_price, _filled_qty, _flatten

logger = logging.getLogger(__name__)

SYMBOL = "BTC/USD"


class MomentumBreakoutStrategy:
    def __init__(self, api, risk_manager, portfolio):
        self.api = api
        self.rm = risk_manager
        self.portfolio = portfolio

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_bars(self) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        # 5 days covers 120+ 1-hour bars with room for gaps
        start = end - timedelta(days=5)
        bars = self.api.get_crypto_bars(
            SYMBOL,
            MOMENTUM_TIMEFRAME,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=200,
        ).df
        return _flatten(bars, SYMBOL)

    # ------------------------------------------------------------------
    # Main entry point (always called; crypto is 24/7)
    # ------------------------------------------------------------------

    def run(self, symbol: str = SYMBOL):
        try:
            bars = self._get_bars()
        except AttributeError:
            # get_crypto_bars not available in this alpaca-trade-api version — fall back
            try:
                bars = self._get_bars_fallback()
            except Exception as exc:
                logger.error("%s: Failed to fetch bars: %s", SYMBOL, exc)
                return
        except Exception as exc:
            logger.error("%s: Failed to fetch bars: %s", SYMBOL, exc)
            return

        if len(bars) < MOMENTUM_PERIOD + 2:
            logger.warning("%s: Only %d bars, need %d", SYMBOL, len(bars), MOMENTUM_PERIOD + 2)
            return

        try:
            atr = self.rm.calculate_atr(bars)
        except Exception as exc:
            logger.error("%s: ATR calculation failed: %s", SYMBOL, exc)
            return

        # Lookback uses bars[-period-1 : -1] so the current bar is excluded
        lookback = bars.iloc[-(MOMENTUM_PERIOD + 1):-1]
        period_high = float(lookback["high"].max())
        period_low = float(lookback["low"].min())
        avg_volume = float(lookback["volume"].mean())

        current_price = float(bars["close"].iloc[-1])
        current_volume = float(bars["volume"].iloc[-1])
        volume_ok = current_volume >= VOLUME_MULTIPLIER * avg_volume

        is_long = self.portfolio.is_long(SYMBOL)
        is_short = self.portfolio.is_short(SYMBOL)

        # --- Update and check trailing stop ---
        if is_long:
            self.portfolio.update_trailing_stop(SYMBOL, current_price, atr, MOMENTUM_TRAILING_ATR, "long")
            if self.portfolio.trailing_stop_triggered(SYMBOL, current_price):
                logger.info("%s: Trailing stop triggered at %.4f", SYMBOL, current_price)
                self._close("long", current_price)
                return

        if is_short:
            self.portfolio.update_trailing_stop(SYMBOL, current_price, atr, MOMENTUM_TRAILING_ATR, "short")
            if self.portfolio.trailing_stop_triggered(SYMBOL, current_price):
                logger.info("%s: Trailing stop triggered at %.4f", SYMBOL, current_price)
                self._close("short", current_price)
                return

        # --- Exit long if price breaks 20-period low (direction reversal) ---
        if is_long and current_price < period_low:
            logger.info("%s: Exit long — price %.4f broke below 20-period low %.4f", SYMBOL, current_price, period_low)
            self._close("long", current_price)
            return

        # --- New entries (only when flat) ---
        if not is_long and not is_short:
            equity = self.rm.get_account_equity()
            size = self.rm.calculate_position_size(atr, equity, current_price)

            if current_price > period_high and volume_ok:
                if self.portfolio.blocks_new_long(SYMBOL):
                    return
                logger.info(
                    "%s: Long breakout — %.4f > 20H %.4f, vol %.0f vs avg %.0f",
                    SYMBOL, current_price, period_high, current_volume, avg_volume,
                )
                stop = round(current_price - MOMENTUM_TRAILING_ATR * atr, 4)
                self._enter("buy", "long", size, stop)

            elif current_price < period_low and volume_ok:
                logger.info(
                    "%s: Short breakout — %.4f < 20L %.4f, vol %.0f vs avg %.0f",
                    SYMBOL, current_price, period_low, current_volume, avg_volume,
                )
                stop = round(current_price + MOMENTUM_TRAILING_ATR * atr, 4)
                self._enter("sell", "short", size, stop)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _enter(self, side: str, direction: str, size: float, stop: float):
        try:
            order = self.api.submit_order(
                symbol=SYMBOL,
                qty=str(size),
                side=side,
                type="market",
                time_in_force="gtc",   # crypto uses GTC
            )
        except Exception as exc:
            logger.error("%s: Market order failed (%s): %s", SYMBOL, side, exc)
            return

        filled = _filled_qty(order, size)
        entry_price = _fill_price(order, self.api, SYMBOL)

        stop_side = "sell" if direction == "long" else "buy"
        try:
            stop_order = self.api.submit_order(
                symbol=SYMBOL,
                qty=str(round(filled, 6)),
                side=stop_side,
                type="stop",
                time_in_force="gtc",
                stop_price=str(stop),
            )
        except Exception as exc:
            logger.critical(
                "%s: STOP ORDER FAILED after market fill — closing immediately. "
                "market_order=%s filled=%.6f error=%s",
                SYMBOL, order.id, filled, exc,
            )
            try:
                self.api.close_position(SYMBOL)
            except Exception as close_exc:
                logger.critical("%s: Emergency close also failed: %s", SYMBOL, close_exc)
            return

        self.portfolio.record_entry(SYMBOL, direction, entry_price, filled, stop_order.id, stop)
        logger.info("%s: Entered %s %.6f at %.4f, stop %.4f", SYMBOL, direction, filled, entry_price, stop)

    def _close(self, direction: str, current_price: float):
        try:
            stop_id = self.portfolio.stop_order_ids.get(SYMBOL)
            if stop_id:
                try:
                    self.api.cancel_order(stop_id)
                except Exception:
                    pass

            self.api.close_position(SYMBOL)
            entry_price = self.portfolio.entry_prices.get(SYMBOL, current_price)
            size = self.portfolio.entry_sizes.get(SYMBOL, 0)
            self.portfolio.log_trade(SYMBOL, direction, entry_price, current_price, size)
            self.portfolio.clear_position(SYMBOL)
        except Exception as exc:
            logger.error("%s: Failed to close: %s", SYMBOL, exc)

    def _get_bars_fallback(self) -> pd.DataFrame:
        """Fall back to get_bars with BTCUSD symbol for older alpaca-trade-api versions."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=5)
        bars = self.api.get_bars(
            "BTCUSD",
            MOMENTUM_TIMEFRAME,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=200,
        ).df
        return _flatten(bars, "BTCUSD")
