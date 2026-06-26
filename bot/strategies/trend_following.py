"""Trend Following strategy for GLD and USO on 4-hour candles.

Bars are fetched as 1-hour candles and resampled to 4-hour in pandas to avoid
relying on broker-side 4H candle support (which varies by API version).

Entry: 50 EMA crosses above 200 EMA (golden cross) → long
       50 EMA crosses below 200 EMA (death cross)  → short
Exit:  opposite crossover or trailing stop at 3x ATR.
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import TREND_FAST_EMA, TREND_SLOW_EMA, TREND_TRAILING_ATR
from bot.strategies.mean_reversion import _fill_price, _filled_qty, _flatten

logger = logging.getLogger(__name__)

_4H_RESAMPLE_COLS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


class TrendFollowingStrategy:
    def __init__(self, api, risk_manager, portfolio):
        self.api = api
        self.rm = risk_manager
        self.portfolio = portfolio

    # ------------------------------------------------------------------
    # Data — fetch 1H bars and resample to 4H
    # ------------------------------------------------------------------

    def _get_bars(self, symbol: str) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        # 200 * 4H bars ≈ 800 hours. Equity markets trade ~6.5H/day so we need
        # ~800 / 6.5 ≈ 124 trading days ≈ 175 calendar days. Use 400 days of 1H
        # bars to be safe and then resample.
        start = end - timedelta(days=400)
        bars_1h = self.api.get_bars(
            symbol,
            "1Hour",
            start=start.isoformat(),
            end=end.isoformat(),
            limit=5000,
            feed="iex",
            adjustment="raw",
        ).df
        bars_1h = _flatten(bars_1h, symbol)

        # Resample to 4-hour aligned to session open (14:30 UTC = 9:30 ET)
        bars_4h = (
            bars_1h.resample("4h", offset="30min")
            .agg({k: v for k, v in _4H_RESAMPLE_COLS.items() if k in bars_1h.columns})
            .dropna(subset=["close"])
        )
        return bars_4h

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, symbol: str):
        try:
            bars = self._get_bars(symbol)
        except Exception as exc:
            logger.error("%s: Failed to fetch bars: %s", symbol, exc)
            return

        if len(bars) < TREND_SLOW_EMA + 2:
            logger.warning(
                "%s: Only %d 4H bars available; need %d for 200 EMA warm-up",
                symbol, len(bars), TREND_SLOW_EMA + 2,
            )
            return

        try:
            atr = self.rm.calculate_atr(bars)
        except Exception as exc:
            logger.error("%s: ATR calculation failed: %s", symbol, exc)
            return

        closes = bars["close"]
        fast = closes.ewm(span=TREND_FAST_EMA, adjust=False).mean()
        slow = closes.ewm(span=TREND_SLOW_EMA, adjust=False).mean()

        fast_prev, fast_curr = float(fast.iloc[-2]), float(fast.iloc[-1])
        slow_prev, slow_curr = float(slow.iloc[-2]), float(slow.iloc[-1])
        current_price = float(closes.iloc[-1])

        golden_cross = fast_prev < slow_prev and fast_curr >= slow_curr
        death_cross = fast_prev > slow_prev and fast_curr <= slow_curr

        is_long = self.portfolio.is_long(symbol)
        is_short = self.portfolio.is_short(symbol)

        # --- Update and check trailing stop ---
        if is_long:
            self.portfolio.update_trailing_stop(symbol, current_price, atr, TREND_TRAILING_ATR, "long")
            if self.portfolio.trailing_stop_triggered(symbol, current_price):
                logger.info("%s: Trailing stop triggered at %.4f", symbol, current_price)
                self._close(symbol, "long", current_price)
                is_long = False

        if is_short:
            self.portfolio.update_trailing_stop(symbol, current_price, atr, TREND_TRAILING_ATR, "short")
            if self.portfolio.trailing_stop_triggered(symbol, current_price):
                logger.info("%s: Trailing stop triggered at %.4f", symbol, current_price)
                self._close(symbol, "short", current_price)
                is_short = False

        # --- Death cross: exit long → enter short ---
        if death_cross:
            if is_long:
                logger.info("%s: Death cross — exiting long at %.4f", symbol, current_price)
                self._close(symbol, "long", current_price)
                is_long = False
            if not is_short:
                equity = self.rm.get_account_equity()
                size = self.rm.integer_position_size(atr, equity, current_price)
                if size > 0:
                    logger.info(
                        "%s: Death cross — entering short at %.4f (50EMA=%.4f, 200EMA=%.4f)",
                        symbol, current_price, fast_curr, slow_curr,
                    )
                    stop = self.rm.stop_price("short", current_price, atr * TREND_TRAILING_ATR)
                    self._enter(symbol, "sell", "short", size, stop, atr)

        # --- Golden cross: exit short → enter long ---
        elif golden_cross:
            if is_short:
                logger.info("%s: Golden cross — exiting short at %.4f", symbol, current_price)
                self._close(symbol, "short", current_price)
                is_short = False
            if not is_long:
                equity = self.rm.get_account_equity()
                size = self.rm.integer_position_size(atr, equity, current_price)
                if size > 0:
                    logger.info(
                        "%s: Golden cross — entering long at %.4f (50EMA=%.4f, 200EMA=%.4f)",
                        symbol, current_price, fast_curr, slow_curr,
                    )
                    stop = self.rm.stop_price("long", current_price, atr * TREND_TRAILING_ATR)
                    self._enter(symbol, "buy", "long", size, stop, atr)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _enter(self, symbol: str, side: str, direction: str, size: int, stop: float, atr: float):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=size,
                side=side,
                type="market",
                time_in_force="day",
            )
        except Exception as exc:
            logger.error("%s: Market order failed (%s): %s", symbol, side, exc)
            return

        filled = _filled_qty(order, size)
        entry_price = _fill_price(order, self.api, symbol)

        stop_side = "sell" if direction == "long" else "buy"
        try:
            stop_order = self.api.submit_order(
                symbol=symbol,
                qty=filled,
                side=stop_side,
                type="stop",
                time_in_force="gtc",
                stop_price=str(round(stop, 4)),
            )
        except Exception as exc:
            logger.critical(
                "%s: STOP ORDER FAILED after market fill — closing immediately. "
                "market_order=%s filled=%d error=%s",
                symbol, order.id, filled, exc,
            )
            try:
                self.api.close_position(symbol)
            except Exception as close_exc:
                logger.critical("%s: Emergency close also failed: %s", symbol, close_exc)
            return

        self.portfolio.record_entry(symbol, direction, entry_price, filled, stop_order.id, stop)
        logger.info("%s: Entered %s x%d at %.4f, stop %.4f", symbol, direction, filled, entry_price, stop)

    def _close(self, symbol: str, direction: str, current_price: float):
        try:
            stop_id = self.portfolio.stop_order_ids.get(symbol)
            if stop_id:
                try:
                    self.api.cancel_order(stop_id)
                except Exception:
                    pass

            self.api.close_position(symbol)
            entry_price = self.portfolio.entry_prices.get(symbol, current_price)
            size = self.portfolio.entry_sizes.get(symbol, 0)
            self.portfolio.log_trade(symbol, direction, entry_price, current_price, size)
            self.portfolio.clear_position(symbol)
        except Exception as exc:
            logger.error("%s: Failed to close: %s", symbol, exc)
