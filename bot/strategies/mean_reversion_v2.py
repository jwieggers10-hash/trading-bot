"""Strategy 2.0 — Pooled-capital mean reversion for SPY, QQQ, GLD, USO.

Key differences from the original MeanReversionStrategy (v1):
  - Per-symbol std-dev thresholds tuned via grid-search backtest:
      SPY=0.25  QQQ=0.30  GLD=0.25  USO=0.25
  - Dynamic position cap: each symbol may use up to (total_equity / n_symbols),
    which compounds as the portfolio grows.
  - Risk per trade: 0.25% of total portfolio equity (not 1% of per-symbol equity).
  - Shared capital awareness: locked notional across all 4 symbols is subtracted
    from available capital before sizing a new entry.
  - Account equity is cached for 60 s to avoid hammering the API when all 4
    symbols fire within the same 15-min tick.

Entry / exit / stop logic is identical to v1:
  - Long  when close < SMA − threshold×std
  - Short when close > SMA + threshold×std
  - Exit  when close crosses back to SMA
  - Hard broker-side stop at 1 ATR from entry price
"""
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.portfolio import PositionFetchError
from bot.risk_manager import round_stop_price
from bot.telegram_notifier import notifier
from config import (
    MEAN_REVERSION_PERIOD,
    MEAN_REVERSION_TIMEFRAME,
    MR_V2_THRESHOLDS,
    MR_V2_RISK_PCT,
    MR_V2_SYMBOLS,
    STOP_COOLDOWN_SECONDS,
)

logger = logging.getLogger(__name__)

_EQUITY_CACHE_TTL = 60.0   # seconds between account equity refreshes


class MeanReversionV2:
    def __init__(self, api, risk_manager, portfolio):
        self.api       = api
        self.rm        = risk_manager
        self.portfolio = portfolio
        self.n_symbols = len(MR_V2_SYMBOLS)
        self.risk_pct  = MR_V2_RISK_PCT

        self._equity_cache: float = 0.0
        self._equity_ts:    float = 0.0   # monotonic time of last refresh

    # ------------------------------------------------------------------
    # Equity cache (shared across all run() calls within a tick)
    # ------------------------------------------------------------------

    def _total_equity(self) -> float:
        """Return account equity, refreshing at most once per TTL window."""
        if time.monotonic() - self._equity_ts > _EQUITY_CACHE_TTL:
            self._equity_cache = self.rm.get_account_equity()
            self._equity_ts    = time.monotonic()
        return self._equity_cache

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_bars(self, symbol: str) -> pd.DataFrame:
        end   = datetime.now(timezone.utc)
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
        from bot.strategies.mean_reversion import _flatten
        return _flatten(bars, symbol)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _signals(self, bars: pd.DataFrame, symbol: str) -> dict:
        closes    = bars["close"]
        sma       = closes.rolling(MEAN_REVERSION_PERIOD).mean()
        std       = closes.rolling(MEAN_REVERSION_PERIOD).std()
        threshold = MR_V2_THRESHOLDS.get(symbol, 0.25)
        return {
            "price": float(closes.iloc[-1]),
            "sma":   float(sma.iloc[-1]),
            "upper": float(sma.iloc[-1] + threshold * std.iloc[-1]),
            "lower": float(sma.iloc[-1] - threshold * std.iloc[-1]),
        }

    # ------------------------------------------------------------------
    # Main entry point (called once per symbol per 15-min tick)
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
        sma   = sig["sma"]

        try:
            pos_state = self.portfolio.position_state(symbol)
        except PositionFetchError:
            logger.error(
                "%s: Cannot determine position state — skipping tick", symbol)
            return

        is_long  = pos_state == "long"
        is_short = pos_state == "short"

        # --- Hard stop check ---
        if (is_long or is_short) and self.portfolio.trailing_stop_triggered(symbol, price):
            direction = "long" if is_long else "short"
            logger.info("%s: Hard stop triggered at %.4f", symbol, price)
            self._close(symbol, direction, price, reason="stop_loss")
            self.portfolio.record_stop_out(symbol)
            return

        # --- SMA exit ---
        if is_long and price >= sma:
            logger.info("%s: Exit long — price %.4f returned to SMA %.4f",
                        symbol, price, sma)
            self._close(symbol, "long", price, reason="sma_exit")
            return

        if is_short and price <= sma:
            logger.info("%s: Exit short — price %.4f returned to SMA %.4f",
                        symbol, price, sma)
            self._close(symbol, "short", price, reason="sma_exit")
            return

        # --- New entry (only when flat) ---
        if not is_long and not is_short:
            if self.portfolio.in_stop_cooldown(symbol):
                logger.info("%s: In %d-min stop-out cooldown — skipping entry",
                            symbol, STOP_COOLDOWN_SECONDS // 60)
                return

            # Dynamic cap: total equity / n_symbols, minus locked notional in others
            total_equity  = self._total_equity()
            per_sym_cap   = total_equity / self.n_symbols
            locked_others = self.portfolio.get_locked_notional(exclude_symbol=symbol)
            available     = max(0.0, total_equity - locked_others)
            max_notional  = min(per_sym_cap, available)

            size = self.rm.v2_position_size(
                atr, total_equity, price, max_notional, self.risk_pct
            )

            if size <= 0:
                logger.warning(
                    "%s: Sizing returned 0 — skipping entry "
                    "(ATR=%.4f equity=%.0f per_sym_cap=%.0f available=%.0f)",
                    symbol, atr, total_equity, per_sym_cap, available,
                )
                return

            if price < sig["lower"]:
                logger.info(
                    "%s: Long signal — %.4f < lower band %.4f "
                    "(SMA %.4f, ATR %.4f, size %d, cap $%.0f)",
                    symbol, price, sig["lower"], sma, atr, size, max_notional,
                )
                stop = self.rm.stop_price("long", price, atr)
                self._enter(symbol, "buy", "long", size, stop, est_price=price)

            elif price > sig["upper"]:
                logger.info(
                    "%s: Short signal — %.4f > upper band %.4f "
                    "(SMA %.4f, ATR %.4f, size %d, cap $%.0f)",
                    symbol, price, sig["upper"], sma, atr, size, max_notional,
                )
                stop = self.rm.stop_price("short", price, atr)
                self._enter(symbol, "sell", "short", size, stop, est_price=price)

    # ------------------------------------------------------------------
    # Order helpers (identical to v1)
    # ------------------------------------------------------------------

    def _enter(self, symbol: str, side: str, direction: str,
               size: int, stop: float, est_price: float = 0.0):
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

        from bot.strategies.mean_reversion import _filled_qty, _fill_price
        filled      = _filled_qty(order, size)
        entry_price = _fill_price(order, self.api, symbol)

        stop_side    = "sell" if direction == "long" else "buy"
        rounded_stop = round_stop_price(stop, stop_side, symbol)
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
                "%s: STOP ORDER FAILED after market fill — closing immediately. "
                "order=%s filled=%d error=%s",
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

        self.portfolio.record_entry(
            symbol, direction, entry_price, filled, stop_order.id, rounded_stop
        )
        notifier.trade_entry_filled(symbol, direction, filled, entry_price, stop)
        notifier.stop_order_submitted(symbol, stop_side, filled, stop)
        logger.info(
            "%s [v2]: Entered %s x%d at %.4f, stop %.4f  "
            "(risk $%.0f on $%.0f equity)",
            symbol, direction, filled, entry_price, stop,
            self._total_equity() * self.risk_pct,
            self._total_equity(),
        )

    def _close(self, symbol: str, direction: str,
               current_price: float, reason: str = "sma_exit"):
        try:
            stop_id = self.portfolio.stop_order_ids.get(symbol)
            if stop_id:
                try:
                    self.api.cancel_order(stop_id)
                except Exception:
                    pass

            self.api.close_position(symbol)
            entry_price = self.portfolio.entry_prices.get(symbol, current_price)
            size        = self.portfolio.entry_sizes.get(symbol, 0)
            pnl         = self.portfolio.log_trade(
                symbol, direction, entry_price, current_price, size
            )
            self.portfolio.clear_position(symbol)

            if reason == "stop_loss":
                notifier.stop_loss_triggered(symbol, direction, current_price, size, pnl)
            else:
                notifier.sma_exit(symbol, direction, current_price, size, pnl)
        except Exception as exc:
            logger.error("%s: Failed to close position: %s", symbol, exc)
