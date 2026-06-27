import logging
import math

import pandas as pd
import numpy as np
from config import ATR_PERIOD, CAPITAL_PER_SYMBOL, RISK_PER_TRADE


def round_stop_price(price: float, stop_side: str, symbol: str) -> float:
    """Round a stop price to the precision Alpaca requires for the instrument.

    Equities accept at most 2 decimal places.  Round conservatively so the stop
    remains on the protective side of the intended price:
    - sell stop (protecting a long): floor to 2 dp — stop sits at or below intent
    - buy stop  (protecting a short): ceil to 2 dp — stop sits at or above intent

    Crypto symbols (containing '/') are rounded to 4 decimal places.
    """
    if "/" in symbol:
        return round(price, 4)
    if stop_side == "sell":
        return math.floor(price * 100) / 100
    return math.ceil(price * 100) / 100

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, api):
        self.api = api

    def get_account_equity(self) -> float:
        account = self.api.get_account()
        return float(account.equity)

    def calculate_atr(self, bars: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """14-period ATR using Wilder's true range definition."""
        high = bars["high"]
        low = bars["low"]
        close = bars["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=period).mean().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            raise ValueError(f"ATR is invalid ({atr}); not enough bars or zero volatility")
        return float(atr)

    def calculate_position_size(self, atr: float, equity: float,
                                price: float, cap: float = CAPITAL_PER_SYMBOL) -> float:
        """Fractional size for crypto: 1 ATR loss = 1% equity, capped so notional <= cap."""
        if atr <= 0 or price <= 0:
            return 0.0
        atr_size = (equity * RISK_PER_TRADE) / atr
        return round(min(atr_size, cap / price), 6)

    def integer_position_size(self, atr: float, equity: float,
                              price: float, cap: float = CAPITAL_PER_SYMBOL) -> int:
        """Whole-share size for equities: 1 ATR loss = 1% equity, capped so notional <= cap."""
        if atr <= 0 or price <= 0:
            return 0
        atr_size = int((equity * RISK_PER_TRADE) / atr)
        if atr_size <= 0:
            return 0
        return min(atr_size, int(cap / price))

    def stop_price(self, direction: str, entry_price: float, atr: float) -> float:
        """Hard stop at exactly 1 ATR from entry (1% equity by construction)."""
        if direction == "long":
            return round(entry_price - atr, 4)
        return round(entry_price + atr, 4)
