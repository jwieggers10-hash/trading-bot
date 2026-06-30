"""Compatibility shim: exposes the alpaca-trade-api REST interface backed by alpaca-py.

All existing bot code passes a single `api` object around. This module provides
a `REST` class with the same method signatures so no other files need to change.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

_TIMEFRAME_MAP = {
    "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
    "4Hour": TimeFrame(4,  TimeFrameUnit.Hour),
    "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
}


class REST:
    def __init__(self, key, secret, base_url, api_version="v2"):
        paper = "paper" in base_url
        self._trading = TradingClient(key, secret, paper=paper)
        self._stock   = StockHistoricalDataClient(key, secret)
        self._crypto  = CryptoHistoricalDataClient(key, secret)

    def get_clock(self):
        return self._trading.get_clock()

    def get_account(self):
        return self._trading.get_account()

    def list_positions(self):
        return self._trading.get_all_positions()

    def submit_order(self, symbol, qty, side, type, time_in_force, stop_price=None):
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        tif = TimeInForce(time_in_force.lower())
        qty = float(qty)
        if type == "market":
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=tif,
            )
        elif type == "stop":
            req = StopOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=tif,
                stop_price=float(stop_price),
            )
        else:
            raise ValueError(f"Unsupported order type: {type!r}")
        return self._trading.submit_order(req)

    def cancel_order(self, order_id):
        self._trading.cancel_order_by_id(order_id)

    def list_orders(self, status="open", symbols=None):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        status_map = {
            "open":   QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all":    QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(
            status=status_map.get(status, QueryOrderStatus.OPEN),
            symbols=symbols,
        )
        return self._trading.get_orders(req)

    def get_order(self, order_id):
        return self._trading.get_order_by_id(str(order_id))

    def close_position(self, symbol):
        return self._trading.close_position(symbol_or_asset_id=symbol)

    def get_bars(self, symbol, timeframe, start=None, end=None, limit=None,
                 feed=None, adjustment=None):
        tf = _TIMEFRAME_MAP.get(timeframe, timeframe)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
            feed=feed,
            adjustment=adjustment,
        )
        return self._stock.get_stock_bars(req)

    def get_crypto_bars(self, symbol, timeframe, start=None, end=None, limit=None):
        tf = _TIMEFRAME_MAP.get(timeframe, timeframe)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )
        return self._crypto.get_crypto_bars(req)

    def get_latest_trade(self, symbol):
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        result = self._stock.get_stock_latest_trade(req)
        return result[symbol]
