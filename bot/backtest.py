# -*- coding: utf-8 -*-
"""
Backtests all three strategies against Alpaca historical data.

Outputs per-strategy and combined performance metrics, an ASCII equity curve,
and saves backtest_trades.csv / backtest_equity.csv to the project root.

Usage
-----
    python -m bot.backtest
    python -m bot.backtest --start 2024-01-01 --end 2025-12-31
    python -m bot.backtest --equity 50000 --start 2023-01-01
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd

from bot import alpaca_client as tradeapi
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ATR_PERIOD,
    MEAN_REVERSION_PERIOD,
    MEAN_REVERSION_SYMBOLS,
    MOMENTUM_BREAKOUT_SYMBOLS,
    MOMENTUM_PERIOD,
    MOMENTUM_TRAILING_ATR,
    RISK_PER_TRADE,
    STD_DEV_THRESHOLDS,
    TREND_FAST_EMA,
    TREND_FOLLOWING_SYMBOLS,
    TREND_SLOW_EMA,
    TREND_TRAILING_ATR,
    VOLUME_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Trade(NamedTuple):
    symbol: str
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float


@dataclass
class _Position:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_price: float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _atr_series(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def _size_int(atr: float, equity: float, price: float, cap: float) -> int:
    """Whole-share size: 1 ATR loss = 1% equity, capped so notional <= cap."""
    if atr <= 0 or price <= 0:
        return 0
    atr_size = int((equity * RISK_PER_TRADE) / atr)
    if atr_size <= 0:
        return 0
    return min(atr_size, int(cap / price))


def _size_frac(atr: float, equity: float, price: float, cap: float) -> float:
    """Fractional size for crypto, capped so notional <= cap."""
    if atr <= 0 or price <= 0:
        return 0.0
    atr_size = (equity * RISK_PER_TRADE) / atr
    return round(min(atr_size, cap / price), 6)


def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        return df.xs(symbol, level=0)
    return df


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_equity(api, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    bars = api.get_bars(
        symbol, timeframe,
        start=start, end=end,
        limit=50000, feed="iex", adjustment="raw",
    ).df
    bars = _flatten(bars, symbol)
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars


def _fetch_crypto(api, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    try:
        bars = api.get_crypto_bars(symbol, timeframe, start=start, end=end, limit=50000).df
        bars = _flatten(bars, symbol)
    except AttributeError:
        bars = api.get_bars("BTCUSD", timeframe, start=start, end=end, limit=50000).df
        bars = _flatten(bars, "BTCUSD")
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars


# ---------------------------------------------------------------------------
# Mean Reversion — SPY / QQQ, 15-min bars
# ---------------------------------------------------------------------------

def backtest_mean_reversion(
    api, symbol: str, start: str, end: str, initial_equity: float,
    custom_thresholds: dict | None = None,
) -> list[Trade]:
    """
    Entry:  close < lower band (long) or close > upper band (short).
            Signal fires at bar close; order fills at the NEXT bar's open.
    Exit:   close crosses back to SMA (filled at that bar's close).
    Stop:   hard stop at 1 ATR from entry (checked on bar low/high).
    """
    print(f"  [{symbol}] fetching 15-min bars ...", end=" ", flush=True)
    try:
        bars = _fetch_equity(api, symbol, "15Min", start, end)
    except Exception as exc:
        print(f"FAILED — {exc}")
        return []
    print(f"{len(bars):,} bars")

    atr_s = _atr_series(bars)
    closes = bars["close"]
    sma = closes.rolling(MEAN_REVERSION_PERIOD).mean()
    std = closes.rolling(MEAN_REVERSION_PERIOD).std()
    thresholds = custom_thresholds if custom_thresholds is not None else STD_DEV_THRESHOLDS
    threshold = thresholds.get(symbol, 1.5)
    upper = sma + threshold * std
    lower = sma - threshold * std

    trades: list[Trade] = []
    pos: _Position | None = None
    pending: dict | None = None  # entry queued on prior bar's signal; fills at next bar's open
    equity = initial_equity
    warmup = MEAN_REVERSION_PERIOD + ATR_PERIOD

    for i in range(warmup, len(bars)):
        bar = bars.iloc[i]
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        bar_close = float(bar["close"])
        bar_time = bars.index[i]

        cur_atr = float(atr_s.iloc[i])
        cur_sma = float(sma.iloc[i])
        cur_upper = float(upper.iloc[i])
        cur_lower = float(lower.iloc[i])

        if pd.isna(cur_atr) or pd.isna(cur_sma):
            pending = None  # discard stale pending; indicators not yet valid
            continue

        # Fill pending entry at this bar's open (signal was from the previous bar's close)
        if pending is not None and pos is None:
            d = pending["direction"]
            entry_price = bar_open
            stop = (round(entry_price - pending["atr"], 4) if d == "long"
                    else round(entry_price + pending["atr"], 4))
            pos = _Position(symbol, d, bar_time, entry_price, pending["size"], stop)
            pending = None

        if pos is not None:
            # Hard stop (fill at stop price; gap-through risk not modelled)
            if pos.direction == "long" and bar_low <= pos.stop_price:
                pnl = (pos.stop_price - pos.entry_price) * pos.size
                trades.append(Trade(symbol, "Mean Reversion", "long",
                                    pos.entry_time, bar_time,
                                    pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                equity += pnl
                pos = None
                continue

            if pos.direction == "short" and bar_high >= pos.stop_price:
                pnl = (pos.entry_price - pos.stop_price) * pos.size
                trades.append(Trade(symbol, "Mean Reversion", "short",
                                    pos.entry_time, bar_time,
                                    pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                equity += pnl
                pos = None
                continue

            # Mean-reversion exit at SMA
            if pos.direction == "long" and bar_close >= cur_sma:
                pnl = (bar_close - pos.entry_price) * pos.size
                trades.append(Trade(symbol, "Mean Reversion", "long",
                                    pos.entry_time, bar_time,
                                    pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                equity += pnl
                pos = None

            elif pos.direction == "short" and bar_close <= cur_sma:
                pnl = (pos.entry_price - bar_close) * pos.size
                trades.append(Trade(symbol, "Mean Reversion", "short",
                                    pos.entry_time, bar_time,
                                    pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                equity += pnl
                pos = None

        else:
            # No open position: queue an entry for the next bar's open
            if bar_close < cur_lower:
                size = _size_int(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    pending = {"direction": "long", "size": size, "atr": cur_atr}

            elif bar_close > cur_upper:
                size = _size_int(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    pending = {"direction": "short", "size": size, "atr": cur_atr}

    # Close open position at end of data
    if pos is not None:
        final = float(bars["close"].iloc[-1])
        mult = 1 if pos.direction == "long" else -1
        pnl = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "Mean Reversion", pos.direction,
                            pos.entry_time, bars.index[-1],
                            pos.entry_price, final, pos.size, round(pnl, 2)))

    return trades


# ---------------------------------------------------------------------------
# Momentum Breakout — BTC/USD, 1-hour bars
# ---------------------------------------------------------------------------

def backtest_momentum_breakout(
    api, symbol: str, start: str, end: str, initial_equity: float
) -> list[Trade]:
    """
    Entry:  close breaks 20-period high/low with volume > 1.5× average.
            Signal fires at bar close; order fills at the NEXT bar's open.
    Exit:   trailing stop at 2× ATR (ratcheted) or direction reversal (longs only).

    Note: equity bars are fetched from the IEX feed, which captures only a
    fraction of total market volume.  The 1.5× volume filter is calibrated to
    full-market volume and may fire more (or less) often on IEX-only data.
    Crypto bars are unaffected (separate endpoint).
    """
    print(f"  [{symbol}] fetching 1-hour bars ...", end=" ", flush=True)
    try:
        bars = _fetch_crypto(api, symbol, "1Hour", start, end)
    except Exception as exc:
        print(f"FAILED — {exc}")
        return []
    print(f"{len(bars):,} bars")

    atr_s = _atr_series(bars)
    trades: list[Trade] = []
    pos: _Position | None = None
    pending: dict | None = None  # entry queued on prior bar's signal; fills at next bar's open
    equity = initial_equity
    warmup = MOMENTUM_PERIOD + ATR_PERIOD

    for i in range(warmup, len(bars)):
        bar = bars.iloc[i]
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        bar_close = float(bar["close"])
        bar_vol = float(bar["volume"])
        bar_time = bars.index[i]

        cur_atr = float(atr_s.iloc[i])
        if pd.isna(cur_atr) or cur_atr <= 0:
            pending = None
            continue

        lookback = bars.iloc[i - MOMENTUM_PERIOD: i]
        period_high = float(lookback["high"].max())
        period_low = float(lookback["low"].min())
        avg_vol = float(lookback["volume"].mean())
        vol_ok = bar_vol >= VOLUME_MULTIPLIER * avg_vol

        # Fill pending entry at this bar's open (signal was from the previous bar's close)
        if pending is not None and pos is None:
            d = pending["direction"]
            entry_price = bar_open
            stop = (round(entry_price - MOMENTUM_TRAILING_ATR * pending["atr"], 4) if d == "long"
                    else round(entry_price + MOMENTUM_TRAILING_ATR * pending["atr"], 4))
            pos = _Position(symbol, d, bar_time, entry_price, pending["size"], stop)
            pending = None

        if pos is not None:
            if pos.direction == "long":
                # Ratchet trailing stop upward
                candidate = round(bar_close - MOMENTUM_TRAILING_ATR * cur_atr, 4)
                if candidate > pos.stop_price:
                    pos.stop_price = candidate

                if bar_low <= pos.stop_price:
                    pnl = (pos.stop_price - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Momentum Breakout", "long",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    continue

                # Direction reversal exit
                if bar_close < period_low:
                    pnl = (bar_close - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Momentum Breakout", "long",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    continue

            elif pos.direction == "short":
                # Ratchet trailing stop downward
                candidate = round(bar_close + MOMENTUM_TRAILING_ATR * cur_atr, 4)
                if candidate < pos.stop_price:
                    pos.stop_price = candidate

                if bar_high >= pos.stop_price:
                    pnl = (pos.entry_price - pos.stop_price) * pos.size
                    trades.append(Trade(symbol, "Momentum Breakout", "short",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    continue

        else:
            # No open position: queue an entry for the next bar's open
            if bar_close > period_high and vol_ok:
                size = _size_frac(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    pending = {"direction": "long", "size": size, "atr": cur_atr}

            elif bar_close < period_low and vol_ok:
                size = _size_frac(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    pending = {"direction": "short", "size": size, "atr": cur_atr}

    if pos is not None:
        final = float(bars["close"].iloc[-1])
        mult = 1 if pos.direction == "long" else -1
        pnl = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "Momentum Breakout", pos.direction,
                            pos.entry_time, bars.index[-1],
                            pos.entry_price, final, pos.size, round(pnl, 2)))

    return trades


# ---------------------------------------------------------------------------
# Trend Following — GLD / USO, 4-hour bars (resampled from 1H)
# ---------------------------------------------------------------------------

def backtest_trend_following(
    api, symbol: str, start: str, end: str, initial_equity: float
) -> list[Trade]:
    """
    Entry:  50 EMA crosses above 200 EMA (long) or below (short).
            Signal fires at bar close; order fills at the NEXT bar's open.
    Exit:   opposite crossover (at that bar's close) or trailing stop at 3× ATR.
    Bars:   1H fetched with 400-day pre-start warm-up, resampled to 4H.
    """
    try:
        dt_start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    except ValueError:
        dt_start = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    fetch_from = (dt_start - timedelta(days=400)).strftime("%Y-%m-%d")

    print(f"  [{symbol}] fetching 1-hour bars (+ 400-day warm-up) ...", end=" ", flush=True)
    try:
        bars_1h = _fetch_equity(api, symbol, "1Hour", fetch_from, end)
    except Exception as exc:
        print(f"FAILED — {exc}")
        return []

    bars = (
        bars_1h.resample("4h", offset="30min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    print(f"{len(bars):,} 4H bars (from {len(bars_1h):,} 1H bars)")

    if len(bars) < TREND_SLOW_EMA + 2:
        print(f"  Not enough 4H bars for {symbol} (need {TREND_SLOW_EMA + 2})")
        return []

    closes = bars["close"]
    fast_ema = closes.ewm(span=TREND_FAST_EMA, adjust=False).mean()
    slow_ema = closes.ewm(span=TREND_SLOW_EMA, adjust=False).mean()
    atr_s = _atr_series(bars)

    trades: list[Trade] = []
    pos: _Position | None = None
    pending: dict | None = None  # entry queued on prior bar's signal; fills at next bar's open
    equity = initial_equity
    start_ts = pd.Timestamp(start, tz="UTC")

    for i in range(1, len(bars)):
        bar_time = bars.index[i]
        if bar_time < start_ts:
            continue  # skip warm-up period; let EMAs stabilise

        bar = bars.iloc[i]
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        bar_close = float(bar["close"])

        cur_atr = float(atr_s.iloc[i])
        if pd.isna(cur_atr) or cur_atr <= 0:
            pending = None
            continue

        fast_curr = float(fast_ema.iloc[i])
        fast_prev = float(fast_ema.iloc[i - 1])
        slow_curr = float(slow_ema.iloc[i])
        slow_prev = float(slow_ema.iloc[i - 1])

        golden_cross = fast_prev < slow_prev and fast_curr >= slow_curr
        death_cross = fast_prev > slow_prev and fast_curr <= slow_curr

        # Fill pending entry at this bar's open (signal was from the previous bar's close)
        if pending is not None and pos is None:
            d = pending["direction"]
            entry_price = bar_open
            stop = (round(entry_price - TREND_TRAILING_ATR * pending["atr"], 4) if d == "long"
                    else round(entry_price + TREND_TRAILING_ATR * pending["atr"], 4))
            pos = _Position(symbol, d, bar_time, entry_price, pending["size"], stop)
            pending = None

        if pos is not None:
            if pos.direction == "long":
                candidate = round(bar_close - TREND_TRAILING_ATR * cur_atr, 4)
                if candidate > pos.stop_price:
                    pos.stop_price = candidate

                if bar_low <= pos.stop_price:
                    pnl = (pos.stop_price - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Trend Following", "long",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    continue  # skip signal detection; re-evaluate on the next bar

                elif death_cross:
                    pnl = (bar_close - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Trend Following", "long",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    # fall through to queue the opposite entry below

            elif pos.direction == "short":
                candidate = round(bar_close + TREND_TRAILING_ATR * cur_atr, 4)
                if candidate < pos.stop_price:
                    pos.stop_price = candidate

                if bar_high >= pos.stop_price:
                    pnl = (pos.entry_price - pos.stop_price) * pos.size
                    trades.append(Trade(symbol, "Trend Following", "short",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    continue  # skip signal detection; re-evaluate on the next bar

                elif golden_cross:
                    pnl = (pos.entry_price - bar_close) * pos.size
                    trades.append(Trade(symbol, "Trend Following", "short",
                                        pos.entry_time, bar_time,
                                        pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                    equity += pnl
                    pos = None
                    # fall through to queue the opposite entry below

        # Queue a next-bar entry when flat (crossover exit falls through to here)
        if pos is None and pending is None:
            size = _size_int(cur_atr, equity, bar_close, initial_equity)
            if golden_cross and size > 0:
                pending = {"direction": "long", "size": size, "atr": cur_atr}
            elif death_cross and size > 0:
                pending = {"direction": "short", "size": size, "atr": cur_atr}

    if pos is not None:
        final = float(bars["close"].iloc[-1])
        mult = 1 if pos.direction == "long" else -1
        pnl = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "Trend Following", pos.direction,
                            pos.entry_time, bars.index[-1],
                            pos.entry_price, final, pos.size, round(pnl, 2)))

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(trades: list[Trade], initial_equity: float, n_days: int) -> dict:
    empty = {
        "n_trades": 0, "total_pnl": 0.0, "total_return_pct": 0.0,
        "annualized_return_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
        "max_drawdown_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "equity_curve": [initial_equity],
    }
    if not trades:
        return empty

    # Sort chronologically so the equity curve and drawdown reflect actual time order.
    # This matters for multi-symbol strategies where trades from different instruments
    # are interleaved in time but were appended symbol-by-symbol.
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    pnls = [t.pnl for t in sorted_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    total_return = total_pnl / initial_equity
    total_return_pct = total_return * 100

    years = n_days / 365.25
    if years > 0 and total_return > -1:
        ann_return_pct = ((1 + total_return) ** (1 / years) - 1) * 100
    else:
        ann_return_pct = 0.0

    win_rate = len(wins) / len(pnls) * 100
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    # Equity curve (time-ordered)
    equity_curve = [initial_equity]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)

    # Max drawdown
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peak) / peak * 100
    max_dd = float(np.min(drawdowns))

    # Per-trade Sharpe: each trade's return is P&L divided by equity before that trade,
    # not by the fixed initial_equity.  This stays consistent as the account compounds.
    if len(pnls) > 1:
        eq_before = np.array(equity_curve[:-1])  # equity at start of each trade's exit
        ret_arr = np.array(pnls) / eq_before
        trades_per_year = len(pnls) / years if years > 0 else len(pnls)
        ann_factor = np.sqrt(trades_per_year)
        sharpe = float(np.mean(ret_arr) / np.std(ret_arr, ddof=1) * ann_factor)
        downside = ret_arr[ret_arr < 0]
        sortino = float(np.mean(ret_arr) / np.std(downside, ddof=1) * ann_factor) if len(downside) > 1 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    calmar = ann_return_pct / abs(max_dd) if max_dd < 0 else float("inf")

    return {
        "n_trades": len(pnls),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(ann_return_pct, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2) if calmar != float("inf") else float("inf"),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Equity curve (time-indexed, monthly sampling)
# ---------------------------------------------------------------------------

def _build_time_curve(
    trades: list[Trade], initial_equity: float, start: str, end: str
) -> tuple[list[str], list[float]]:
    """Returns (date_labels, equity_values) sampled monthly."""
    dt_start = pd.Timestamp(start, tz="UTC")
    dt_end = pd.Timestamp(end, tz="UTC")
    dates = pd.date_range(dt_start, dt_end, freq="MS")

    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    values = []
    for d in dates:
        cum_pnl = sum(t.pnl for t in sorted_trades if t.exit_time <= d)
        values.append(initial_equity + cum_pnl)

    return [str(d.date()) for d in dates], values


def _print_ascii_curve(
    dates: list[str], values: list[float], label: str, height: int = 12
) -> None:
    if len(values) < 2:
        print("  (no data)\n")
        return

    min_v, max_v = min(values), max(values)
    span = max_v - min_v or 1.0
    n = len(values)

    # Prefer Unicode block chars; fall back to ASCII if the console can't render them
    try:
        sparks = " ▁▂▃▄▅▆▇█"
        line = "".join(sparks[int((v - min_v) / span * (len(sparks) - 1))] for v in values)
        # Test that the console can encode these
        line.encode(sys.stdout.encoding or "ascii")
        hrule = "─" * (n + 2)
        top_corner, bot_corner = "┐", "┘"
    except (UnicodeEncodeError, LookupError):
        sparks = " ._-:=+*#"
        line = "".join(sparks[int((v - min_v) / span * (len(sparks) - 1))] for v in values)
        hrule = "-" * (n + 2)
        top_corner, bot_corner = "|", "|"

    print(f"\n  {label}")
    print(f"  {hrule}")
    print(f"  ${max_v:>10,.0f} {top_corner}")
    print(f"   {'':>10}  {line}")
    print(f"  ${min_v:>10,.0f} {bot_corner}")

    # Date labels: start / mid / end
    start_lbl = dates[0][:7]
    mid_lbl = dates[n // 2][:7]
    end_lbl = dates[-1][:7]
    pad = n // 2 - len(start_lbl)
    tail = n - n // 2 - len(mid_lbl) - len(end_lbl)
    print(f"   {'':>10}  {start_lbl}{' ' * max(0, pad)}{mid_lbl}{' ' * max(0, tail)}{end_lbl}")
    print()


# ---------------------------------------------------------------------------
# Monthly returns and losing streak
# ---------------------------------------------------------------------------

def _monthly_returns(trades: list[Trade], initial_equity: float) -> dict[tuple[int, int], float]:
    """Monthly return percentages keyed by (year, month), using running equity as the base."""
    if not trades:
        return {}
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    monthly_pnl: dict[tuple[int, int], float] = {}
    for t in sorted_trades:
        key = (t.exit_time.year, t.exit_time.month)
        monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t.pnl

    result: dict[tuple[int, int], float] = {}
    equity = initial_equity
    yr = sorted_trades[0].exit_time.year
    mo = sorted_trades[0].exit_time.month
    end_yr = sorted_trades[-1].exit_time.year
    end_mo = sorted_trades[-1].exit_time.month

    while (yr, mo) <= (end_yr, end_mo):
        pnl = monthly_pnl.get((yr, mo), 0.0)
        result[(yr, mo)] = pnl / equity * 100 if equity > 0 else 0.0
        equity += pnl
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1

    return result


_MNAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _print_monthly_table(monthly: dict[tuple[int, int], float]) -> None:
    if not monthly:
        return
    years = sorted({yr for yr, _ in monthly})
    W = 7

    print(f"\n  {'Year':4} " + " ".join(f"{m:>{W}}" for m in _MNAMES) + f"  {'Annual':>{W}}")
    print(f"  {'----':4} " + " ".join("-" * W for _ in _MNAMES) + f"  {'-' * W}")

    for yr in years:
        cells = []
        for mo in range(1, 13):
            key = (yr, mo)
            if key in monthly:
                cells.append(f"{monthly[key]:>+6.1f}%")
            else:
                cells.append(f"{'--':>{W}}")

        ks = [(yr, mo) for mo in range(1, 13) if (yr, mo) in monthly]
        if ks:
            compound = 1.0
            for k in ks:
                compound *= (1 + monthly[k] / 100)
            annual_s = f"{(compound - 1) * 100:>+6.1f}%"
        else:
            annual_s = f"{'--':>{W}}"

        print(f"  {yr:4} " + " ".join(cells) + f"  {annual_s}")


def _worst_streak(trades: list[Trade]) -> tuple[int, float]:
    """Longest consecutive losing streak by trade count, plus total P&L of that run."""
    if not trades:
        return 0, 0.0
    best, best_pnl = 0, 0.0
    cur, cur_pnl = 0, 0.0
    for t in sorted(trades, key=lambda t: t.exit_time):
        if t.pnl < 0:
            cur += 1
            cur_pnl += t.pnl
            if cur > best:
                best, best_pnl = cur, cur_pnl
        else:
            cur, cur_pnl = 0, 0.0
    return best, round(best_pnl, 2)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_strategy_table(name: str, m: dict) -> None:
    inf_str = "  inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:>5.2f}x"
    calmar_str = "   inf" if m["calmar"] == float("inf") else f"{m['calmar']:>6.2f}"
    print(f"\n  {'-' * 52}")
    print(f"  {name}")
    print(f"  {'-' * 52}")
    print(f"  Trades          {m['n_trades']:>6}")
    print(f"  Win rate        {m['win_rate']:>5.1f}%")
    print(f"  Total P&L       ${m['total_pnl']:>10,.2f}")
    print(f"  Total return    {m['total_return_pct']:>+6.2f}%")
    print(f"  CAGR            {m['annualized_return_pct']:>+6.2f}%")
    print(f"  Profit factor   {inf_str}")
    print(f"  Max drawdown    {m['max_drawdown_pct']:>+6.2f}%")
    print(f"  Sharpe (ann.)   {m['sharpe']:>6.2f}")
    print(f"  Sortino (ann.)  {m['sortino']:>6.2f}")
    print(f"  Calmar ratio    {calmar_str}")
    print(f"  Avg win         ${m['avg_win']:>10,.2f}")
    print(f"  Avg loss        ${m['avg_loss']:>10,.2f}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_trades(trades: list[Trade], path: str = "backtest_trades.csv") -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "symbol", "direction", "entry_time", "exit_time",
                    "entry_price", "exit_price", "size", "pnl"])
        for t in sorted(trades, key=lambda x: x.entry_time):
            w.writerow([t.strategy, t.symbol, t.direction,
                        t.entry_time, t.exit_time,
                        round(t.entry_price, 4), round(t.exit_price, 4),
                        round(t.size, 6), t.pnl])
    print(f"  Trades saved -> {path}")


def _save_equity_curve(
    dates: list[str], values: list[float], path: str = "backtest_equity.csv"
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "equity"])
        for d, v in zip(dates, values):
            w.writerow([d, round(v, 2)])
    print(f"  Equity curve saved -> {path}")


def _try_plot(dates: list[str], values: list[float], start: str, end: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, values, linewidth=1.5, color="#2196f3")
        ax.axhline(values[0], color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.fill_between(ts, values[0], values, alpha=0.08, color="#2196f3")
        ax.set_title(f"Combined Equity Curve  ({start} → {end})", fontsize=13)
        ax.set_ylabel("Portfolio Equity ($)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=30, ha="right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        plt.tight_layout()
        out = "backtest_equity_curve.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Chart saved -> {out}")
    except ImportError:
        pass  # matplotlib optional
    except Exception as exc:
        print(f"  (chart skipped: {exc})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest all trading strategies.")
    now = datetime.now(timezone.utc)
    default_end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    default_start = (now - timedelta(days=730)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default=default_start,
                        help="Backtest start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", default=default_end,
                        help="Backtest end date YYYY-MM-DD (default: 3 days ago)")
    parser.add_argument("--equity", type=float, default=100_000.0,
                        help="Starting equity per symbol (default: 100000)")
    parser.add_argument("--spy-std", type=float, default=None,
                        help="Override SPY std-dev threshold (default: from config)")
    parser.add_argument("--qqq-std", type=float, default=None,
                        help="Override QQQ std-dev threshold (default: from config)")
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    initial_equity = args.equity
    start, end = args.start, args.end

    custom_thresholds: dict | None = None
    if args.spy_std is not None or args.qqq_std is not None:
        custom_thresholds = dict(STD_DEV_THRESHOLDS)
        if args.spy_std is not None:
            custom_thresholds["SPY"] = args.spy_std
        if args.qqq_std is not None:
            custom_thresholds["QQQ"] = args.qqq_std
    n_days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days

    print(f"\n{'=' * 56}")
    print(f"  BACKTEST  {start} -> {end}  (equity ${initial_equity:,.0f}/symbol)")
    print(f"{'=' * 56}")

    # --- Mean Reversion (SPY / QQQ) only ---
    if custom_thresholds:
        thr_display = ", ".join(f"{k}={v}" for k, v in custom_thresholds.items())
        print(f"\nMean Reversion (SPY / QQQ)  [custom thresholds: {thr_display}]")
    else:
        print("\nMean Reversion (SPY / QQQ)")
    trades: list[Trade] = []
    for sym in MEAN_REVERSION_SYMBOLS:
        trades.extend(backtest_mean_reversion(api, sym, start, end, initial_equity,
                                              custom_thresholds=custom_thresholds))
    total_equity = initial_equity * len(MEAN_REVERSION_SYMBOLS)
    m = _compute_metrics(trades, total_equity, n_days)
    _print_strategy_table(
        f"Mean Reversion  (SPY + QQQ)  [${total_equity:,.0f} total]", m
    )

    # Monthly returns
    monthly = _monthly_returns(trades, total_equity)
    print("\n  Monthly returns (%)")
    _print_monthly_table(monthly)

    # Worst losing streak
    streak_len, streak_pnl = _worst_streak(trades)
    print(f"\n  Worst losing streak  {streak_len} consecutive losses  (${streak_pnl:,.2f} total)")

    # Equity curve
    dates, values = _build_time_curve(trades, total_equity, start, end)
    _print_ascii_curve(dates, values, f"Equity  ${total_equity:,.0f} start")

    # Save outputs
    print("Saving outputs...")
    _save_trades(trades)
    _save_equity_curve(dates, values)
    _try_plot(dates, values, start, end)

    print(f"\n{'=' * 56}\n")


if __name__ == "__main__":
    main()
