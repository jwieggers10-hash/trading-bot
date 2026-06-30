# -*- coding: utf-8 -*-
"""
Full-parameter optimizer for GLD (gold), USO (oil), and BTC/USD.

For each instrument tests three strategy types:
  1. Mean Reversion   — 15-min bars (equities) / 1H bars (BTC, 24/7 market)
  2. Momentum Breakout — 1-hour bars
  3. Trend Following  — 4-hour bars (resampled from 1H), EMA crossover

Fetches bars once per instrument, then sweeps all parameter combinations
in memory. Reports top-5 by Calmar and top-5 by CAGR per instrument,
plus a final cross-instrument summary.

Usage
-----
    python -m bot.optimize_assets
    python -m bot.optimize_assets --equity 100000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd

from bot import alpaca_client as tradeapi
from bot.backtest import _atr_series, _flatten, _size_int, _size_frac, Trade, _Position
from bot.optimize_thresholds import _metrics
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ATR_PERIOD,
    MEAN_REVERSION_PERIOD,
)

START = "2020-07-01"   # earliest available on IEX 15-min feed


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

MR_THRESHOLDS   = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
MOM_PERIODS     = [10, 20, 30]
MOM_VOL_MULTS   = [1.0, 1.5, 2.0]   # 1.0 = no filter
MOM_TRAIL_ATRS  = [1.5, 2.0, 3.0]
TREND_FAST_EMAS = [20, 50]
TREND_SLOW_EMAS = [100, 150, 200]
TREND_TRAIL_ATRS = [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch(api, symbol: str, timeframe: str, start: str, end: str,
           crypto: bool = False) -> pd.DataFrame:
    if crypto:
        try:
            bars = api.get_crypto_bars(symbol, timeframe, start=start, end=end,
                                       limit=1_000_000).df
            bars = _flatten(bars, symbol)
        except AttributeError:
            sym = symbol.replace("/", "")
            bars = api.get_bars(sym, timeframe, start=start, end=end,
                                limit=1_000_000).df
            bars = _flatten(bars, sym)
    else:
        bars = api.get_bars(
            symbol, timeframe, start=start, end=end,
            limit=1_000_000, feed="iex", adjustment="raw",
        ).df
        bars = _flatten(bars, symbol)
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars


def _resample_4h(bars_1h: pd.DataFrame) -> pd.DataFrame:
    return (
        bars_1h.resample("4h", offset="30min")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )


# ---------------------------------------------------------------------------
# Simulation: Mean Reversion
# ---------------------------------------------------------------------------

def sim_mean_rev(bars: pd.DataFrame, symbol: str, threshold: float,
                 initial_equity: float, frac: bool = False) -> list[Trade]:
    atr_s = _atr_series(bars)
    closes = bars["close"]
    sma = closes.rolling(MEAN_REVERSION_PERIOD).mean()
    std = closes.rolling(MEAN_REVERSION_PERIOD).std()
    upper = sma + threshold * std
    lower = sma - threshold * std

    trades: list[Trade] = []
    pos: _Position | None = None
    pending: dict | None = None
    equity = initial_equity
    warmup = MEAN_REVERSION_PERIOD + ATR_PERIOD

    for i in range(warmup, len(bars)):
        bar = bars.iloc[i]
        bar_open, bar_low, bar_high, bar_close = (
            float(bar["open"]), float(bar["low"]),
            float(bar["high"]), float(bar["close"]))
        bar_time = bars.index[i]
        cur_atr  = float(atr_s.iloc[i])
        cur_sma  = float(sma.iloc[i])
        cur_upper = float(upper.iloc[i])
        cur_lower = float(lower.iloc[i])

        if pd.isna(cur_atr) or pd.isna(cur_sma):
            pending = None
            continue

        if pending is not None and pos is None:
            d = pending["direction"]
            ep = bar_open
            stop = round(ep - pending["atr"], 4) if d == "long" else round(ep + pending["atr"], 4)
            pos = _Position(symbol, d, bar_time, ep, pending["size"], stop)
            pending = None

        if pos is not None:
            if pos.direction == "long" and bar_low <= pos.stop_price:
                pnl = (pos.stop_price - pos.entry_price) * pos.size
                trades.append(Trade(symbol, "MeanRev", "long", pos.entry_time, bar_time,
                                    pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                equity += pnl; pos = None; continue
            if pos.direction == "short" and bar_high >= pos.stop_price:
                pnl = (pos.entry_price - pos.stop_price) * pos.size
                trades.append(Trade(symbol, "MeanRev", "short", pos.entry_time, bar_time,
                                    pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                equity += pnl; pos = None; continue
            if pos.direction == "long" and bar_close >= cur_sma:
                pnl = (bar_close - pos.entry_price) * pos.size
                trades.append(Trade(symbol, "MeanRev", "long", pos.entry_time, bar_time,
                                    pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                equity += pnl; pos = None
            elif pos.direction == "short" and bar_close <= cur_sma:
                pnl = (pos.entry_price - bar_close) * pos.size
                trades.append(Trade(symbol, "MeanRev", "short", pos.entry_time, bar_time,
                                    pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                equity += pnl; pos = None
        else:
            size = (_size_frac if frac else _size_int)(cur_atr, equity, bar_close, initial_equity)
            if size > 0:
                if bar_close < cur_lower:
                    pending = {"direction": "long",  "size": size, "atr": cur_atr}
                elif bar_close > cur_upper:
                    pending = {"direction": "short", "size": size, "atr": cur_atr}

    if pos is not None:
        final = float(bars["close"].iloc[-1])
        mult = 1 if pos.direction == "long" else -1
        pnl = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "MeanRev", pos.direction, pos.entry_time,
                            bars.index[-1], pos.entry_price, final, pos.size, round(pnl, 2)))
    return trades


# ---------------------------------------------------------------------------
# Simulation: Momentum Breakout
# ---------------------------------------------------------------------------

def sim_momentum(bars: pd.DataFrame, symbol: str,
                 period: int, vol_mult: float, trail_atr: float,
                 initial_equity: float, frac: bool = False) -> list[Trade]:
    atr_s = _atr_series(bars)
    trades: list[Trade] = []
    pos: _Position | None = None
    pending: dict | None = None
    equity = initial_equity
    warmup = period + ATR_PERIOD

    for i in range(warmup, len(bars)):
        bar = bars.iloc[i]
        bar_open, bar_low, bar_high, bar_close = (
            float(bar["open"]), float(bar["low"]),
            float(bar["high"]), float(bar["close"]))
        bar_vol  = float(bar["volume"])
        bar_time = bars.index[i]
        cur_atr  = float(atr_s.iloc[i])

        if pd.isna(cur_atr) or cur_atr <= 0:
            pending = None; continue

        lookback    = bars.iloc[i - period: i]
        period_high = float(lookback["high"].max())
        period_low  = float(lookback["low"].min())
        avg_vol     = float(lookback["volume"].mean())
        vol_ok      = (vol_mult <= 1.0) or (bar_vol >= vol_mult * avg_vol)

        if pending is not None and pos is None:
            d  = pending["direction"]
            ep = bar_open
            stop = (round(ep - trail_atr * pending["atr"], 4) if d == "long"
                    else round(ep + trail_atr * pending["atr"], 4))
            pos = _Position(symbol, d, bar_time, ep, pending["size"], stop)
            pending = None

        if pos is not None:
            if pos.direction == "long":
                cand = round(bar_close - trail_atr * cur_atr, 4)
                if cand > pos.stop_price:
                    pos.stop_price = cand
                if bar_low <= pos.stop_price:
                    pnl = (pos.stop_price - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Momentum", "long", pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None; continue
                if bar_close < period_low:
                    pnl = (bar_close - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Momentum", "long", pos.entry_time, bar_time,
                                        pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None; continue
            else:
                cand = round(bar_close + trail_atr * cur_atr, 4)
                if cand < pos.stop_price:
                    pos.stop_price = cand
                if bar_high >= pos.stop_price:
                    pnl = (pos.entry_price - pos.stop_price) * pos.size
                    trades.append(Trade(symbol, "Momentum", "short", pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None; continue
        else:
            if vol_ok:
                size = (_size_frac if frac else _size_int)(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    if bar_close > period_high:
                        pending = {"direction": "long",  "size": size, "atr": cur_atr}
                    elif bar_close < period_low:
                        pending = {"direction": "short", "size": size, "atr": cur_atr}

    if pos is not None:
        final = float(bars["close"].iloc[-1])
        mult  = 1 if pos.direction == "long" else -1
        pnl   = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "Momentum", pos.direction, pos.entry_time,
                            bars.index[-1], pos.entry_price, final, pos.size, round(pnl, 2)))
    return trades


# ---------------------------------------------------------------------------
# Simulation: Trend Following (4H bars pre-resampled)
# ---------------------------------------------------------------------------

def sim_trend(bars_4h: pd.DataFrame, symbol: str,
              fast_ema: int, slow_ema: int, trail_atr: float,
              initial_equity: float, frac: bool = False) -> list[Trade]:
    if len(bars_4h) < slow_ema + 2:
        return []

    closes   = bars_4h["close"]
    fast_s   = closes.ewm(span=fast_ema, adjust=False).mean()
    slow_s   = closes.ewm(span=slow_ema, adjust=False).mean()
    atr_s    = _atr_series(bars_4h)

    trades: list[Trade] = []
    pos: _Position | None = None
    pending: dict | None = None
    equity = initial_equity

    for i in range(1, len(bars_4h)):
        bar = bars_4h.iloc[i]
        bar_open, bar_low, bar_high, bar_close = (
            float(bar["open"]), float(bar["low"]),
            float(bar["high"]), float(bar["close"]))
        bar_time = bars_4h.index[i]
        cur_atr  = float(atr_s.iloc[i])

        if pd.isna(cur_atr) or cur_atr <= 0:
            pending = None; continue

        fast_curr = float(fast_s.iloc[i]);   fast_prev = float(fast_s.iloc[i - 1])
        slow_curr = float(slow_s.iloc[i]);   slow_prev = float(slow_s.iloc[i - 1])
        golden = fast_prev < slow_prev and fast_curr >= slow_curr
        death  = fast_prev > slow_prev and fast_curr <= slow_curr

        if pending is not None and pos is None:
            d  = pending["direction"]
            ep = bar_open
            stop = (round(ep - trail_atr * pending["atr"], 4) if d == "long"
                    else round(ep + trail_atr * pending["atr"], 4))
            pos = _Position(symbol, d, bar_time, ep, pending["size"], stop)
            pending = None

        if pos is not None:
            if pos.direction == "long":
                cand = round(bar_close - trail_atr * cur_atr, 4)
                if cand > pos.stop_price:
                    pos.stop_price = cand
                if bar_low <= pos.stop_price:
                    pnl = (pos.stop_price - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Trend", "long", pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None; continue
                elif death:
                    pnl = (bar_close - pos.entry_price) * pos.size
                    trades.append(Trade(symbol, "Trend", "long", pos.entry_time, bar_time,
                                        pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None
            elif pos.direction == "short":
                cand = round(bar_close + trail_atr * cur_atr, 4)
                if cand < pos.stop_price:
                    pos.stop_price = cand
                if bar_high >= pos.stop_price:
                    pnl = (pos.entry_price - pos.stop_price) * pos.size
                    trades.append(Trade(symbol, "Trend", "short", pos.entry_time, bar_time,
                                        pos.entry_price, pos.stop_price, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None; continue
                elif golden:
                    pnl = (pos.entry_price - bar_close) * pos.size
                    trades.append(Trade(symbol, "Trend", "short", pos.entry_time, bar_time,
                                        pos.entry_price, bar_close, pos.size, round(pnl, 2)))
                    equity += pnl; pos = None

        if pos is None and pending is None:
            size = (_size_frac if frac else _size_int)(cur_atr, equity, bar_close, initial_equity)
            if golden and size > 0:
                pending = {"direction": "long",  "size": size, "atr": cur_atr}
            elif death and size > 0:
                pending = {"direction": "short", "size": size, "atr": cur_atr}

    if pos is not None:
        final = float(bars_4h["close"].iloc[-1])
        mult  = 1 if pos.direction == "long" else -1
        pnl   = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "Trend", pos.direction, pos.entry_time,
                            bars_4h.index[-1], pos.entry_price, final, pos.size, round(pnl, 2)))
    return trades


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _print_top(results: list[dict], key: str, n: int, label: str, n_days: int) -> None:
    ranked = sorted(results, key=lambda r: r[key], reverse=True)[:n]
    years  = n_days / 365.25
    print(f"\n  TOP {n} BY {label.upper()}")
    print(f"  {'Strategy':<12} {'Params':<30} {'Trades':>6}  {'WR%':>5}  "
          f"{'CAGR%':>7}  {'Sharpe':>6}  {'Calmar':>6}  {'MaxDD%':>7}  {'WorstStrk':>9}")
    print(f"  {'-' * 100}")
    for r in ranked:
        calmar_s = f"{r['calmar']:>6.2f}" if r["calmar"] < 999 else "   inf"
        print(
            f"  {r['strategy']:<12} {r['params']:<30} {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
            f"{r['cagr']:>+7.2f}%  {r['sharpe']:>6.2f}  {calmar_s}  "
            f"{r['max_dd']:>+6.2f}%  "
            f"{r['worst_streak']:>3} / ${r['worst_streak_pnl']:>7,.0f}"
        )


def _sweep(results: list[dict], label: str, symbol: str, trades_list: list[Trade],
           strategy: str, params: str, equity: float, n_days: int) -> None:
    m = _metrics(trades_list, equity, n_days)
    if m["n_trades"] < 10:
        return
    results.append({
        "symbol": symbol, "strategy": strategy, "params": params,
        **m,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    now = datetime.now(timezone.utc)
    default_end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end",   default=default_end)
    parser.add_argument("--equity", type=float, default=100_000.0)
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing"); sys.exit(1)

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    eq  = args.equity
    end = args.end

    INSTRUMENTS = [
        {"name": "GLD",    "label": "GOLD",  "equity_sym": "GLD",   "crypto": False},
        {"name": "USO",    "label": "OIL",   "equity_sym": "USO",   "crypto": False},
        {"name": "BTC/USD","label": "BTC",   "equity_sym": "BTC/USD","crypto": True},
    ]

    all_bests: list[dict] = []

    for inst in INSTRUMENTS:
        sym    = inst["name"]
        label  = inst["label"]
        frac   = inst["crypto"]
        print(f"\n{'=' * 70}")
        print(f"  {label} ({sym})")
        print(f"{'=' * 70}")

        # ---- Fetch data ----
        print("  Fetching bars...")
        # 15-min bars (equity) or 1H bars (BTC) for mean reversion
        if frac:
            bars_mr = _fetch(api, sym, "1Hour", args.start, end, crypto=True)
            print(f"  [1H  / MeanRev] {len(bars_mr):,} bars  "
                  f"({bars_mr.index[0].date()} -> {bars_mr.index[-1].date()})")
        else:
            bars_mr = _fetch(api, sym, "15Min", args.start, end, crypto=False)
            print(f"  [15m / MeanRev] {len(bars_mr):,} bars  "
                  f"({bars_mr.index[0].date()} -> {bars_mr.index[-1].date()})")

        bars_1h = _fetch(api, sym, "1Hour", args.start, end, crypto=frac)
        print(f"  [1H  / Mom+Trd] {len(bars_1h):,} bars")

        bars_4h = _resample_4h(bars_1h)
        print(f"  [4H  / Trend  ] {len(bars_4h):,} bars")

        n_days_mr  = (bars_mr.index[-1]  - bars_mr.index[0]).days
        n_days_1h  = (bars_1h.index[-1]  - bars_1h.index[0]).days
        n_days_4h  = (bars_4h.index[-1]  - bars_4h.index[0]).days

        results: list[dict] = []
        total_combos = len(MR_THRESHOLDS) + len(MOM_PERIODS)*len(MOM_VOL_MULTS)*len(MOM_TRAIL_ATRS) \
                       + len(TREND_FAST_EMAS)*len(TREND_SLOW_EMAS)*len(TREND_TRAIL_ATRS)
        done = 0

        # ---- Mean Reversion sweep ----
        for thresh in MR_THRESHOLDS:
            trades = sim_mean_rev(bars_mr, sym, thresh, eq, frac=frac)
            _sweep(results, label, sym, trades, "MeanRev",
                   f"thr={thresh}", eq, n_days_mr)
            done += 1
            print(f"  {done}/{total_combos} combos...", end="\r")

        # ---- Momentum Breakout sweep ----
        for period, vol_m, trail in product(MOM_PERIODS, MOM_VOL_MULTS, MOM_TRAIL_ATRS):
            trades = sim_momentum(bars_1h, sym, period, vol_m, trail, eq, frac=frac)
            _sweep(results, label, sym, trades, "Momentum",
                   f"p={period} vol={vol_m} trl={trail}", eq, n_days_1h)
            done += 1
            print(f"  {done}/{total_combos} combos...", end="\r")

        # ---- Trend Following sweep ----
        for fast, slow, trail in product(TREND_FAST_EMAS, TREND_SLOW_EMAS, TREND_TRAIL_ATRS):
            trades = sim_trend(bars_4h, sym, fast, slow, trail, eq, frac=frac)
            _sweep(results, label, sym, trades, "Trend",
                   f"ema={fast}/{slow} trl={trail}", eq, n_days_4h)
            done += 1
            print(f"  {done}/{total_combos} combos...", end="\r")

        print(f"  {done}/{total_combos} combos done.  "
              f"({len(results)} valid results)")

        _print_top(results, "calmar", 5, "Calmar", n_days_1h)
        _print_top(results, "cagr",   5, "CAGR",   n_days_1h)

        # Store absolute best Calmar and best CAGR per instrument
        if results:
            best_c = max(results, key=lambda r: r["calmar"])
            best_g = max(results, key=lambda r: r["cagr"])
            all_bests.append({"label": label, "metric": "Calmar", **best_c})
            if best_g["params"] != best_c["params"]:
                all_bests.append({"label": label, "metric": "CAGR",   **best_g})

    # ---------------------------------------------------------------------------
    # Cross-instrument summary
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("  FINAL SUMMARY — BEST PER INSTRUMENT")
    print(f"{'=' * 70}")
    print(f"  {'Asset':<6} {'Optimised for':<14} {'Strategy':<12} {'Params':<30} "
          f"{'CAGR%':>7}  {'Sharpe':>6}  {'Calmar':>6}  {'MaxDD%':>7}  {'Streak':>6}")
    print(f"  {'-' * 108}")
    for b in all_bests:
        calmar_s = f"{b['calmar']:>6.2f}" if b["calmar"] < 999 else "   inf"
        print(
            f"  {b['label']:<6} {b['metric']:<14} {b['strategy']:<12} {b['params']:<30} "
            f"{b['cagr']:>+7.2f}%  {b['sharpe']:>6.2f}  {calmar_s}  "
            f"{b['max_dd']:>+6.2f}%  "
            f"{b['worst_streak']:>3} / ${b['worst_streak_pnl']:>7,.0f}"
        )
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
