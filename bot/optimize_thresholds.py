# -*- coding: utf-8 -*-
"""
Grid search over SPY / QQQ std-dev thresholds for the mean reversion strategy.

Fetches bars once per symbol, then sweeps all threshold combinations in memory.
Prints a ranked comparison table sorted by Calmar ratio.

Usage
-----
    python -m bot.optimize_thresholds
    python -m bot.optimize_thresholds --start 2024-01-01 --end 2025-12-31
    python -m bot.optimize_thresholds --equity 100000
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd

from bot import alpaca_client as tradeapi
from bot.backtest import (
    Trade,
    _Position,
    _atr_series,
    _flatten,
    _size_int,
    _worst_streak,
)
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ATR_PERIOD,
    MEAN_REVERSION_PERIOD,
    RISK_PER_TRADE,
)


# ---------------------------------------------------------------------------
# Simulation (pre-fetched bars, variable threshold)
# ---------------------------------------------------------------------------

def _simulate(
    bars: pd.DataFrame,
    symbol: str,
    threshold: float,
    initial_equity: float,
) -> list[Trade]:
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
            pending = None
            continue

        if pending is not None and pos is None:
            d = pending["direction"]
            entry_price = bar_open
            stop = (round(entry_price - pending["atr"], 4) if d == "long"
                    else round(entry_price + pending["atr"], 4))
            pos = _Position(symbol, d, bar_time, entry_price, pending["size"], stop)
            pending = None

        if pos is not None:
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
            if bar_close < cur_lower:
                size = _size_int(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    pending = {"direction": "long", "size": size, "atr": cur_atr}
            elif bar_close > cur_upper:
                size = _size_int(cur_atr, equity, bar_close, initial_equity)
                if size > 0:
                    pending = {"direction": "short", "size": size, "atr": cur_atr}

    if pos is not None:
        final = float(bars["close"].iloc[-1])
        mult = 1 if pos.direction == "long" else -1
        pnl = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(symbol, "Mean Reversion", pos.direction,
                            pos.entry_time, bars.index[-1],
                            pos.entry_price, final, pos.size, round(pnl, 2)))

    return trades


# ---------------------------------------------------------------------------
# Metrics (lightweight, no printing)
# ---------------------------------------------------------------------------

def _metrics(trades: list[Trade], initial_equity: float, n_days: int) -> dict:
    if not trades:
        return {"n_trades": 0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "max_dd": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "worst_streak": 0, "worst_streak_pnl": 0.0}

    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    pnls = [t.pnl for t in sorted_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    total_return = total_pnl / initial_equity
    years = n_days / 365.25
    cagr = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 and total_return > -1 else 0.0

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    equity_curve = [initial_equity]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)

    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peak) / peak * 100
    max_dd = float(np.min(drawdowns))

    if len(pnls) > 1:
        eq_before = np.array(equity_curve[:-1])
        ret_arr = np.array(pnls) / eq_before
        trades_per_year = len(pnls) / years if years > 0 else len(pnls)
        ann_factor = np.sqrt(trades_per_year)
        sharpe = float(np.mean(ret_arr) / np.std(ret_arr, ddof=1) * ann_factor)
        downside = ret_arr[ret_arr < 0]
        sortino = float(np.mean(ret_arr) / np.std(downside, ddof=1) * ann_factor) if len(downside) > 1 else 0.0
    else:
        sharpe = sortino = 0.0

    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")
    streak_len, streak_pnl = _worst_streak(trades)

    return {
        "n_trades": len(pnls),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2) if calmar != float("inf") else 999.0,
        "max_dd": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "worst_streak": streak_len,
        "worst_streak_pnl": streak_pnl,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search SPY/QQQ std-dev thresholds.")
    now = datetime.now(timezone.utc)
    default_end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    default_start = (now - timedelta(days=730)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--equity", type=float, default=100_000.0,
                        help="Starting equity per symbol (default: 100000)")
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    start, end = args.start, args.end
    initial_equity = args.equity
    n_days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    total_equity = initial_equity * 2  # SPY + QQQ

    # Fetch bars once per symbol
    print(f"\n{'=' * 62}")
    print(f"  THRESHOLD OPTIMIZER  {start} -> {end}  (${initial_equity:,.0f}/symbol)")
    print(f"{'=' * 62}")
    print("\nFetching bars (one-time)...")

    def fetch(symbol: str) -> pd.DataFrame:
        bars = api.get_bars(
            symbol, "15Min", start=start, end=end,
            limit=50000, feed="iex", adjustment="raw",
        ).df
        bars = _flatten(bars, symbol)
        bars.index = pd.to_datetime(bars.index, utc=True)
        print(f"  [{symbol}] {len(bars):,} bars")
        return bars

    spy_bars = fetch("SPY")
    qqq_bars = fetch("QQQ")

    # Grid: 0.25 to 2.0 in steps of 0.25
    spy_thresholds = [round(v, 2) for v in np.arange(0.25, 2.01, 0.25)]
    qqq_thresholds = [round(v, 2) for v in np.arange(0.30, 2.41, 0.30)]

    # Highlight these known combinations
    KNOWN = {(1.5, 1.8): "original", (0.75, 0.9): "halved"}

    print(f"\nRunning {len(spy_thresholds) * len(qqq_thresholds)} combinations "
          f"({len(spy_thresholds)} SPY × {len(qqq_thresholds)} QQQ) ...")

    results = []
    total = len(spy_thresholds) * len(qqq_thresholds)
    done = 0
    for spy_t, qqq_t in product(spy_thresholds, qqq_thresholds):
        spy_trades = _simulate(spy_bars, "SPY", spy_t, initial_equity)
        qqq_trades = _simulate(qqq_bars, "QQQ", qqq_t, initial_equity)
        combined = spy_trades + qqq_trades
        m = _metrics(combined, total_equity, n_days)
        results.append({"spy": spy_t, "qqq": qqq_t, **m})
        done += 1
        if done % 16 == 0 or done == total:
            print(f"  {done}/{total} ...", end="\r")

    print()

    results.sort(key=lambda r: r["calmar"], reverse=True)

    # ---------------------------------------------------------------------------
    # Full grid table
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print("  FULL RESULTS  (sorted by Calmar)")
    print(f"{'=' * 62}")
    HDR = (f"  {'SPY':>5} {'QQQ':>5}  {'Trades':>6}  {'WR%':>5}  "
           f"{'CAGR%':>6}  {'Sharpe':>6}  {'Calmar':>6}  {'MaxDD%':>7}  "
           f"{'WorstStrk':>9}  {'Label'}")
    print(HDR)
    print(f"  {'-' * 94}")

    for r in results:
        label = KNOWN.get((r["spy"], r["qqq"]), "")
        calmar_s = f"{r['calmar']:>6.2f}" if r["calmar"] < 999 else "   inf"
        print(
            f"  {r['spy']:>5.2f} {r['qqq']:>5.2f}  {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
            f"{r['cagr']:>+6.2f}%  {r['sharpe']:>6.2f}  {calmar_s}  "
            f"{r['max_dd']:>+6.2f}%  {r['worst_streak']:>5} / ${r['worst_streak_pnl']:>8,.0f}"
            f"  {label}"
        )

    # ---------------------------------------------------------------------------
    # Top 10 by Calmar
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print("  TOP 10 BY CALMAR")
    print(f"{'=' * 62}")
    print(HDR)
    print(f"  {'-' * 94}")
    for r in results[:10]:
        label = KNOWN.get((r["spy"], r["qqq"]), "")
        calmar_s = f"{r['calmar']:>6.2f}" if r["calmar"] < 999 else "   inf"
        print(
            f"  {r['spy']:>5.2f} {r['qqq']:>5.2f}  {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
            f"{r['cagr']:>+6.2f}%  {r['sharpe']:>6.2f}  {calmar_s}  "
            f"{r['max_dd']:>+6.2f}%  {r['worst_streak']:>5} / ${r['worst_streak_pnl']:>8,.0f}"
            f"  {label}"
        )

    # ---------------------------------------------------------------------------
    # Top 10 by Sharpe
    # ---------------------------------------------------------------------------
    by_sharpe = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    print(f"\n{'=' * 62}")
    print("  TOP 10 BY SHARPE")
    print(f"{'=' * 62}")
    print(HDR)
    print(f"  {'-' * 94}")
    for r in by_sharpe[:10]:
        label = KNOWN.get((r["spy"], r["qqq"]), "")
        calmar_s = f"{r['calmar']:>6.2f}" if r["calmar"] < 999 else "   inf"
        print(
            f"  {r['spy']:>5.2f} {r['qqq']:>5.2f}  {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
            f"{r['cagr']:>+6.2f}%  {r['sharpe']:>6.2f}  {calmar_s}  "
            f"{r['max_dd']:>+6.2f}%  {r['worst_streak']:>5} / ${r['worst_streak_pnl']:>8,.0f}"
            f"  {label}"
        )

    # ---------------------------------------------------------------------------
    # Top 10 by CAGR
    # ---------------------------------------------------------------------------
    by_cagr = sorted(results, key=lambda r: r["cagr"], reverse=True)
    print(f"\n{'=' * 62}")
    print("  TOP 10 BY CAGR")
    print(f"{'=' * 62}")
    print(HDR)
    print(f"  {'-' * 94}")
    for r in by_cagr[:10]:
        label = KNOWN.get((r["spy"], r["qqq"]), "")
        calmar_s = f"{r['calmar']:>6.2f}" if r["calmar"] < 999 else "   inf"
        print(
            f"  {r['spy']:>5.2f} {r['qqq']:>5.2f}  {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
            f"{r['cagr']:>+6.2f}%  {r['sharpe']:>6.2f}  {calmar_s}  "
            f"{r['max_dd']:>+6.2f}%  {r['worst_streak']:>5} / ${r['worst_streak_pnl']:>8,.0f}"
            f"  {label}"
        )

    # ---------------------------------------------------------------------------
    # Where do the known combos rank?
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print("  KNOWN COMBINATIONS")
    print(f"{'=' * 62}")
    print(HDR)
    print(f"  {'-' * 94}")
    for r in results:
        label = KNOWN.get((r["spy"], r["qqq"]), "")
        if not label:
            continue
        calmar_s = f"{r['calmar']:>6.2f}" if r["calmar"] < 999 else "   inf"
        rank_calmar = next(i + 1 for i, x in enumerate(results) if x["spy"] == r["spy"] and x["qqq"] == r["qqq"])
        rank_sharpe = next(i + 1 for i, x in enumerate(by_sharpe) if x["spy"] == r["spy"] and x["qqq"] == r["qqq"])
        print(
            f"  {r['spy']:>5.2f} {r['qqq']:>5.2f}  {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
            f"{r['cagr']:>+6.2f}%  {r['sharpe']:>6.2f}  {calmar_s}  "
            f"{r['max_dd']:>+6.2f}%  {r['worst_streak']:>5} / ${r['worst_streak_pnl']:>8,.0f}"
            f"  {label}  [Calmar rank #{rank_calmar}, Sharpe rank #{rank_sharpe}]"
        )

    print(f"\n{'=' * 62}\n")


if __name__ == "__main__":
    main()
