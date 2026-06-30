# -*- coding: utf-8 -*-
"""
Combined 4-symbol mean-reversion backtest.

  SPY  threshold=0.25   (best CAGR from grid search)
  QQQ  threshold=0.30   (best CAGR from grid search)
  GLD  threshold=0.25   (best CAGR from grid search)
  USO  threshold=0.25   (best CAGR from grid search)

Fetches the full available 15-min history for each symbol, runs all four
in parallel, then reports per-symbol stats and a combined portfolio view
with monthly returns table and ASCII equity curve.

Usage
-----
    python -m bot.backtest_combined
    python -m bot.backtest_combined --equity 100000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bot import alpaca_client as tradeapi
from bot.backtest import Trade, _flatten, _worst_streak
from bot.optimize_thresholds import _simulate, _metrics
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

START = "2020-07-01"

SYMBOLS = [
    {"sym": "SPY", "threshold": 0.25},
    {"sym": "QQQ", "threshold": 0.30},
    {"sym": "GLD", "threshold": 0.25},
    {"sym": "USO", "threshold": 0.25},
]

_MNAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fetch(api, symbol: str, start: str, end: str) -> pd.DataFrame:
    bars = api.get_bars(
        symbol, "15Min", start=start, end=end,
        limit=1_000_000, feed="iex", adjustment="raw",
    ).df
    bars = _flatten(bars, symbol)
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars


def _monthly_returns(trades: list[Trade], initial_equity: float) -> dict:
    if not trades:
        return {}
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    monthly_pnl: dict = {}
    for t in sorted_trades:
        key = (t.exit_time.year, t.exit_time.month)
        monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t.pnl

    result: dict = {}
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


def _print_monthly(monthly: dict, initial_equity: float) -> None:
    if not monthly:
        return
    years = sorted({yr for yr, _ in monthly})
    W = 7
    losing = sum(1 for v in monthly.values() if v < 0)
    total  = len(monthly)
    print(f"\n  Monthly returns  ({losing} losing months out of {total})")
    print(f"  {'Year':4} " + " ".join(f"{m:>{W}}" for m in _MNAMES) + f"  {'Annual':>{W}}")
    print(f"  {'----':4} " + " ".join("-" * W for _ in _MNAMES) + f"  {'-' * W}")
    for yr in years:
        cells = []
        for mo in range(1, 13):
            key = (yr, mo)
            cells.append(f"{monthly[key]:>+6.1f}%" if key in monthly else f"{'--':>{W}}")
        ks = [(yr, mo) for mo in range(1, 13) if (yr, mo) in monthly]
        if ks:
            compound = 1.0
            for k in ks:
                compound *= (1 + monthly[k] / 100)
            annual_s = f"{(compound - 1) * 100:>+6.1f}%"
        else:
            annual_s = f"{'--':>{W}}"
        print(f"  {yr:4} " + " ".join(cells) + f"  {annual_s}")


def _ascii_curve(trades: list[Trade], initial_equity: float,
                 start: str, end: str) -> None:
    dt_start = pd.Timestamp(start, tz="UTC")
    dt_end   = pd.Timestamp(end,   tz="UTC")
    dates    = pd.date_range(dt_start, dt_end, freq="MS")
    sorted_t = sorted(trades, key=lambda t: t.exit_time)
    values   = [initial_equity + sum(t.pnl for t in sorted_t if t.exit_time <= d)
                for d in dates]

    if len(values) < 2:
        return

    min_v, max_v = min(values), max(values)
    span = max_v - min_v or 1.0
    n    = len(values)

    try:
        sparks = " ▁▂▃▄▅▆▇█"
        line   = "".join(sparks[int((v - min_v) / span * (len(sparks) - 1))] for v in values)
        line.encode(sys.stdout.encoding or "ascii")
        hrule  = "─" * (n + 2)
    except (UnicodeEncodeError, LookupError):
        sparks = " ._-:=+*#"
        line   = "".join(sparks[int((v - min_v) / span * (len(sparks) - 1))] for v in values)
        hrule  = "-" * (n + 2)

    print(f"\n  Equity curve  (${initial_equity:,.0f} start)")
    print(f"  {hrule}")
    print(f"  ${max_v:>12,.0f} |")
    print(f"   {'':>12}  {line}")
    print(f"  ${min_v:>12,.0f} |")
    lbl_s = str(dates[0].date())[:7]
    lbl_m = str(dates[n // 2].date())[:7]
    lbl_e = str(dates[-1].date())[:7]
    pad   = n // 2 - len(lbl_s)
    tail  = n - n // 2 - len(lbl_m) - len(lbl_e)
    print(f"   {'':>12}  {lbl_s}{' ' * max(0,pad)}{lbl_m}{' ' * max(0,tail)}{lbl_e}\n")


def _print_metrics_row(label: str, m: dict, width: int = 22) -> None:
    calmar_s = f"{m['calmar']:>6.2f}" if m["calmar"] < 999 else "   inf"
    print(
        f"  {label:<{width}}  {m['n_trades']:>6}  {m['win_rate']:>5.1f}%  "
        f"{m['cagr']:>+7.2f}%  {m['sharpe']:>6.2f}  {calmar_s}  "
        f"{m['max_dd']:>+7.2f}%  "
        f"{m['worst_streak']:>3} / ${m['worst_streak_pnl']:>8,.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    now = datetime.now(timezone.utc)
    default_end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    parser.add_argument("--start",  default=START)
    parser.add_argument("--end",    default=default_end)
    parser.add_argument("--equity", type=float, default=100_000.0)
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing"); sys.exit(1)

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    eq  = args.equity
    end = args.end

    print(f"\n{'=' * 72}")
    print(f"  COMBINED BACKTEST — SPY/QQQ/GLD/USO  Mean Reversion")
    print(f"  Target window: {args.start} -> {end}   ${eq:,.0f}/symbol  (${eq*4:,.0f} total)")
    print(f"{'=' * 72}")

    # ---- Fetch & simulate each symbol ----
    print("\nFetching 15-min bars...")
    sym_results: list[dict] = []
    all_trades:  list[Trade] = []
    actual_start = actual_end = None

    for cfg in SYMBOLS:
        sym   = cfg["sym"]
        thresh = cfg["threshold"]
        bars  = _fetch(api, sym, args.start, end)
        n_days = (bars.index[-1] - bars.index[0]).days
        print(f"  [{sym}]  {len(bars):,} bars  "
              f"({bars.index[0].date()} -> {bars.index[-1].date()})"
              f"  threshold={thresh}")

        trades = _simulate(bars, sym, thresh, eq)
        m      = _metrics(trades, eq, n_days)
        sym_results.append({"sym": sym, "thresh": thresh, **m, "n_days": n_days})
        all_trades.extend(trades)

        if actual_start is None or bars.index[0] < actual_start:
            actual_start = bars.index[0]
        if actual_end is None or bars.index[-1] > actual_end:
            actual_end = bars.index[-1]

    total_equity = eq * len(SYMBOLS)
    combined_days = (actual_end - actual_start).days
    combined_m = _metrics(all_trades, total_equity, combined_days)

    # ---- Per-symbol table ----
    HDR_LINE = (f"  {'Symbol (thr)':<22}  {'Trades':>6}  {'WR%':>6}  "
                f"{'CAGR%':>8}  {'Sharpe':>6}  {'Calmar':>6}  "
                f"{'MaxDD%':>8}  {'Worst streak':>14}")

    print(f"\n{'=' * 72}")
    print("  PER-SYMBOL BREAKDOWN")
    print(f"{'=' * 72}")
    print(HDR_LINE)
    print(f"  {'-' * 90}")
    for r in sym_results:
        label = f"{r['sym']} (thr={r['thresh']})"
        _print_metrics_row(label, r)

    print(f"  {'-' * 90}")
    _print_metrics_row("COMBINED PORTFOLIO", combined_m)
    print(f"  (${total_equity:,.0f} total capital)")

    # ---- Combined metrics detail ----
    print(f"\n{'=' * 72}")
    print("  COMBINED PORTFOLIO — FULL METRICS")
    print(f"  Window: {actual_start.date()} -> {actual_end.date()}"
          f"  ({combined_days} days / {combined_days/365.25:.1f} years)")
    print(f"{'=' * 72}")
    calmar_s = f"{combined_m['calmar']:.2f}" if combined_m["calmar"] < 999 else "inf"
    print(f"  Trades            {combined_m['n_trades']:>8,}")
    print(f"  Win rate          {combined_m['win_rate']:>7.1f}%")
    print(f"  Total P&L         ${combined_m['n_trades'] and sum(t.pnl for t in all_trades):>12,.2f}")
    print(f"  CAGR              {combined_m['cagr']:>+7.2f}%")
    print(f"  Sharpe            {combined_m['sharpe']:>8.2f}")
    print(f"  Sortino           {combined_m['sortino']:>8.2f}")
    print(f"  Calmar            {calmar_s:>8}")
    print(f"  Max drawdown      {combined_m['max_dd']:>+7.2f}%")
    print(f"  Avg win           ${combined_m['avg_win']:>10,.2f}")
    print(f"  Avg loss          ${combined_m['avg_loss']:>10,.2f}")
    print(f"  Profit factor     {combined_m['profit_factor']:>8.2f}x")
    print(f"  Worst streak      {combined_m['worst_streak']} losses  "
          f"(${combined_m['worst_streak_pnl']:,.2f})")

    # ---- Monthly returns ----
    monthly = _monthly_returns(all_trades, total_equity)
    _print_monthly(monthly, total_equity)

    # ---- Equity curve ----
    _ascii_curve(all_trades, total_equity, str(actual_start.date()), str(actual_end.date()))

    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
