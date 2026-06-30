# -*- coding: utf-8 -*-
"""
Head-to-head comparison of 4 mean-reversion threshold configurations.

Fetches the maximum available history for SPY and QQQ (15-min bars),
then runs all 4 strategies and prints a side-by-side table.

Usage
-----
    python -m bot.compare_strategies
    python -m bot.compare_strategies --start 2016-01-01
    python -m bot.compare_strategies --equity 100000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bot import alpaca_client as tradeapi
from bot.backtest import Trade, _Position, _atr_series, _flatten, _size_int, _worst_streak
from bot.optimize_thresholds import _simulate, _metrics
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    MEAN_REVERSION_PERIOD,
)

STRATEGIES = [
    {"label": "Original",     "spy": 1.50, "qqq": 1.80},
    {"label": "Halved",       "spy": 0.75, "qqq": 0.90},
    {"label": "Best Calmar",  "spy": 2.00, "qqq": 0.30},
    {"label": "Best CAGR",    "spy": 0.25, "qqq": 0.30},
]

_MNAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fetch(api, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch all available 15-min bars, paginating automatically."""
    bars = api.get_bars(
        symbol, "15Min",
        start=start, end=end,
        limit=1_000_000,
        feed="iex",
        adjustment="raw",
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


def _print_monthly(monthly: dict, label: str) -> None:
    if not monthly:
        return
    years = sorted({yr for yr, _ in monthly})
    W = 7
    print(f"\n  {label} — Monthly returns (%)")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 4 threshold strategies.")
    now = datetime.now(timezone.utc)
    default_end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default="2016-01-01",
                        help="Backtest start date (default: 2016-01-01)")
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--equity", type=float, default=100_000.0)
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    initial_equity = args.equity
    total_equity = initial_equity * 2
    end = args.end

    # Fetch bars — use requested start but clamp to what actually exists
    print(f"\n{'=' * 70}")
    print(f"  4-STRATEGY COMPARISON  (target: {args.start} -> {end})")
    print(f"  Equity: ${initial_equity:,.0f}/symbol  (${total_equity:,.0f} total)")
    print(f"{'=' * 70}")
    print("\nFetching bars (one-time per symbol)...")

    spy_bars = _fetch(api, "SPY", args.start, end)
    qqq_bars = _fetch(api, "QQQ", args.start, end)

    # Actual date range from returned data
    actual_start = min(spy_bars.index[0], qqq_bars.index[0])
    actual_end   = max(spy_bars.index[-1], qqq_bars.index[-1])
    n_days = (actual_end - actual_start).days

    print(f"  [SPY] {len(spy_bars):,} bars  ({spy_bars.index[0].date()} -> {spy_bars.index[-1].date()})")
    print(f"  [QQQ] {len(qqq_bars):,} bars  ({qqq_bars.index[0].date()} -> {qqq_bars.index[-1].date()})")
    print(f"  Actual window: {n_days} days ({n_days/365.25:.1f} years)")

    # Run all 4 strategies
    print("\nRunning simulations...")
    results = []
    for s in STRATEGIES:
        spy_trades = _simulate(spy_bars, "SPY", s["spy"], initial_equity)
        qqq_trades = _simulate(qqq_bars, "QQQ", s["qqq"], initial_equity)
        combined = spy_trades + qqq_trades
        m = _metrics(combined, total_equity, n_days)
        results.append({**s, **m, "trades": combined})
        print(f"  [{s['label']:12}]  {m['n_trades']:,} trades")

    # ---------------------------------------------------------------------------
    # Side-by-side comparison table
    # ---------------------------------------------------------------------------
    COL = 18
    labels = [r["label"] for r in results]

    def row(name: str, vals: list[str]) -> str:
        return f"  {name:<22}" + "".join(f"{v:>{COL}}" for v in vals)

    print(f"\n{'=' * 94}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 94}")
    print(row("", labels))
    print(row("", [f"SPY={r['spy']} / QQQ={r['qqq']}" for r in results]))
    print(f"  {'-' * 91}")
    print(row("Trades",        [f"{r['n_trades']:,}" for r in results]))
    print(row("Win rate",      [f"{r['win_rate']:.1f}%" for r in results]))
    print(row("CAGR",          [f"{r['cagr']:+.2f}%" for r in results]))
    print(row("Sharpe",        [f"{r['sharpe']:.2f}" for r in results]))
    print(row("Sortino",       [f"{r['sortino']:.2f}" for r in results]))
    print(row("Calmar",        [f"{r['calmar']:.2f}" for r in results]))
    print(row("Max drawdown",  [f"{r['max_dd']:+.2f}%" for r in results]))
    print(row("Avg win",       [f"${r['avg_win']:,.0f}" for r in results]))
    print(row("Avg loss",      [f"${r['avg_loss']:,.0f}" for r in results]))
    print(row("Worst streak",  [f"{r['worst_streak']} losses" for r in results]))
    print(row("  streak P&L",  [f"${r['worst_streak_pnl']:,.0f}" for r in results]))
    print(row("Profit factor", [f"{r['profit_factor']:.2f}x" for r in results]))
    print(f"{'=' * 94}")

    # Rank each strategy per metric (1 = best)
    def rank_metric(key: str, higher_is_better: bool = True) -> list[int]:
        vals = [r[key] for r in results]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=higher_is_better)
        ranks = [0] * len(vals)
        for pos, idx in enumerate(order):
            ranks[idx] = pos + 1
        return ranks

    rc = rank_metric("cagr")
    rs = rank_metric("sharpe")
    rl = rank_metric("calmar")
    rd = rank_metric("max_dd", higher_is_better=False)  # less negative = better
    rw = rank_metric("worst_streak", higher_is_better=False)

    print(f"\n  RANKINGS (1 = best per metric)")
    print(row("", labels))
    print(f"  {'-' * 91}")
    print(row("CAGR rank",      [f"#{r}" for r in rc]))
    print(row("Sharpe rank",    [f"#{r}" for r in rs]))
    print(row("Calmar rank",    [f"#{r}" for r in rl]))
    print(row("Max DD rank",    [f"#{r}" for r in rd]))
    print(row("Streak rank",    [f"#{r}" for r in rw]))

    overall = [rc[i] + rs[i] + rl[i] + rd[i] + rw[i] for i in range(len(results))]
    print(row("Overall score",  [f"{s} pts" for s in overall]))
    print(f"  {'(lower = better across all 5 metrics)'}")
    print(f"{'=' * 94}")

    # ---------------------------------------------------------------------------
    # Monthly returns per strategy
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 94}")
    print("  MONTHLY RETURNS BY STRATEGY")
    print(f"{'=' * 94}")
    for r in results:
        monthly = _monthly_returns(r["trades"], total_equity)
        losing_months = sum(1 for v in monthly.values() if v < 0)
        total_months = len(monthly)
        _print_monthly(monthly, f"{r['label']} (SPY={r['spy']}, QQQ={r['qqq']})")
        print(f"  Losing months: {losing_months}/{total_months}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
