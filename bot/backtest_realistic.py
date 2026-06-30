# -*- coding: utf-8 -*-
"""
Realistic pooled-capital backtest for SPY/QQQ/GLD/USO.

Key improvements over the naive per-symbol simulation:
  - ONE shared capital pool ($400k total)
  - Capital is physically locked when a position is open, released on close
  - A new entry is blocked (or reduced) if insufficient capital is available
  - Two sizing models tested:
      Fixed cap   — each symbol capped at initial_equity/n_syms regardless of growth
      Dynamic cap — cap grows with total portfolio equity (compounding)
  - Risk percentage swept from 0.25% to 2.5% to find optimal sizing

Usage
-----
    python -m bot.backtest_realistic
    python -m bot.backtest_realistic --equity 400000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bot import alpaca_client as tradeapi
from bot.backtest import Trade, _Position, _atr_series, _flatten
from bot.optimize_thresholds import _metrics
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, MEAN_REVERSION_PERIOD, ATR_PERIOD

START = "2020-07-01"

SYMBOLS_CFG = [
    {"sym": "SPY", "threshold": 0.25},
    {"sym": "QQQ", "threshold": 0.30},
    {"sym": "GLD", "threshold": 0.25},
    {"sym": "USO", "threshold": 0.25},
]
RISK_PCTS   = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025]
_MNAMES     = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _fetch(api, symbol: str, start: str, end: str) -> pd.DataFrame:
    bars = api.get_bars(
        symbol, "15Min", start=start, end=end,
        limit=1_000_000, feed="iex", adjustment="raw",
    ).df
    bars = _flatten(bars, symbol)
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars


# ---------------------------------------------------------------------------
# Pooled simulation
# ---------------------------------------------------------------------------

def simulate_pooled(
    bars_dict: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    initial_equity: float,
    risk_pct: float,
    dynamic_cap: bool = False,
) -> list[Trade]:
    """
    Simulate all symbols sharing a single capital pool.

    risk_pct : fraction of TOTAL portfolio equity to risk per individual trade
               (e.g. 0.01 = 1% of portfolio per trade)
    dynamic_cap : if True, each symbol's notional cap scales with current equity
                  (compounding); if False, cap is fixed at initial_equity/n_syms.
    """
    n_syms        = len(bars_dict)
    init_per_sym  = initial_equity / n_syms
    warmup        = MEAN_REVERSION_PERIOD + ATR_PERIOD

    # Pre-compute indicators and build timestamp → bar-index maps
    ind: dict[str, dict] = {}
    for sym, bars in bars_dict.items():
        thresh  = thresholds[sym]
        closes  = bars["close"]
        atr_arr = _atr_series(bars).values
        sma_arr = closes.rolling(MEAN_REVERSION_PERIOD).mean().values
        std_arr = closes.rolling(MEAN_REVERSION_PERIOD).std().values
        upper   = sma_arr + thresh * std_arr
        lower   = sma_arr - thresh * std_arr
        ind[sym] = {
            "bars"     : bars,
            "atr"      : atr_arr,
            "sma"      : sma_arr,
            "upper"    : upper,
            "lower"    : lower,
            "ts_idx"   : {ts: i for i, ts in enumerate(bars.index)},
        }

    # Merge all timestamps and sort
    all_ts = sorted(set().union(*[set(b.index) for b in bars_dict.values()]))

    # Shared portfolio state
    equity    = initial_equity   # realised equity
    locked    = 0.0              # notional value locked in open positions
    positions : dict[str, _Position] = {}
    pendings  : dict[str, dict]      = {}
    trades    : list[Trade]          = []

    for ts in all_ts:
        for sym in bars_dict:
            si     = ind[sym]
            ts_idx = si["ts_idx"]
            if ts not in ts_idx:
                continue
            i = ts_idx[ts]
            if i < warmup:
                pendings.pop(sym, None)
                continue

            bars      = si["bars"]
            bar       = bars.iloc[i]
            bar_open  = float(bar["open"])
            bar_low   = float(bar["low"])
            bar_high  = float(bar["high"])
            bar_close = float(bar["close"])

            cur_atr   = si["atr"][i]
            cur_sma   = si["sma"][i]
            cur_upper = si["upper"][i]
            cur_lower = si["lower"][i]

            if np.isnan(cur_atr) or np.isnan(cur_sma) or cur_atr <= 0:
                pendings.pop(sym, None)
                continue

            # ---- Fill pending entry at this bar's open ----
            if sym in pendings and sym not in positions:
                pend      = pendings.pop(sym)
                available = max(0.0, equity - locked)
                per_sym_cap = (equity / n_syms) if dynamic_cap else init_per_sym
                max_notional = min(per_sym_cap, available)

                if max_notional >= bar_open:          # can afford at least 1 share
                    risk_dollars = equity * risk_pct  # $ to risk on this trade
                    atr_shares   = int(risk_dollars / cur_atr)
                    cap_shares   = int(max_notional / bar_open)
                    size         = min(atr_shares, cap_shares)

                    if size > 0:
                        d    = pend["direction"]
                        stop = (round(bar_open - cur_atr, 4) if d == "long"
                                else round(bar_open + cur_atr, 4))
                        positions[sym] = _Position(sym, d, ts, bar_open, size, stop)
                        locked += size * bar_open

            pos = positions.get(sym)

            if pos is not None:
                # Hard stop
                if pos.direction == "long" and bar_low <= pos.stop_price:
                    pnl = (pos.stop_price - pos.entry_price) * pos.size
                    trades.append(Trade(sym, "MeanRev", "long",
                                        pos.entry_time, ts,
                                        pos.entry_price, pos.stop_price,
                                        pos.size, round(pnl, 2)))
                    locked -= pos.size * pos.entry_price
                    equity += pnl
                    del positions[sym]
                    continue

                if pos.direction == "short" and bar_high >= pos.stop_price:
                    pnl = (pos.entry_price - pos.stop_price) * pos.size
                    trades.append(Trade(sym, "MeanRev", "short",
                                        pos.entry_time, ts,
                                        pos.entry_price, pos.stop_price,
                                        pos.size, round(pnl, 2)))
                    locked -= pos.size * pos.entry_price
                    equity += pnl
                    del positions[sym]
                    continue

                # SMA exit
                if pos.direction == "long" and bar_close >= cur_sma:
                    pnl = (bar_close - pos.entry_price) * pos.size
                    trades.append(Trade(sym, "MeanRev", "long",
                                        pos.entry_time, ts,
                                        pos.entry_price, bar_close,
                                        pos.size, round(pnl, 2)))
                    locked -= pos.size * pos.entry_price
                    equity += pnl
                    del positions[sym]

                elif pos.direction == "short" and bar_close <= cur_sma:
                    pnl = (pos.entry_price - bar_close) * pos.size
                    trades.append(Trade(sym, "MeanRev", "short",
                                        pos.entry_time, ts,
                                        pos.entry_price, bar_close,
                                        pos.size, round(pnl, 2)))
                    locked -= pos.size * pos.entry_price
                    equity += pnl
                    del positions[sym]

            elif sym not in pendings:
                # Queue signal for next bar's open
                available = max(0.0, equity - locked)
                per_sym_cap = (equity / n_syms) if dynamic_cap else init_per_sym
                max_notional = min(per_sym_cap, available)

                if max_notional >= bar_close:
                    risk_dollars = equity * risk_pct
                    atr_shares   = int(risk_dollars / cur_atr)
                    cap_shares   = int(max_notional / bar_close)
                    size         = min(atr_shares, cap_shares)

                    if size > 0:
                        if bar_close < cur_lower:
                            pendings[sym] = {"direction": "long",  "atr": cur_atr}
                        elif bar_close > cur_upper:
                            pendings[sym] = {"direction": "short", "atr": cur_atr}

    # Close remaining open positions at last price
    for sym, pos in positions.items():
        final = float(ind[sym]["bars"]["close"].iloc[-1])
        mult  = 1 if pos.direction == "long" else -1
        pnl   = (final - pos.entry_price) * pos.size * mult
        trades.append(Trade(sym, "MeanRev", pos.direction,
                            pos.entry_time, ind[sym]["bars"].index[-1],
                            pos.entry_price, final, pos.size, round(pnl, 2)))
    return trades


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _monthly_returns(trades: list[Trade], initial_equity: float) -> dict:
    if not trades:
        return {}
    sorted_t = sorted(trades, key=lambda t: t.exit_time)
    monthly_pnl: dict = {}
    for t in sorted_t:
        key = (t.exit_time.year, t.exit_time.month)
        monthly_pnl[key] = monthly_pnl.get(key, 0.0) + t.pnl
    result: dict = {}
    eq = initial_equity
    yr, mo = sorted_t[0].exit_time.year, sorted_t[0].exit_time.month
    ey, em = sorted_t[-1].exit_time.year, sorted_t[-1].exit_time.month
    while (yr, mo) <= (ey, em):
        pnl = monthly_pnl.get((yr, mo), 0.0)
        result[(yr, mo)] = pnl / eq * 100 if eq > 0 else 0.0
        eq += pnl
        mo += 1
        if mo > 12:
            mo, yr = 1, yr + 1
    return result


def _print_monthly(monthly: dict) -> None:
    if not monthly:
        return
    years  = sorted({yr for yr, _ in monthly})
    W      = 7
    losing = sum(1 for v in monthly.values() if v < 0)
    total  = len(monthly)
    print(f"\n  Monthly returns  ({losing} losing months / {total})")
    print(f"  {'Year':4} " + " ".join(f"{m:>{W}}" for m in _MNAMES) + f"  {'Annual':>{W}}")
    print(f"  {'----':4} " + " ".join("-" * W for _ in _MNAMES) + f"  {'-' * W}")
    for yr in years:
        cells = []
        for mo in range(1, 13):
            key = (yr, mo)
            cells.append(f"{monthly[key]:>+6.1f}%" if key in monthly else f"{'--':>{W}}")
        ks = [(yr, mo) for mo in range(1, 13) if (yr, mo) in monthly]
        compound = 1.0
        for k in ks:
            compound *= (1 + monthly[k] / 100)
        annual_s = f"{(compound-1)*100:>+6.1f}%" if ks else f"{'--':>{W}}"
        print(f"  {yr:4} " + " ".join(cells) + f"  {annual_s}")


def _print_full(m: dict, initial_equity: float, trades: list[Trade], label: str) -> None:
    calmar_s = f"{m['calmar']:.2f}" if m["calmar"] < 999 else "inf"
    total_pnl = sum(t.pnl for t in trades)
    final_eq  = initial_equity + total_pnl
    print(f"\n  Trades            {m['n_trades']:>8,}")
    print(f"  Win rate          {m['win_rate']:>7.1f}%")
    print(f"  Total P&L         ${total_pnl:>12,.2f}")
    print(f"  Final equity      ${final_eq:>12,.2f}")
    print(f"  CAGR              {m['cagr']:>+7.2f}%")
    print(f"  Sharpe            {m['sharpe']:>8.2f}")
    print(f"  Sortino           {m['sortino']:>8.2f}")
    print(f"  Calmar            {calmar_s:>8}")
    print(f"  Max drawdown      {m['max_dd']:>+7.2f}%")
    print(f"  Avg win           ${m['avg_win']:>10,.2f}")
    print(f"  Avg loss          ${m['avg_loss']:>10,.2f}")
    print(f"  Profit factor     {m['profit_factor']:>8.2f}x")
    print(f"  Worst streak      {m['worst_streak']} losses  (${m['worst_streak_pnl']:,.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    now = datetime.now(timezone.utc)
    default_end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    parser.add_argument("--start",  default=START)
    parser.add_argument("--end",    default=default_end)
    parser.add_argument("--equity", type=float, default=400_000.0,
                        help="Total portfolio starting equity (default: 400000)")
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: missing API keys"); sys.exit(1)

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    initial_equity = args.equity
    end            = args.end

    print(f"\n{'=' * 72}")
    print(f"  REALISTIC POOLED-CAPITAL BACKTEST")
    print(f"  SPY(0.25) / QQQ(0.30) / GLD(0.25) / USO(0.25)  —  Mean Reversion")
    print(f"  Total capital: ${initial_equity:,.0f}   Window: {args.start} -> {end}")
    print(f"{'=' * 72}")

    # ---- Fetch bars ----
    print("\nFetching 15-min bars (one-time)...")
    bars_dict  : dict[str, pd.DataFrame] = {}
    thresholds : dict[str, float]        = {}
    for cfg in SYMBOLS_CFG:
        sym   = cfg["sym"]
        bars  = _fetch(api, sym, args.start, end)
        bars_dict[sym]  = bars
        thresholds[sym] = cfg["threshold"]
        print(f"  [{sym}]  {len(bars):,} bars  "
              f"({bars.index[0].date()} -> {bars.index[-1].date()})")

    actual_start = min(b.index[0]  for b in bars_dict.values())
    actual_end   = max(b.index[-1] for b in bars_dict.values())
    n_days       = (actual_end - actual_start).days
    print(f"  Actual window: {n_days} days ({n_days/365.25:.1f} years)\n")

    # ---- Sweep risk percentages — FIXED cap ----
    print(f"{'=' * 72}")
    print(f"  RISK-PCT SWEEP  —  Fixed cap (${initial_equity/len(SYMBOLS_CFG):,.0f}/symbol)")
    print(f"{'=' * 72}")
    HDR = (f"  {'Risk%':>6}  {'Trades':>7}  {'WR%':>5}  {'CAGR%':>7}  "
           f"{'Sharpe':>6}  {'Calmar':>6}  {'MaxDD%':>7}  "
           f"{'FinalEq':>14}  {'Streak':>14}")
    print(HDR)
    print(f"  {'-' * 98}")

    fixed_results = []
    for rp in RISK_PCTS:
        print(f"  {rp*100:.2f}%  running...", end="\r")
        trades = simulate_pooled(bars_dict, thresholds, initial_equity,
                                 risk_pct=rp, dynamic_cap=False)
        m = _metrics(trades, initial_equity, n_days)
        total_pnl = sum(t.pnl for t in trades)
        fixed_results.append({"rp": rp, "m": m, "trades": trades, "pnl": total_pnl})
        calmar_s = f"{m['calmar']:>6.2f}" if m["calmar"] < 999 else "   inf"
        print(
            f"  {rp*100:>5.2f}%  {m['n_trades']:>7,}  {m['win_rate']:>4.1f}%  "
            f"{m['cagr']:>+7.2f}%  {m['sharpe']:>6.2f}  {calmar_s}  "
            f"{m['max_dd']:>+6.2f}%  "
            f"${initial_equity + total_pnl:>13,.0f}  "
            f"{m['worst_streak']:>3}L / ${m['worst_streak_pnl']:>7,.0f}"
        )

    # ---- Sweep risk percentages — DYNAMIC cap ----
    print(f"\n{'=' * 72}")
    print(f"  RISK-PCT SWEEP  —  Dynamic cap (equity/4 per symbol, compounds)")
    print(f"{'=' * 72}")
    print(HDR)
    print(f"  {'-' * 98}")

    dyn_results = []
    for rp in RISK_PCTS:
        print(f"  {rp*100:.2f}%  running...", end="\r")
        trades = simulate_pooled(bars_dict, thresholds, initial_equity,
                                 risk_pct=rp, dynamic_cap=True)
        m = _metrics(trades, initial_equity, n_days)
        total_pnl = sum(t.pnl for t in trades)
        dyn_results.append({"rp": rp, "m": m, "trades": trades, "pnl": total_pnl})
        calmar_s = f"{m['calmar']:>6.2f}" if m["calmar"] < 999 else "   inf"
        print(
            f"  {rp*100:>5.2f}%  {m['n_trades']:>7,}  {m['win_rate']:>4.1f}%  "
            f"{m['cagr']:>+7.2f}%  {m['sharpe']:>6.2f}  {calmar_s}  "
            f"{m['max_dd']:>+6.2f}%  "
            f"${initial_equity + total_pnl:>13,.0f}  "
            f"{m['worst_streak']:>3}L / ${m['worst_streak_pnl']:>7,.0f}"
        )

    # ---- Best configurations ----
    best_fixed_calmar = max(fixed_results, key=lambda r: r["m"]["calmar"])
    best_fixed_cagr   = max(fixed_results, key=lambda r: r["m"]["cagr"])
    best_dyn_calmar   = max(dyn_results,   key=lambda r: r["m"]["calmar"])
    best_dyn_cagr     = max(dyn_results,   key=lambda r: r["m"]["cagr"])

    for label, res in [
        ("FIXED CAP — Best by Calmar",  best_fixed_calmar),
        ("FIXED CAP — Best by CAGR",    best_fixed_cagr),
        ("DYNAMIC CAP — Best by Calmar", best_dyn_calmar),
        ("DYNAMIC CAP — Best by CAGR",   best_dyn_cagr),
    ]:
        rp = res["rp"]
        m  = res["m"]
        trades = res["trades"]
        print(f"\n{'=' * 72}")
        print(f"  {label}  (risk_pct={rp*100:.2f}%)")
        print(f"{'=' * 72}")
        _print_full(m, initial_equity, trades, label)
        monthly = _monthly_returns(trades, initial_equity)
        _print_monthly(monthly)

    # ---- Recommendation ----
    print(f"\n{'=' * 72}")
    print("  RECOMMENDATION")
    print(f"{'=' * 72}")

    # Best overall: highest Calmar with Sharpe > 5
    candidates = (
        [(r, "Fixed") for r in fixed_results if r["m"]["sharpe"] > 5] +
        [(r, "Dynamic") for r in dyn_results  if r["m"]["sharpe"] > 5]
    )
    if candidates:
        best, model = max(candidates, key=lambda x: x[0]["m"]["calmar"])
        m = best["m"]
        print(f"\n  Optimal configuration (best Calmar with Sharpe > 5):")
        print(f"    Model      : {model} cap")
        print(f"    Risk/trade : {best['rp']*100:.2f}% of total portfolio")
        print(f"    CAGR       : {m['cagr']:+.2f}%")
        print(f"    Sharpe     : {m['sharpe']:.2f}")
        print(f"    Calmar     : {m['calmar']:.2f}")
        print(f"    Max DD     : {m['max_dd']:+.2f}%")
        print(f"    Worst strk : {m['worst_streak']} losses / ${m['worst_streak_pnl']:,.0f}")
        per_trade_risk = initial_equity * best["rp"]
        print(f"\n  In dollar terms on ${initial_equity:,.0f} portfolio:")
        print(f"    Max risk per trade : ${per_trade_risk:,.0f}")
        print(f"    (position sized so a 1-ATR adverse move = ${per_trade_risk:,.0f} loss)")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
