"""
Dry-run verification for the SPY/QQQ mean reversion bot.

Fetches real bar data from Alpaca, runs signal and sizing logic end-to-end,
but intercepts every order mutation so nothing is actually traded or modified.

Checks:
  1. Credentials point to the paper trading endpoint (not live)
  2. Only SPY and QQQ mean reversion is active; BTC/trend calls removed from main()
  3. Position sizing cap of $100,000 per symbol is enforced
  4. Signal computation runs end-to-end against real bar data
  5. All intercepted orders are within the allowed symbol set

Usage:
    python -m bot.dry_run
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from bot import alpaca_client as tradeapi
from bot.portfolio import Portfolio
from bot.risk_manager import RiskManager
from bot.strategies.mean_reversion import MeanReversionStrategy
from config import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
    CAPITAL_PER_SYMBOL,
    MEAN_REVERSION_SYMBOLS,
    MOMENTUM_BREAKOUT_SYMBOLS,
    RISK_PER_TRADE,
    TREND_FOLLOWING_SYMBOLS,
)


# ---------------------------------------------------------------------------
# Logging — DEBUG level so every signal decision is visible
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dry_run")


# ---------------------------------------------------------------------------
# Mock order object
# ---------------------------------------------------------------------------

class _MockOrder:
    """Minimal stand-in for the alpaca-py Order object returned by submit_order."""
    _counter = 0

    def __init__(self, symbol: str, price: float):
        _MockOrder._counter += 1
        self.filled_avg_price = str(price)
        self.limit_price = None
        self.id = f"dry-run-{symbol.replace('/', '')}-{_MockOrder._counter:04d}"


# ---------------------------------------------------------------------------
# Intercepting API proxy
# ---------------------------------------------------------------------------

class _DryRunAPI:
    """
    Proxies all read-only calls to the real Alpaca REST client unchanged.
    Intercepts the three mutating calls (submit_order, close_position,
    cancel_order) so no real orders are ever sent.
    """

    def __init__(self, real_api: tradeapi.REST):
        self._real = real_api
        self.intercepted: list[dict] = []

    # -- intercepted mutations ------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty,
        side: str,
        type: str,
        time_in_force: str,
        stop_price=None,
    ) -> _MockOrder:
        price = self._latest_price(symbol)
        qty_f = float(qty)
        notional = qty_f * price
        record = {
            "action": "submit_order",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "tif": time_in_force,
            "stop_price": stop_price,
            "fill_price": price,
            "notional": notional,
        }
        self.intercepted.append(record)
        log.info(
            "[DRY-RUN] WOULD SUBMIT: %-4s %-6s  qty=%-10s  type=%-6s  stop=%-10s  "
            "fill~$%,.4f  notional~$%,.2f",
            side.upper(), symbol, qty, type,
            stop_price if stop_price is not None else "--",
            price, notional,
        )
        return _MockOrder(symbol, price)

    def close_position(self, symbol: str) -> None:
        self.intercepted.append({"action": "close_position", "symbol": symbol})
        log.info("[DRY-RUN] WOULD CLOSE POSITION: %s", symbol)

    def cancel_order(self, order_id: str) -> None:
        log.debug("[DRY-RUN] WOULD CANCEL ORDER: %s", order_id)

    # -- pass-through to real API ---------------------------------------

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def _latest_price(self, symbol: str) -> float:
        try:
            return float(self._real.get_latest_trade(symbol).price)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Checklist helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool]] = []


def _check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "[PASS]" if ok else "[FAIL]"
    line = f"    {tag} {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    _results.append((label, ok))
    return ok


def _info(label: str, value: str) -> None:
    print(f"    [INFO] {label}: {value}")


def _section(title: str) -> None:
    print(f"\n{'-' * 64}")
    print(f"  {title}")
    print(f"{'-' * 64}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    now_utc = datetime.now(timezone.utc)
    print(f"\n{'=' * 64}")
    print("  TRADING BOT DRY-RUN VERIFICATION")
    print(f"  {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'=' * 64}")

    # -- 1. Credentials & connectivity ---------------------------------
    _section("1. Credentials & connectivity")

    _check("ALPACA_API_KEY is set", bool(ALPACA_API_KEY))
    _check("ALPACA_SECRET_KEY is set", bool(ALPACA_SECRET_KEY))
    _check(
        "Base URL points to paper trading endpoint",
        "paper" in (ALPACA_BASE_URL or "").lower(),
        detail=ALPACA_BASE_URL or "(not set)",
    )

    try:
        real_api = tradeapi.REST(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2"
        )
        account = real_api.get_account()
        _check("Alpaca API reachable", True)
        _info("Account ID", str(account.id))
        _info("Account status", str(account.status))
        _info("Equity", f"${float(account.equity):,.2f}")
        _info("Cash", f"${float(account.cash):,.2f}")
        _info("Buying power", f"${float(account.buying_power):,.2f}")
    except Exception as exc:
        _check("Alpaca API reachable", False, detail=str(exc))
        print("\n  Cannot continue without API access.")
        sys.exit(1)

    # -- 2. Active strategy configuration ------------------------------
    _section("2. Active strategy / symbol configuration")

    _check(
        "Mean reversion symbols are SPY and QQQ only",
        set(MEAN_REVERSION_SYMBOLS) == {"SPY", "QQQ"},
        detail=repr(MEAN_REVERSION_SYMBOLS),
    )

    # Inspect main() source at runtime to confirm disabled strategies
    # are not wired up — catches regressions where someone re-enables them
    try:
        import bot.main as main_mod
        src = inspect.getsource(main_mod.main)
        _check(
            "main() does not call momentum.run()  [BTC disabled]",
            "momentum.run" not in src,
        )
        _check(
            "main() does not call trend.run()  [GLD/USO disabled]",
            "trend.run" not in src,
        )
    except Exception as exc:
        _check("main.py source inspection", False, detail=str(exc))

    try:
        clock = real_api.get_clock()
        market_open = clock.is_open
        _info(
            "Market status",
            f"{'OPEN' if market_open else 'CLOSED'}  "
            f"(next open: {clock.next_open})",
        )
    except Exception:
        market_open = False
        _info("Market status", "unknown — proceeding anyway")

    # -- 3. Position sizing cap -----------------------------------------
    _section("3. Position sizing cap  ($100,000 per symbol)")

    _info("CAPITAL_PER_SYMBOL", f"${CAPITAL_PER_SYMBOL:,.0f}")
    _info("RISK_PER_TRADE", f"{RISK_PER_TRADE * 100:.0f}% of equity per 1 ATR move")

    equity = float(account.equity)
    rm = RiskManager(real_api)

    for sym in MEAN_REVERSION_SYMBOLS:
        print(f"\n  [{sym}]")
        try:
            end = now_utc
            start = end - timedelta(hours=8)
            raw_bars = real_api.get_bars(
                sym, "15Min",
                start=start.isoformat(),
                end=end.isoformat(),
                limit=60, feed="iex", adjustment="raw",
            ).df
            if isinstance(raw_bars.index, pd.MultiIndex):
                raw_bars = raw_bars.xs(sym, level=0)

            if len(raw_bars) < 2:
                print(
                    f"    [WARN] Only {len(raw_bars)} bar(s) in last 8 h — "
                    "market closed or pre-market; sizing check skipped"
                )
                continue

            atr = rm.calculate_atr(raw_bars)
            price = float(raw_bars["close"].iloc[-1])
            atr_raw = int((equity * RISK_PER_TRADE) / atr)
            cap_shares = int(CAPITAL_PER_SYMBOL / price)
            final = rm.integer_position_size(atr, equity, price)
            notional = final * price

            _info("Latest 15-min close", f"${price:,.4f}")
            _info("14-period ATR (15-min)", f"${atr:,.4f}")
            _info("Account equity", f"${equity:,.2f}")
            _info(
                "ATR-based raw size",
                f"{atr_raw:,} shares  (~${atr_raw * price:,.0f} notional — pre-cap)",
            )
            _info(
                "Notional cap",
                f"floor($100,000 / ${price:,.4f}) = {cap_shares:,} shares  (~${cap_shares * price:,.0f})",
            )
            _info("Final size  (min of above)", f"{final:,} shares")
            _info("Final notional", f"${notional:,.2f}")

            _check(
                f"[{sym}] notional does not exceed $100,000",
                notional <= CAPITAL_PER_SYMBOL,
                detail=f"${notional:,.2f}",
            )

            if atr_raw > cap_shares:
                _info(
                    "Cap verdict",
                    f"BINDING — ATR formula gave {atr_raw:,}, capped to {final:,} shares "
                    f"(saved ${(atr_raw - final) * price:,.0f} in potential over-exposure)",
                )
            else:
                _info(
                    "Cap verdict",
                    f"not binding — ATR formula gives {final:,} shares (under the limit)",
                )

        except Exception as exc:
            _check(f"[{sym}] sizing calculation", False, detail=str(exc))

    # -- 4. End-to-end signal run ---------------------------------------
    _section("4. End-to-end signal run  (orders intercepted, none submitted)")

    dry_api = _DryRunAPI(real_api)
    # Use an isolated temp file so the dry run never writes to the production state file.
    dry_state = os.path.join(tempfile.gettempdir(), "dry_run_position_state.json")
    portfolio = Portfolio(dry_api, state_file=dry_state)
    strategy = MeanReversionStrategy(dry_api, RiskManager(dry_api), portfolio)

    log.info("Fetching current positions from Alpaca ...")
    try:
        positions = real_api.list_positions()
        if positions:
            print(f"\n  Open positions ({len(positions)}):")
            for p in positions:
                print(f"    {p.symbol:8s}  qty={p.qty:>10}  side={'long' if float(p.qty) > 0 else 'short'}"
                      f"  unrealised P&L ${float(p.unrealized_pl):+,.2f}")
        else:
            print("\n  No open positions.")
    except Exception as exc:
        print(f"\n  [WARN] Could not fetch positions: {exc}")

    print()
    for sym in MEAN_REVERSION_SYMBOLS:
        print(f"  {'-' * 28}  {sym}  {'-' * 28}")
        strategy.run(sym)
        print()

    # -- 5. Intercepted order summary -----------------------------------
    _section("5. Intercepted order summary")

    if not dry_api.intercepted:
        print(
            "  No orders intercepted.\n"
            "  Possible reasons:\n"
            "    • Already in a position on both symbols (exit/hold logic ran instead)\n"
            "    • Price is within the Bollinger bands (no entry signal)\n"
            "    • Market is closed and bars are stale (strategy skipped after bar-count check)"
        )
    else:
        print(f"\n  {len(dry_api.intercepted)} action(s) would have been sent to the broker:\n")
        for i, rec in enumerate(dry_api.intercepted, 1):
            if rec["action"] == "submit_order":
                print(
                    f"  {i:2d}. {rec['side'].upper():4s} {rec['symbol']:6s}"
                    f"  qty={str(rec['qty']):>8}"
                    f"  type={rec['type']:6s}"
                    f"  stop={str(rec.get('stop_price') or '--'):>12}"
                    f"  fill~${rec['fill_price']:>10,.4f}"
                    f"  notional~${rec['notional']:>10,.2f}"
                )
            elif rec["action"] == "close_position":
                print(f"  {i:2d}. CLOSE  {rec['symbol']}")

        # Guard: every intercepted order must be for an allowed symbol
        touched = {
            r["symbol"]
            for r in dry_api.intercepted
            if r.get("action") == "submit_order"
        }
        unexpected = touched - set(MEAN_REVERSION_SYMBOLS)
        _check(
            "All intercepted orders are for SPY or QQQ",
            not unexpected,
            detail=f"unexpected: {unexpected}" if unexpected else "clean",
        )

        # Guard: no order exceeds the per-symbol cap
        for rec in dry_api.intercepted:
            if rec.get("action") != "submit_order":
                continue
            if rec["fill_price"] > 0:
                _check(
                    f"Order notional <= $100,000  [{rec['symbol']} {rec['side']} {rec['qty']}]",
                    rec["notional"] <= CAPITAL_PER_SYMBOL,
                    detail=f"${rec['notional']:,.2f}",
                )

    # -- Summary --------------------------------------------------------
    _section("Summary")

    passed = sum(1 for _, ok in _results if ok)
    failed = sum(1 for _, ok in _results if not ok)
    print(f"  {passed} checks passed  /  {failed} checks failed\n")

    if failed:
        print("  [FAIL] Fix the failures above before running the live bot.")
        sys.exit(1)
    else:
        print("  [PASS] All checks passed — bot is correctly configured for paper trading.")
    print()


if __name__ == "__main__":
    main()
