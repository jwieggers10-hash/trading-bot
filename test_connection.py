"""Verify Alpaca paper trading connectivity.

Usage:
    python test_connection.py
"""
import sys
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

load_dotenv()

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from bot import alpaca_client as tradeapi

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, fn):
    try:
        result = fn()
        print(f"{PASS} {label}")
        return result
    except Exception as exc:
        print(f"{FAIL} {label}: {exc}")
        return None


def main():
    print(f"Base URL : {ALPACA_BASE_URL}")
    print(f"API key  : {ALPACA_API_KEY[:8]}..." if ALPACA_API_KEY else "API key  : (not set)")
    print()

    # 1. Credentials present
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print(f"{FAIL} Credentials: ALPACA_API_KEY or ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)
    print(f"{PASS} Credentials: both keys loaded from .env")

    # 2. Client construction
    api = check("Client init", lambda: tradeapi.REST(
        ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    ))
    if api is None:
        sys.exit(1)

    # 3. Account info
    account = check("Get account", api.get_account)
    if account:
        print(f"       account id : {account.id}")
        print(f"       status     : {account.status}")
        print(f"       equity     : ${float(account.equity):,.2f}")
        print(f"       cash       : ${float(account.cash):,.2f}")

    # 4. Market clock
    clock = check("Get clock", api.get_clock)
    if clock:
        print(f"       market open: {clock.is_open}")
        print(f"       next open  : {clock.next_open}")
        print(f"       next close : {clock.next_close}")

    # 5. List positions
    positions = check("List positions", api.list_positions)
    if positions is not None:
        print(f"       open positions: {len(positions)}")
        for pos in positions:
            print(f"         {pos.symbol:10s} qty={pos.qty:>8}  unrealized P&L=${float(pos.unrealized_pl):+,.2f}")

    # 6. Equity bar fetch (SPY, last 5 days)
    def fetch_spy_bars():
        start = datetime.now(timezone.utc) - timedelta(days=7)
        bars = api.get_bars("SPY", "1Day", start=start)
        df = bars.df
        return df

    df = check("Fetch SPY daily bars (last 7 days)", fetch_spy_bars)
    if df is not None and not df.empty:
        print(f"       rows returned: {len(df)}")
        print(f"       latest close : ${df['close'].iloc[-1]:,.2f}")

    # 7. Crypto bar fetch (BTC/USD, last 3 days)
    def fetch_btc_bars():
        start = datetime.now(timezone.utc) - timedelta(days=3)
        bars = api.get_crypto_bars("BTC/USD", "1Hour", start=start)
        df = bars.df
        return df

    df = check("Fetch BTC/USD hourly bars (last 3 days)", fetch_btc_bars)
    if df is not None and not df.empty:
        print(f"       rows returned: {len(df)}")
        print(f"       latest close : ${df['close'].iloc[-1]:,.2f}")

    print()
    print("Connection test complete.")


if __name__ == "__main__":
    main()
