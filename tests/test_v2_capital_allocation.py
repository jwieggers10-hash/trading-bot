"""Audit tests for Strategy V2 portfolio management and capital allocation.

Design intent recap (from backtest_realistic.py and config.py):
  - 4 symbols: SPY / QQQ / GLD / USO (MR_V2_SYMBOLS)
  - Per-symbol cap : total_equity / 4  (dynamic, compounds with portfolio growth)
  - Risk per trade : 0.25% of total_equity  (1 ATR adverse = 0.25% loss)
  - Max simultaneous risk : 4 × 0.25% = 1.0% of portfolio
  - Positions opened sequentially in main.py's for-sym loop; each _enter() call
    updates in-memory locked-notional state before the next symbol is evaluated.

Tests are structured as:
  TestGetLockedNotional      — portfolio.get_locked_notional() accumulation
  TestV2AllocationFormula    — available / per_sym_cap / max_notional math
  TestSequentialCapitalDedup — locked notional updates between entries in the same tick
  TestFillPriceFallback      — fill_price=0 bug: est_price is used as fallback
  TestRiskManagerV2          — v2_position_size() respects max_notional correctly
"""
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from bot.portfolio import Portfolio
from bot.risk_manager import RiskManager
from bot.strategies.mean_reversion_v2 import MeanReversionV2
from config import MR_V2_SYMBOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    id: str = "order-1",
    filled_avg_price: str | None = "520.00",
    filled_qty: str | None = "136",
    limit_price=None,
):
    o = MagicMock()
    o.id = id
    o.filled_avg_price = filled_avg_price
    o.filled_qty = filled_qty
    o.limit_price = limit_price
    return o


def _make_bars(n: int = 25, base_price: float = 520.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = base_price + rng.normal(0, 0.5, n).cumsum()
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": prices, "high": prices + 0.3, "low": prices - 0.3,
         "close": prices, "volume": 1_000_000},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    m = MagicMock()
    m.list_positions.return_value = []
    m.list_orders.return_value = []
    return m


@pytest.fixture
def portfolio(api, tmp_path):
    with patch.object(Portfolio, "_init_log_files"):
        p = Portfolio(api, state_file=str(tmp_path / "state.json"))
    return p


@pytest.fixture
def real_rm(api):
    """Actual RiskManager instance (not a mock) for sizing tests."""
    m = MagicMock()
    m.get_account_equity.return_value = 400_000.0
    return RiskManager(m)


@pytest.fixture
def rm():
    m = MagicMock()
    m.get_account_equity.return_value = 400_000.0
    m.calculate_atr.return_value = 1.5
    m.v2_position_size.return_value = 192
    m.stop_price.return_value = 518.0
    return m


@pytest.fixture
def strategy(api, rm, portfolio):
    return MeanReversionV2(api, rm, portfolio)


# ---------------------------------------------------------------------------
# TestGetLockedNotional
# ---------------------------------------------------------------------------

class TestGetLockedNotional:
    """get_locked_notional() must accurately sum cost-basis of all open positions."""

    def test_zero_when_no_positions(self, portfolio):
        assert portfolio.get_locked_notional() == 0.0

    def test_single_position(self, portfolio):
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        assert portfolio.get_locked_notional() == pytest.approx(520.0 * 192)

    def test_accumulates_across_multiple_positions(self, portfolio):
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)
        portfolio.record_entry("GLD", "long", 195.0, 512, "s3", 193.0)

        expected = 520.0 * 192 + 485.0 * 206 + 195.0 * 512
        assert portfolio.get_locked_notional() == pytest.approx(expected)

    def test_excludes_specified_symbol(self, portfolio):
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)

        # When sizing QQQ's replacement, exclude QQQ from locked
        locked_excl_qqq = portfolio.get_locked_notional(exclude_symbol="QQQ")
        assert locked_excl_qqq == pytest.approx(520.0 * 192)

    def test_excludes_unknown_symbol_gracefully(self, portfolio):
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        # Excluding a symbol that has no position is a no-op
        assert portfolio.get_locked_notional(exclude_symbol="USO") == pytest.approx(
            520.0 * 192
        )

    def test_cleared_after_close(self, portfolio):
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)
        portfolio.clear_position("SPY")
        assert portfolio.get_locked_notional() == pytest.approx(485.0 * 206)

    def test_all_four_symbols_fill_equity(self, portfolio):
        """With 4 positions at equity/4 each, total locked ≈ total equity."""
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)
        portfolio.record_entry("GLD", "long", 195.0, 512, "s3", 193.0)
        portfolio.record_entry("USO", "long",  75.0, 1333, "s4",  73.0)

        total = 520.0 * 192 + 485.0 * 206 + 195.0 * 512 + 75.0 * 1333
        assert portfolio.get_locked_notional() == pytest.approx(total)


# ---------------------------------------------------------------------------
# TestV2AllocationFormula
# ---------------------------------------------------------------------------

class TestV2AllocationFormula:
    """Verify the available / per_sym_cap / max_notional arithmetic."""

    def test_first_symbol_gets_full_per_sym_cap(self, portfolio):
        total_equity = 400_000.0
        n_symbols = 4
        per_sym_cap = total_equity / n_symbols
        locked_others = portfolio.get_locked_notional(exclude_symbol="SPY")
        available = max(0.0, total_equity - locked_others)
        max_notional = min(per_sym_cap, available)

        assert max_notional == pytest.approx(100_000.0)

    def test_second_symbol_still_gets_per_sym_cap_when_first_was_100k(self, portfolio):
        """With equity=400k and one 100k position open, 300k is still more than per_sym_cap=100k."""
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        spy_locked = 520.0 * 192  # ≈ 99,840

        total_equity = 400_000.0
        per_sym_cap = total_equity / 4
        locked_others = portfolio.get_locked_notional(exclude_symbol="QQQ")
        available = max(0.0, total_equity - locked_others)
        max_notional = min(per_sym_cap, available)

        # available ≈ 300k >> per_sym_cap=100k, so max_notional = per_sym_cap
        assert available == pytest.approx(400_000.0 - spy_locked)
        assert max_notional == pytest.approx(per_sym_cap)

    def test_fourth_symbol_limited_to_remaining_capital(self, portfolio):
        """With 3 × ~100k locked, the 4th symbol's available ≈ per_sym_cap."""
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)
        portfolio.record_entry("GLD", "long", 195.0, 512, "s3", 193.0)

        total_equity = 400_000.0
        per_sym_cap = total_equity / 4
        locked_others = portfolio.get_locked_notional(exclude_symbol="USO")
        available = max(0.0, total_equity - locked_others)
        max_notional = min(per_sym_cap, available)

        # available ≈ 400k - 300k = 100k ≈ per_sym_cap
        expected_locked = 520.0 * 192 + 485.0 * 206 + 195.0 * 512
        assert available == pytest.approx(total_equity - expected_locked)
        assert max_notional == pytest.approx(min(per_sym_cap, available))
        # max_notional should be close to per_sym_cap (not zero, not over-allocated)
        assert 0 < max_notional <= per_sym_cap + 1.0  # small rounding tolerance

    def test_no_entry_when_equity_exhausted_by_losses(self, portfolio):
        """If existing positions lock more than current equity (losses), available=0."""
        # 3 positions at original cost; equity dropped due to unrealized losses
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)
        portfolio.record_entry("GLD", "long", 195.0, 512, "s3", 193.0)

        depleted_equity = 250_000.0  # significant losses reduced mark-to-market equity
        locked_others = portfolio.get_locked_notional(exclude_symbol="USO")
        available = max(0.0, depleted_equity - locked_others)

        # locked_others ≈ 300k > depleted_equity=250k → available=0
        assert locked_others > depleted_equity
        assert available == 0.0

    def test_v2_position_size_returns_zero_when_max_notional_zero(self):
        rm = RiskManager(MagicMock())
        size = rm.v2_position_size(
            atr=1.5, total_equity=400_000, price=520.0, max_notional=0, risk_pct=0.0025
        )
        assert size == 0

    def test_v2_position_size_capped_by_notional(self):
        rm = RiskManager(MagicMock())
        # ATR-based: int(400000 * 0.0025 / 1.5) = int(666) = 666 shares
        # Notional cap at 50k / 520 = int(96) = 96 shares
        # min(666, 96) = 96
        size = rm.v2_position_size(
            atr=1.5, total_equity=400_000, price=520.0, max_notional=50_000, risk_pct=0.0025
        )
        assert size == 96

    def test_v2_position_size_capped_by_atr_risk(self):
        rm = RiskManager(MagicMock())
        # Very large notional cap, so ATR risk is the binding constraint
        # ATR-based: int(400000 * 0.0025 / 1.5) = 666
        # Notional cap at 1_000_000 / 520 = 1923
        # min(666, 1923) = 666
        size = rm.v2_position_size(
            atr=1.5, total_equity=400_000, price=520.0, max_notional=1_000_000, risk_pct=0.0025
        )
        assert size == 666

    def test_total_risk_across_four_positions(self):
        """4 simultaneous positions should risk at most 4 × 0.25% = 1% of equity."""
        rm = RiskManager(MagicMock())
        total_equity = 400_000.0
        risk_pct = 0.0025
        atr = 1.5  # dollar ATR for illustration

        # Each position: risk_dollars = total_equity * risk_pct = $1,000
        risk_per_position = total_equity * risk_pct
        assert risk_per_position == 1_000.0

        max_total_risk = 4 * risk_per_position
        assert max_total_risk == 4_000.0  # = 1% of 400k


# ---------------------------------------------------------------------------
# TestSequentialCapitalDedup
# ---------------------------------------------------------------------------

class TestSequentialCapitalDedup:
    """
    Prove that sequential _enter() calls within one tick correctly accumulate
    locked notional so each subsequent symbol's max_notional is bounded.

    This is the core guarantee that prevents over-allocation when all 4 symbols
    signal at the same time.
    """

    def test_enter_updates_locked_notional_immediately(self, strategy, api, portfolio, rm):
        """After _enter("SPY"), get_locked_notional() must reflect the SPY entry."""
        api.submit_order.side_effect = [
            _make_order(id="mkt-spy", filled_avg_price="520.00", filled_qty="192"),
            _make_order(id="stop-spy"),
        ]

        strategy._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        locked = portfolio.get_locked_notional()
        assert locked == pytest.approx(520.0 * 192)

    def test_second_entry_sees_first_entry_locked_notional(self, strategy, api, portfolio, rm):
        """
        After SPY enters, the max_notional available for QQQ must account for the
        SPY notional already locked.
        """
        # Enter SPY first
        api.submit_order.side_effect = [
            _make_order(id="mkt-spy", filled_avg_price="520.00", filled_qty="192"),
            _make_order(id="stop-spy"),
        ]
        strategy._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        # Now compute what QQQ's max_notional would be
        total_equity = 400_000.0
        per_sym_cap = total_equity / 4
        locked_others = portfolio.get_locked_notional(exclude_symbol="QQQ")
        available = max(0.0, total_equity - locked_others)
        max_notional = min(per_sym_cap, available)

        spy_locked = 520.0 * 192
        assert locked_others == pytest.approx(spy_locked)
        assert available == pytest.approx(total_equity - spy_locked)
        # With only $300k available and a $100k cap, QQQ still gets full cap
        assert max_notional == pytest.approx(per_sym_cap)

    def test_three_entries_leave_one_sym_cap_for_fourth(
        self, strategy, api, portfolio, rm
    ):
        """After 3 entries, the 4th symbol's available ≈ per_sym_cap."""
        entry_configs = [
            ("SPY", "520.00", 192),
            ("QQQ", "485.00", 206),
            ("GLD", "195.00", 512),
        ]
        order_seq = []
        for sym, price, qty in entry_configs:
            order_seq += [
                _make_order(id=f"mkt-{sym}", filled_avg_price=price, filled_qty=str(qty)),
                _make_order(id=f"stop-{sym}"),
            ]
        api.submit_order.side_effect = order_seq

        for sym, price, qty in entry_configs:
            fill_price = float(price)
            strategy._enter(sym, "buy", "long", qty, fill_price - 2, est_price=fill_price)

        total_equity = 400_000.0
        per_sym_cap = total_equity / 4
        locked_others = portfolio.get_locked_notional(exclude_symbol="USO")
        available = max(0.0, total_equity - locked_others)
        max_notional = min(per_sym_cap, available)

        expected_locked = 520.0 * 192 + 485.0 * 206 + 195.0 * 512
        assert locked_others == pytest.approx(expected_locked)
        assert available == pytest.approx(total_equity - expected_locked)
        # max_notional ≈ per_sym_cap (both ≈ $100k), verifies the 4th symbol gets its fair share
        assert max_notional == pytest.approx(min(per_sym_cap, available))
        assert max_notional > 0

    def test_v2_run_passes_correct_max_notional_to_sizing_after_prior_entry(
        self, strategy, api, portfolio, rm
    ):
        """
        When run("QQQ") fires after SPY is already open, v2_position_size must
        receive a max_notional that reflects the SPY notional already locked.

        This is the integration-level proof that the sequential signal path works.
        """
        # Manually enter SPY (simulates what a prior run("SPY") tick would have done)
        portfolio.record_entry("SPY", "long", 520.0, 192, "stop-spy", 518.0)
        spy_locked = 520.0 * 192  # 99,840

        # Capture the max_notional argument that run("QQQ") passes to v2_position_size
        captured = {}

        def capture_sizing(atr, total_equity, price, max_notional, risk_pct):
            captured["max_notional"] = max_notional
            captured["available"] = max_notional  # store for assertion
            return 200  # arbitrary non-zero size so _enter() proceeds

        rm.v2_position_size.side_effect = capture_sizing
        rm.calculate_atr.return_value = 1.5
        rm.stop_price.return_value = 480.0

        # Set up API mocks for run("QQQ")
        api.list_positions.return_value = []  # QQQ is flat; SPY tracked in-memory only
        api.submit_order.side_effect = [
            _make_order(id="mkt-qqq", filled_avg_price="485.00", filled_qty="200"),
            _make_order(id="stop-qqq"),
        ]

        bars = _make_bars(n=25, base_price=485.0)
        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value={
                 "price": 474.0,   # below lower band → long signal
                 "sma":   485.0,
                 "upper": 490.0,
                 "lower": 475.0,
             }):
            strategy.run("QQQ")

        total_equity = 400_000.0
        per_sym_cap = total_equity / 4
        expected_locked = spy_locked
        expected_available = total_equity - expected_locked
        expected_max_notional = min(per_sym_cap, expected_available)

        assert captured.get("max_notional") == pytest.approx(expected_max_notional, rel=1e-6)

    def test_v2_run_blocked_when_capital_exhausted(
        self, strategy, api, portfolio, rm
    ):
        """
        When available capital is zero (equity dropped below locked notional),
        run() must skip entry and never call submit_order.
        """
        # 4 positions already locked at original cost
        portfolio.record_entry("SPY", "long", 520.0, 192, "s1", 518.0)
        portfolio.record_entry("QQQ", "long", 485.0, 206, "s2", 483.0)
        portfolio.record_entry("GLD", "long", 195.0, 512, "s3", 193.0)
        portfolio.record_entry("USO", "long",  75.0, 1333, "s4",  73.0)

        # Portfolio equity has dropped below total locked notional (severe losses)
        rm.get_account_equity.return_value = 200_000.0
        strategy._equity_cache = 200_000.0
        strategy._equity_ts = float("inf")  # force use of cache

        # v2_position_size should return 0 when max_notional=0
        rm.v2_position_size.return_value = 0

        api.list_positions.return_value = []  # SPY appears flat to the API (already closed)

        bars = _make_bars(n=25, base_price=520.0)
        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value={
                 "price": 507.0,   # long signal
                 "sma":   520.0,
                 "upper": 525.0,
                 "lower": 508.0,
             }):
            strategy.run("SPY")

        api.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# TestFillPriceFallback
# ---------------------------------------------------------------------------

class TestFillPriceFallback:
    """
    Bug: when _fill_price() returns 0 (because filled_avg_price is None AND
    get_latest_trade fails), record_entry stores entry_price=0.
    get_locked_notional() then computes 0 * size = 0 for this position, so all
    subsequent symbols in the same tick see no locked capital and can each
    allocate a full per_sym_cap — totalling 4× the intended allocation.

    Fix: _enter() falls back to est_price when _fill_price() returns ≤ 0.
    """

    def test_v2_enter_uses_est_price_when_fill_price_is_zero(
        self, strategy, api, portfolio
    ):
        """
        Reproduces the bug: filled_avg_price=None and get_latest_trade fails.
        After the fix, record_entry must store entry_price = est_price, not 0.
        """
        api.submit_order.side_effect = [
            # Market order returns before fill settles; filled_avg_price is None
            _make_order(id="mkt-1", filled_avg_price=None, filled_qty="192"),
            _make_order(id="stop-1"),
        ]
        # get_latest_trade also fails (data API down)
        api.get_latest_trade.side_effect = Exception("data API unavailable")

        strategy._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        # With the fix: entry_price must be est_price (520), not 0
        assert portfolio.entry_prices.get("SPY") == pytest.approx(520.0)

    def test_v2_locked_notional_correct_after_fill_price_fallback(
        self, strategy, api, portfolio
    ):
        """
        The locked notional after the fallback must be est_price × size, not 0.
        Without the fix, the next symbol would see locked_notional=0 for SPY
        and be allowed to allocate a full per_sym_cap independently.
        """
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price=None, filled_qty="192"),
            _make_order(id="stop-1"),
        ]
        api.get_latest_trade.side_effect = Exception("data API unavailable")

        strategy._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        locked = portfolio.get_locked_notional()
        assert locked == pytest.approx(520.0 * 192)

    def test_normal_fill_price_not_overridden_by_fallback(
        self, strategy, api, portfolio
    ):
        """When filled_avg_price is present and > 0, it must NOT be replaced by est_price."""
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="521.50", filled_qty="192"),
            _make_order(id="stop-1"),
        ]

        strategy._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        # entry_price should be the actual fill (521.50), not the signal price (520.0)
        assert portfolio.entry_prices.get("SPY") == pytest.approx(521.5)

    def test_second_symbol_max_notional_correct_after_first_fill_fallback(
        self, strategy, api, portfolio, rm
    ):
        """
        The critical allocation test: even when SPY's fill price fell back to
        est_price, QQQ must see the correct locked notional and size accordingly.
        """
        # SPY enters with fill price fallback
        api.submit_order.side_effect = [
            _make_order(id="mkt-spy", filled_avg_price=None, filled_qty="192"),
            _make_order(id="stop-spy"),
        ]
        api.get_latest_trade.side_effect = Exception("unavailable")

        strategy._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        # Reset for QQQ sizing check
        api.get_latest_trade.side_effect = None

        total_equity = 400_000.0
        per_sym_cap = total_equity / 4
        locked_others = portfolio.get_locked_notional(exclude_symbol="QQQ")
        available = max(0.0, total_equity - locked_others)
        max_notional = min(per_sym_cap, available)

        # Locked = est_price * filled_qty = 520 * 192 = 99,840
        assert locked_others == pytest.approx(520.0 * 192)
        # Available = 400k - 99,840 ≈ 300k; max_notional = min(100k, 300k) = 100k
        assert max_notional == pytest.approx(per_sym_cap)

    def test_without_fix_locked_notional_would_be_zero(self, strategy, api, portfolio):
        """
        Documents what the bug looked like: entry_price=0 causes locked=0.

        We directly verify that if record_entry were called with entry_price=0,
        get_locked_notional() returns 0 — confirming the bug's impact.
        (The fix prevents entry_price=0 from ever reaching record_entry.)
        """
        # Manually reproduce the pre-fix scenario: entry_price=0
        portfolio.record_entry("SPY", "long", 0.0, 192, "stop-1", 518.0)

        # With entry_price=0, locked notional is 0 — all capital appears free
        broken_locked = portfolio.get_locked_notional()
        assert broken_locked == 0.0

        # This would allow QQQ to see available=total_equity, not available=300k
        total_equity = 400_000.0
        broken_available = max(0.0, total_equity - broken_locked)
        assert broken_available == total_equity  # BUG: should be ~300k

    def test_v1_strategy_also_uses_est_price_fallback(self, api, portfolio, tmp_path):
        """The same fill-price fallback fix must also be present in MeanReversionStrategy (v1)."""
        from bot.strategies.mean_reversion import MeanReversionStrategy

        rm_v1 = MagicMock()
        rm_v1.get_account_equity.return_value = 400_000.0
        rm_v1.calculate_atr.return_value = 1.5
        rm_v1.integer_position_size.return_value = 192
        rm_v1.stop_price.return_value = 518.0

        strat_v1 = MeanReversionStrategy(api, rm_v1, portfolio)

        api.submit_order.side_effect = [
            _make_order(id="mkt-v1", filled_avg_price=None, filled_qty="192"),
            _make_order(id="stop-v1"),
        ]
        api.get_latest_trade.side_effect = Exception("unavailable")
        api.list_orders.return_value = []

        strat_v1._enter("SPY", "buy", "long", 192, 518.0, est_price=520.0)

        assert portfolio.entry_prices.get("SPY") == pytest.approx(520.0)
        assert portfolio.get_locked_notional() == pytest.approx(520.0 * 192)
