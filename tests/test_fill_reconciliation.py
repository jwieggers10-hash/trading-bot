"""Tests for the fill-reconciliation fix (partial fill wrongly recorded as final).

Root cause (production, live account, GLD entry 2026-08-18): the strategy
correctly sized and submitted a 68-share market order. _wait_for_fill()
polled for 10 seconds, gave up while the order was still in-flight, and the
old code trusted whatever filled_qty the last poll happened to see (4) as
final. Alpaca kept filling the order in the background and it eventually
reached 68 shares — but the bot's own bookkeeping stayed frozen at 4,
silently under-counting locked capital for every symbol sized afterward.

The fix: _wait_for_fill() now returns (order, settled), where settled is
only True on an observed terminal status. When settled is False,
_submit_protected_entry() reconciles directly against the live broker
position (Portfolio.reconcile_position_with_broker) instead of trusting the
in-flight snapshot — and records nothing locally if the broker doesn't show
a position yet, rather than recording a fabricated small quantity.

Test structure:
  TestUnsettledOrderReconciliation — _submit_protected_entry()'s timeout path
  TestPartialThenFullFill          — a partial-fill snapshot mid-poll must not
                                      be recorded if the order goes on to settle
  TestReconcileAndCorrect          — Portfolio.reconcile_and_correct() unit tests
  TestNoDuplicateSubmission        — reconciliation never calls submit_order
"""
import itertools
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.portfolio import Portfolio
from bot.strategies.mean_reversion import MeanReversionStrategy, _wait_for_fill
from bot.strategies.mean_reversion_v2 import MeanReversionV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    id: str = "mkt-1",
    filled_avg_price: str | None = "398.59",
    filled_qty: str | None = "4",
    status: str = "partially_filled",
    stop_leg_id: str = "stop-1",
    stop_side: str = "sell",
):
    leg = MagicMock()
    leg.id = stop_leg_id
    leg.side = stop_side
    leg.type = "stop"
    o = MagicMock()
    o.id = id
    o.filled_avg_price = filled_avg_price
    o.filled_qty = filled_qty
    o.limit_price = None
    o.status = status
    o.legs = [leg]
    return o


def _make_position(symbol: str, qty: float, avg_entry_price: float):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.avg_entry_price = str(avg_entry_price)
    return pos


def _make_df_bars(n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 400.0 + rng.normal(0, 0.5, n).cumsum()
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
    mock = MagicMock()
    mock.list_positions.return_value = []
    mock.list_orders.return_value = []
    return mock


@pytest.fixture
def portfolio(api, tmp_path):
    with patch.object(Portfolio, "_init_log_files"):
        p = Portfolio(api, state_file=str(tmp_path / "state.json"))
    return p


@pytest.fixture
def rm():
    mock = MagicMock()
    mock.get_account_equity.return_value = 109_889.0
    mock.calculate_atr.return_value = 0.4671
    mock.integer_position_size.return_value = 68
    mock.stop_price.return_value = 397.9329
    return mock


@pytest.fixture
def strategy(api, rm, portfolio):
    return MeanReversionStrategy(api, rm, portfolio)


@pytest.fixture
def rm_v2():
    mock = MagicMock()
    mock.get_account_equity.return_value = 109_889.0
    mock.calculate_atr.return_value = 0.4671
    mock.v2_position_size.return_value = 68
    mock.stop_price.return_value = 397.9329
    return mock


@pytest.fixture
def strategy_v2(api, rm_v2, portfolio):
    return MeanReversionV2(api, rm_v2, portfolio)


# ---------------------------------------------------------------------------
# TestUnsettledOrderReconciliation — the exact GLD scenario
# ---------------------------------------------------------------------------

class TestUnsettledOrderReconciliation:
    @patch("time.sleep")
    @patch("time.monotonic")
    def test_slow_fill_beyond_ten_seconds_reconciles_to_broker_qty(
        self, mock_monotonic, mock_sleep, strategy, api, portfolio,
    ):
        """The order never reaches a terminal status inside the 10s poll
        window (every get_order call still shows partially_filled qty=4,
        exactly like the production incident). After timeout, the entry must
        be recorded from the live broker position (68 @ 408.78), not frozen
        at the stale in-flight snapshot (4 @ 398.59)."""
        mock_monotonic.side_effect = itertools.count(0, 3)  # forces timeout after a few polls

        submitted = _make_order(id="mkt-gld", status="new", filled_qty=None, filled_avg_price=None)
        still_inflight = _make_order(id="mkt-gld", status="partially_filled", filled_qty="4",
                                      filled_avg_price="398.59")
        api.submit_order.return_value = submitted
        api.get_order.return_value = still_inflight
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        strategy._enter("GLD", "buy", "long", 68, 397.9329)

        assert portfolio.entry_sizes.get("GLD") == 68
        assert portfolio.entry_prices.get("GLD") == pytest.approx(408.78317)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_no_broker_position_yet_records_nothing(
        self, mock_monotonic, mock_sleep, strategy, api, portfolio,
    ):
        """If the order is still unsettled after timeout AND the broker shows
        no position at all yet (order genuinely hasn't filled), the entry
        must not be recorded with a fabricated quantity."""
        mock_monotonic.side_effect = itertools.count(0, 3)

        submitted = _make_order(id="mkt-gld", status="new", filled_qty=None, filled_avg_price=None)
        still_pending = _make_order(id="mkt-gld", status="new", filled_qty=None, filled_avg_price=None)
        api.submit_order.return_value = submitted
        api.get_order.return_value = still_pending
        api.list_positions.return_value = []  # broker shows nothing yet

        strategy._enter("GLD", "buy", "long", 68, 397.9329)

        assert "GLD" not in portfolio.entry_prices
        assert "GLD" not in portfolio.entry_sizes

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_v2_slow_fill_reconciles_to_broker_qty(
        self, mock_monotonic, mock_sleep, strategy_v2, api, portfolio,
    ):
        """Same guarantee for MeanReversionV2 (this is the actual strategy
        that hit the production bug)."""
        mock_monotonic.side_effect = itertools.count(0, 3)

        submitted = _make_order(id="mkt-gld-v2", status="accepted", filled_qty=None, filled_avg_price=None)
        still_inflight = _make_order(id="mkt-gld-v2", status="partially_filled", filled_qty="4",
                                      filled_avg_price="398.59")
        api.submit_order.return_value = submitted
        api.get_order.return_value = still_inflight
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        strategy_v2._enter("GLD", "buy", "long", 68, 397.9329)

        assert portfolio.entry_sizes.get("GLD") == 68
        assert portfolio.entry_prices.get("GLD") == pytest.approx(408.78317)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_reconciled_short_position_uses_broker_direction(
        self, mock_monotonic, mock_sleep, strategy, api, portfolio,
    ):
        """Reconciliation must work for shorts too — broker qty is negative,
        and the recovered size must still be reported as a positive quantity."""
        mock_monotonic.side_effect = itertools.count(0, 3)

        submitted = _make_order(id="mkt-short", status="new", filled_qty=None,
                                 filled_avg_price=None, stop_side="buy")
        still_inflight = _make_order(id="mkt-short", status="partially_filled", filled_qty="10",
                                      filled_avg_price="405.48", stop_side="buy")
        api.submit_order.return_value = submitted
        api.get_order.return_value = still_inflight
        api.list_positions.return_value = [_make_position("GLD", -67, 405.4797)]

        strategy._enter("GLD", "sell", "short", 67, 406.0482)

        assert portfolio.entry_sizes.get("GLD") == 67
        assert portfolio.entry_prices.get("GLD") == pytest.approx(405.4797)
        assert portfolio.entry_directions.get("GLD") == "short"


# ---------------------------------------------------------------------------
# TestPartialThenFullFill — a mid-poll partial snapshot is not final if the
# order goes on to settle within the timeout window
# ---------------------------------------------------------------------------

class TestPartialThenFullFill:
    @patch("time.sleep")
    def test_partial_snapshot_then_full_fill_records_full_qty(self, mock_sleep, strategy, api, portfolio):
        """First poll shows partially_filled qty=4 (non-terminal, so the loop
        keeps polling); the next poll shows filled qty=68. The final recorded
        size must be 68, proving an intermediate partial snapshot is never
        treated as the final answer while the order is still live."""
        submitted = _make_order(id="mkt-gld", status="new", filled_qty=None, filled_avg_price=None)
        partial = _make_order(id="mkt-gld", status="partially_filled", filled_qty="4",
                               filled_avg_price="398.59")
        full = _make_order(id="mkt-gld", status="filled", filled_qty="68",
                            filled_avg_price="408.74")
        api.submit_order.return_value = submitted
        api.get_order.side_effect = [partial, full]

        # Portfolio construction itself calls list_positions() once for startup
        # reconciliation — reset here so the assertion below is scoped to _enter().
        api.list_positions.reset_mock()

        strategy._enter("GLD", "buy", "long", 68, 397.9329)

        assert portfolio.entry_sizes.get("GLD") == 68
        assert portfolio.entry_prices.get("GLD") == pytest.approx(408.74)
        # Broker position query must never have been needed — the order settled on its own.
        api.list_positions.assert_not_called()

    def test_wait_for_fill_settled_true_only_on_terminal_status(self):
        """Direct unit check on _wait_for_fill's return contract."""
        api = MagicMock()
        terminal = _make_order(status="filled", filled_qty="68", filled_avg_price="408.74")

        order, settled = _wait_for_fill(terminal, api, "GLD", timeout=10.0)

        assert settled is True
        assert order.status == "filled"

    @patch("time.sleep")
    def test_wait_for_fill_settled_false_on_timeout(self, mock_sleep):
        """Direct unit check: a non-terminal status that never resolves
        within the timeout must report settled=False, not True."""
        api = MagicMock()
        never_settles = _make_order(status="partially_filled", filled_qty="4",
                                     filled_avg_price="398.59")
        api.get_order.return_value = never_settles

        with patch("time.monotonic", side_effect=itertools.count(0, 3)):
            order, settled = _wait_for_fill(never_settles, api, "GLD", timeout=10.0)

        assert settled is False
        assert order.status == "partially_filled"


# ---------------------------------------------------------------------------
# TestReconcileAndCorrect — Portfolio.reconcile_and_correct() unit tests
# ---------------------------------------------------------------------------

class TestReconcileAndCorrect:
    def test_local_qty_smaller_than_broker_qty_is_corrected(self, portfolio, api):
        portfolio.record_entry("GLD", "long", 398.59, 4, "stop-gld", 397.9329)
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        corrected = portfolio.reconcile_and_correct("GLD")

        assert corrected is True
        assert portfolio.entry_sizes["GLD"] == pytest.approx(68)
        assert portfolio.entry_prices["GLD"] == pytest.approx(408.78317)

    def test_local_price_differing_from_broker_price_is_corrected(self, portfolio, api):
        portfolio.record_entry("GLD", "long", 398.59, 68, "stop-gld", 397.9329)
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        corrected = portfolio.reconcile_and_correct("GLD")

        assert corrected is True
        assert portfolio.entry_prices["GLD"] == pytest.approx(408.78317)
        assert portfolio.entry_sizes["GLD"] == pytest.approx(68)

    def test_matching_state_reports_no_correction(self, portfolio, api):
        portfolio.record_entry("GLD", "long", 408.78317, 68, "stop-gld", 397.9329)
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        corrected = portfolio.reconcile_and_correct("GLD")

        assert corrected is False

    def test_short_position_direction_reconciled_correctly(self, portfolio, api):
        portfolio.record_entry("GLD", "short", 405.0, 10, "stop-gld", 406.0)
        api.list_positions.return_value = [_make_position("GLD", -67, 405.4797)]

        corrected = portfolio.reconcile_and_correct("GLD")

        assert corrected is True
        assert portfolio.entry_sizes["GLD"] == pytest.approx(67)
        assert portfolio.entry_directions["GLD"] == "short"

    def test_no_broker_position_returns_false_without_clearing_local_state(self, portfolio, api):
        """reconcile_and_correct is not responsible for external-exit
        recovery (that's Portfolio.recover_external_exit) — if the broker
        shows nothing, it must simply report no correction, not wipe state."""
        portfolio.record_entry("GLD", "long", 408.78317, 68, "stop-gld", 397.9329)
        api.list_positions.return_value = []

        corrected = portfolio.reconcile_and_correct("GLD")

        assert corrected is False
        assert portfolio.entry_prices["GLD"] == pytest.approx(408.78317)

    def test_correction_persists_to_disk(self, portfolio, api, tmp_path):
        import json
        portfolio.record_entry("GLD", "long", 398.59, 4, "stop-gld", 397.9329)
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        portfolio.reconcile_and_correct("GLD")

        data = json.loads((tmp_path / "state.json").read_text())
        assert data["positions"]["GLD"]["entry_size"] == pytest.approx(68)
        assert data["positions"]["GLD"]["entry_price"] == pytest.approx(408.78317)


# ---------------------------------------------------------------------------
# TestNoDuplicateSubmission — reconciliation paths never place a new order
# ---------------------------------------------------------------------------

class TestNoDuplicateSubmission:
    @patch("time.sleep")
    @patch("time.monotonic")
    def test_only_one_submit_order_call_through_reconciliation_path(
        self, mock_monotonic, mock_sleep, strategy, api, portfolio,
    ):
        mock_monotonic.side_effect = itertools.count(0, 3)

        submitted = _make_order(id="mkt-gld", status="new", filled_qty=None, filled_avg_price=None)
        still_inflight = _make_order(id="mkt-gld", status="partially_filled", filled_qty="4",
                                      filled_avg_price="398.59")
        api.submit_order.return_value = submitted
        api.get_order.return_value = still_inflight
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        strategy._enter("GLD", "buy", "long", 68, 397.9329)

        assert api.submit_order.call_count == 1

    def test_reconcile_position_with_broker_never_submits_orders(self, portfolio, api):
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        portfolio.reconcile_position_with_broker("GLD")

        api.submit_order.assert_not_called()

    def test_reconcile_and_correct_never_submits_orders(self, portfolio, api):
        portfolio.record_entry("GLD", "long", 398.59, 4, "stop-gld", 397.9329)
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]

        portfolio.reconcile_and_correct("GLD")

        api.submit_order.assert_not_called()
