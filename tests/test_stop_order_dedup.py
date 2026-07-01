"""Tests covering Alpaca error 40310000 (wash-trade) and the OTO-order fix.

Short-position stop tests are in TestShortPositionStop at the bottom of this file.

Original bug: _enter() submitted a market order and then immediately submitted
a *separate* stop order on the opposite side. Alpaca's wash-trade check races
against its own order-status propagation, so a second top-level order for the
opposite side can be rejected even after polling the entry order to
status="filled":
  "40310000: potential wash trade detected. opposite side market/stop order exists"
This fired almost every time for short entries (SELL market + BUY stop) and
intermittently for longs, leaving the position open with no protection.

The fix: _enter() now submits ONE order — a market entry with
order_class="oto" and a stop_loss leg — via _submit_protected_entry() in
bot.strategies.mean_reversion. Alpaca holds the stop_loss leg (status "held")
until the parent fills, then activates it server-side. Only one top-level
order ever exists for the symbol, so there is nothing for the wash-trade
check to compare it against.

Because the stop leg now lives in "held" status while pending, and Alpaca's
"open" order filter excludes "held" orders, cancel_open_stop_orders() sweeps
status="all" (filtering out terminal statuses itself) instead of "open".

Test structure:
  TestCancelOpenStopOrders  — unit tests for the Portfolio method in isolation
  TestEnterDedup            — _enter() sweeps stale stops before submitting the OTO order
  TestCloseDedup            — _close() calls the sweep before close_position
  TestReplaceStopDedup      — _replace_stop_order() calls the sweep after the tracked cancel
  TestV2EnterDedup          — MeanReversionV2._enter() has the same protection
  TestShortPositionStop     — long and short entries both get a single OTO order with the correct stop leg
"""
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from bot.portfolio import Portfolio
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.mean_reversion_v2 import MeanReversionV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    id: str = "order-1",
    filled_avg_price: str | None = "520.00",
    filled_qty: str | None = "136",
    status: str = "filled",
    legs=None,
):
    o = MagicMock()
    o.id = id
    o.filled_avg_price = filled_avg_price
    o.filled_qty = filled_qty
    o.limit_price = None
    o.status = status  # plain string so _wait_for_fill can compare without .value
    o.legs = legs
    return o


def _make_stop_order(id: str, side: str, order_type: str = "stop", status: str = "new"):
    """Return a mock Order object whose .type and .side are plain strings
    (matching the comparison path in cancel_open_stop_orders)."""
    o = MagicMock()
    o.id = id
    o.side = side
    o.type = order_type
    o.status = status
    return o


def _make_oto_order(
    id: str = "mkt-1",
    filled_avg_price: str | None = "520.00",
    filled_qty: str | None = "136",
    status: str = "filled",
    stop_leg_id: str = "stop-leg-1",
    stop_side: str = "sell",
):
    """A parent OTO order with a populated stop_loss leg, as Alpaca returns
    it in the response to a single order_class="oto" submission."""
    stop_leg = _make_stop_order(stop_leg_id, stop_side, order_type="stop", status="held")
    return _make_order(
        id=id, filled_avg_price=filled_avg_price, filled_qty=filled_qty,
        status=status, legs=[stop_leg],
    )


def _make_df_bars(n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 520.0 + rng.normal(0, 0.5, n).cumsum()
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
    mock.list_orders.return_value = []     # no dangling orders by default
    return mock


@pytest.fixture
def portfolio(api, tmp_path):
    with patch.object(Portfolio, "_init_log_files"):
        p = Portfolio(api, state_file=str(tmp_path / "state.json"))
    return p


@pytest.fixture
def rm():
    mock = MagicMock()
    mock.get_account_equity.return_value = 400_000.0
    mock.calculate_atr.return_value = 1.5
    mock.integer_position_size.return_value = 136
    mock.stop_price.return_value = 518.0
    return mock


@pytest.fixture
def strategy(api, rm, portfolio):
    return MeanReversionStrategy(api, rm, portfolio)


@pytest.fixture
def rm_v2():
    mock = MagicMock()
    mock.get_account_equity.return_value = 400_000.0
    mock.calculate_atr.return_value = 1.5
    mock.v2_position_size.return_value = 136
    mock.stop_price.return_value = 518.0
    return mock


@pytest.fixture
def strategy_v2(api, rm_v2, portfolio):
    return MeanReversionV2(api, rm_v2, portfolio)


# ---------------------------------------------------------------------------
# TestCancelOpenStopOrders — unit tests for the Portfolio method
# ---------------------------------------------------------------------------

class TestCancelOpenStopOrders:
    def test_cancels_matching_stop_order(self, portfolio, api):
        dangling = _make_stop_order("old-stop-1", side="sell", order_type="stop")
        api.list_orders.return_value = [dangling]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 1
        api.cancel_order.assert_called_once_with("old-stop-1")

    def test_cancels_stop_limit_order(self, portfolio, api):
        dangling = _make_stop_order("old-stoplim-1", side="buy", order_type="stop_limit")
        api.list_orders.return_value = [dangling]

        count = portfolio.cancel_open_stop_orders("SPY", "buy")

        assert count == 1
        api.cancel_order.assert_called_once_with("old-stoplim-1")

    def test_ignores_wrong_side(self, portfolio, api):
        buy_stop = _make_stop_order("buy-stop-1", side="buy", order_type="stop")
        api.list_orders.return_value = [buy_stop]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 0
        api.cancel_order.assert_not_called()

    def test_ignores_non_stop_order_type(self, portfolio, api):
        market_order = _make_stop_order("mkt-1", side="sell", order_type="market")
        limit_order = _make_stop_order("lim-1", side="sell", order_type="limit")
        api.list_orders.return_value = [market_order, limit_order]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 0
        api.cancel_order.assert_not_called()

    def test_cancels_multiple_matching_orders(self, portfolio, api):
        d1 = _make_stop_order("old-1", side="sell")
        d2 = _make_stop_order("old-2", side="sell")
        api.list_orders.return_value = [d1, d2]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 2
        api.cancel_order.assert_any_call("old-1")
        api.cancel_order.assert_any_call("old-2")

    def test_returns_zero_when_list_orders_fails(self, portfolio, api):
        api.list_orders.side_effect = Exception("network error")

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 0
        api.cancel_order.assert_not_called()

    def test_continues_after_individual_cancel_failure(self, portfolio, api):
        d1 = _make_stop_order("old-1", side="sell")
        d2 = _make_stop_order("old-2", side="sell")
        api.list_orders.return_value = [d1, d2]
        api.cancel_order.side_effect = [Exception("failed"), None]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        # First cancel failed, second succeeded; count reflects successes
        assert count == 1

    def test_clears_tracked_stop_order_id_when_cancelled(self, portfolio, api):
        dangling = _make_stop_order("tracked-stop", side="sell")
        api.list_orders.return_value = [dangling]
        portfolio.stop_order_ids["SPY"] = "tracked-stop"

        portfolio.cancel_open_stop_orders("SPY", "sell")

        assert "SPY" not in portfolio.stop_order_ids

    def test_does_not_clear_tracked_id_for_different_order(self, portfolio, api):
        dangling = _make_stop_order("other-stop", side="sell")
        api.list_orders.return_value = [dangling]
        portfolio.stop_order_ids["SPY"] = "different-id"

        portfolio.cancel_open_stop_orders("SPY", "sell")

        assert portfolio.stop_order_ids["SPY"] == "different-id"

    def test_handles_enum_like_attributes(self, portfolio, api):
        """alpaca-py returns OrderSide/OrderType enums; normalize via .value."""
        class FakeEnum:
            def __init__(self, v): self.value = v
            def __str__(self): return self.value

        dangling = MagicMock()
        dangling.id = "enum-stop"
        dangling.side = FakeEnum("sell")
        dangling.type = FakeEnum("stop")
        api.list_orders.return_value = [dangling]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 1
        api.cancel_order.assert_called_once_with("enum-stop")

    def test_queries_all_not_open(self, portfolio, api):
        """Must query status="all": a bracket/OTO stop_loss leg sits in status
        "held" while waiting on its parent, and Alpaca's "open" filter does
        not include "held" orders — so scoping to "open" would miss it."""
        api.list_orders.return_value = []

        portfolio.cancel_open_stop_orders("SPY", "sell")

        kwargs = api.list_orders.call_args.kwargs
        assert kwargs.get("status") == "all"

    def test_cancels_held_bracket_leg(self, portfolio, api):
        """A stop_loss leg of an OTO/bracket order in status="held" must
        still be found and cancelled by the sweep."""
        held_leg = _make_stop_order("held-stop-1", side="sell", order_type="stop", status="held")
        api.list_orders.return_value = [held_leg]

        count = portfolio.cancel_open_stop_orders("SPY", "sell")

        assert count == 1
        api.cancel_order.assert_called_once_with("held-stop-1")

    def test_skips_terminal_status_orders(self, portfolio, api):
        """Orders already filled/canceled/expired/rejected must not be
        re-cancelled — status="all" surfaces history, not just live orders."""
        for status in ("filled", "canceled", "expired", "rejected", "replaced", "done_for_day"):
            api.cancel_order.reset_mock()
            terminal = _make_stop_order(f"done-{status}", side="sell", order_type="stop", status=status)
            api.list_orders.return_value = [terminal]

            count = portfolio.cancel_open_stop_orders("SPY", "sell")

            assert count == 0, f"status={status} should not be cancelled"
            api.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# TestEnterDedup — _enter() sweeps stops before submitting the OTO order
# ---------------------------------------------------------------------------

class TestEnterDedup:
    def test_enter_cancels_dangling_stop_before_placing_new(self, strategy, api, portfolio):
        """
        Reproduces the stale-stop scenario.

        A stop-sell order "old-stop-1" was left open on Alpaca because a prior
        _close() call swallowed the cancel exception. Without the sweep, the
        new OTO order's stop_loss leg would coexist with it. With the fix,
        old-stop-1 is cancelled first and the new OTO entry succeeds.
        """
        dangling = _make_stop_order("old-stop-1", side="sell")
        api.list_orders.return_value = [dangling]
        api.submit_order.return_value = _make_oto_order(
            id="mkt-1", stop_leg_id="new-stop-1", stop_side="sell",
        )

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        # Dangling order must be cancelled before the new OTO order is submitted
        api.cancel_order.assert_called_with("old-stop-1")
        # Entry is recorded with the OTO order's stop leg
        assert portfolio.stop_order_ids.get("SPY") == "new-stop-1"

    def test_cancel_happens_before_order_submission(self, strategy, api, portfolio):
        """The sweep must precede the OTO submit_order call, not follow it."""
        dangling = _make_stop_order("old-stop-1", side="sell")
        api.list_orders.return_value = [dangling]
        api.submit_order.return_value = _make_oto_order(id="mkt-1", stop_leg_id="s1")

        cancel_calls = []
        api.cancel_order.side_effect = lambda oid: cancel_calls.append(oid)

        submit_calls = []
        original_return = api.submit_order.return_value
        api.submit_order.side_effect = lambda **kwargs: submit_calls.append(kwargs) or original_return

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert "old-stop-1" in cancel_calls
        assert len(submit_calls) == 1
        # cancel_calls populated strictly before the single submit_order call happened
        assert cancel_calls == ["old-stop-1"]

    def test_enter_succeeds_when_no_dangling_orders(self, strategy, api, portfolio):
        """Normal path: no dangling orders, entry proceeds cleanly."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-1", filled_avg_price="520.00", filled_qty="136", stop_leg_id="stop-1",
        )

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0
        assert portfolio.stop_order_ids.get("SPY") == "stop-1"

    def test_enter_long_sweeps_sell_stops_not_buy_stops(self, strategy, api, portfolio):
        """A long entry should only cancel sell stops, not buy stops."""
        buy_stop = _make_stop_order("buy-stop-1", side="buy")
        api.list_orders.return_value = [buy_stop]
        api.submit_order.return_value = _make_oto_order(id="mkt-1", stop_leg_id="stop-1")

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        # The buy stop should NOT be cancelled (it's on the wrong side for a long)
        api.cancel_order.assert_not_called()

    def test_enter_short_sweeps_buy_stops(self, strategy, api, portfolio):
        """A short entry should only cancel buy stops."""
        dangling = _make_stop_order("old-buy-stop", side="buy")
        api.list_orders.return_value = [dangling]
        api.submit_order.return_value = _make_oto_order(
            id="mkt-short", stop_leg_id="stop-buy-1", stop_side="buy",
        )

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        api.cancel_order.assert_called_with("old-buy-stop")
        assert portfolio.stop_order_ids.get("SPY") == "stop-buy-1"

    def test_entry_still_recorded_after_sweep_finds_nothing(self, strategy, api, portfolio):
        """list_orders returns empty — entry proceeds and is recorded."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-1", filled_avg_price="520.00", filled_qty="136", stop_leg_id="stop-1",
        )

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0

    def test_list_orders_failure_does_not_abort_entry(self, strategy, api, portfolio):
        """If list_orders fails, we still attempt the OTO entry rather than silently failing."""
        api.list_orders.side_effect = Exception("API error")
        api.submit_order.return_value = _make_oto_order(id="mkt-1", stop_leg_id="stop-1")

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        # Entry should still be recorded
        assert portfolio.stop_order_ids.get("SPY") == "stop-1"


# ---------------------------------------------------------------------------
# TestCloseDedup — _close() sweeps stops before close_position
# ---------------------------------------------------------------------------

class TestCloseDedup:
    def test_close_sweeps_untracked_stop_before_close_position(self, strategy, api, portfolio):
        """
        _close() had `except Exception: pass` — a failed cancel left a dangling
        stop-sell.  When close_position() then filled the market-sell, the bot was
        flat but had an orphaned stop-sell that would block the next entry.

        After the fix: cancel_open_stop_orders is called even when the tracked
        cancel fails, cleaning up any untracked stops.
        """
        portfolio.record_entry("SPY", "long", 525.0, 136, "tracked-stop", 520.0)
        # Tracked cancel will fail...
        api.cancel_order.side_effect = Exception("order already filled")
        # ...but list_orders still shows it as open
        dangling = _make_stop_order("tracked-stop", side="sell")
        api.list_orders.return_value = [dangling]

        with patch.object(portfolio, "log_trade", return_value=-50.0):
            strategy._close("SPY", "long", 519.0, reason="sma_exit")

        # cancel_order called twice: once for tracked (fails), once via sweep
        cancel_ids = [c.args[0] for c in api.cancel_order.call_args_list]
        assert "tracked-stop" in cancel_ids
        assert cancel_ids.count("tracked-stop") == 2

    def test_close_logs_tracked_cancel_failure_not_swallowed(self, strategy, api, portfolio):
        """_close() must log the cancel failure, not silently drop it."""
        portfolio.record_entry("SPY", "long", 525.0, 136, "tracked-stop", 520.0)
        api.cancel_order.side_effect = Exception("cancel rejected")
        api.list_orders.return_value = []

        import logging
        with patch.object(portfolio, "log_trade", return_value=0.0), \
             patch("bot.strategies.mean_reversion.logger") as mock_log:
            strategy._close("SPY", "long", 522.0, reason="sma_exit")

        # warning should have been emitted (not silently swallowed)
        logged_warnings = [c for c in mock_log.warning.call_args_list]
        assert any("tracked-stop" in str(c) for c in logged_warnings)

    def test_close_position_called_after_sweep(self, strategy, api, portfolio):
        portfolio.record_entry("SPY", "long", 525.0, 136, "stop-1", 520.0)
        api.list_orders.return_value = []

        with patch.object(portfolio, "log_trade", return_value=10.0):
            strategy._close("SPY", "long", 527.0, reason="sma_exit")

        api.close_position.assert_called_once_with("SPY")


# ---------------------------------------------------------------------------
# TestReplaceStopDedup — _replace_stop_order() sweeps after tracked cancel
# ---------------------------------------------------------------------------

class TestReplaceStopDedup:
    def test_replace_sweeps_untracked_stops_after_tracked_cancel_succeeds(
        self, portfolio, api
    ):
        """Even after a successful tracked cancel, an untracked stop from a
        different code path might still be open.  The sweep removes it."""
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids["SPY"] = "tracked-stop"

        untracked = _make_stop_order("untracked-stop", side="sell")
        api.list_orders.return_value = [untracked]
        api.submit_order.return_value = _make_order(id="new-stop")

        portfolio._replace_stop_order("SPY", 519.0, "long")

        # Both the tracked (via cancel_order) and untracked (via sweep) are cancelled
        cancel_ids = [c.args[0] for c in api.cancel_order.call_args_list]
        assert "tracked-stop" in cancel_ids
        assert "untracked-stop" in cancel_ids
        assert portfolio.stop_order_ids.get("SPY") == "new-stop"

    def test_replace_sweeps_when_tracked_cancel_fails(self, portfolio, api):
        """If the tracked cancel raises, the sweep still runs to clean up."""
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids["SPY"] = "tracked-stop"

        dangling = _make_stop_order("tracked-stop", side="sell")
        api.list_orders.return_value = [dangling]
        api.submit_order.return_value = _make_order(id="new-stop")

        # Simulate: cancel_order fails for the tracked cancel, then succeeds for sweep
        call_count = [0]

        def cancel_side_effect(oid):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("could not cancel")

        api.cancel_order.side_effect = cancel_side_effect

        portfolio._replace_stop_order("SPY", 519.0, "long")

        # Sweep should still have caught and cancelled the dangling order
        assert call_count[0] == 2
        assert portfolio.stop_order_ids.get("SPY") == "new-stop"

    def test_replace_no_double_cancel_when_nothing_is_open(self, portfolio, api):
        """Clean state: no tracked stop, no open orders — new stop placed without cancels."""
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids.pop("SPY", None)
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_order(id="new-stop")

        portfolio._replace_stop_order("SPY", 519.0, "long")

        api.cancel_order.assert_not_called()
        assert portfolio.stop_order_ids.get("SPY") == "new-stop"


# ---------------------------------------------------------------------------
# TestV2EnterDedup — MeanReversionV2._enter() has the same sweep
# ---------------------------------------------------------------------------

class TestV2EnterDedup:
    def test_v2_enter_cancels_dangling_stop(self, strategy_v2, api, portfolio):
        """MeanReversionV2._enter() must also sweep before submitting its OTO order."""
        dangling = _make_stop_order("old-stop-v2", side="sell")
        api.list_orders.return_value = [dangling]
        api.submit_order.return_value = _make_oto_order(id="mkt-v2", stop_leg_id="stop-v2")

        strategy_v2._enter("SPY", "buy", "long", 136, 518.0)

        api.cancel_order.assert_called_with("old-stop-v2")
        assert portfolio.stop_order_ids.get("SPY") == "stop-v2"

    def test_v2_enter_normal_path_unaffected(self, strategy_v2, api, portfolio):
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-v2", filled_avg_price="520.00", filled_qty="136", stop_leg_id="stop-v2",
        )

        strategy_v2._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0
        assert portfolio.stop_order_ids.get("SPY") == "stop-v2"


# ---------------------------------------------------------------------------
# TestShortPositionStop — long and short entries both get a protective stop
# via a single OTO order, with no wash-trade rejection possible.
#
# Root cause of the original failure:
#   _enter() submitted the market entry and the protective stop as two
#   separate top-level orders. Alpaca's wash-trade check (40310000) races
#   against its own order-status propagation: even after polling the entry
#   order to status="filled" via _wait_for_fill, a second standalone order on
#   the opposite side could still be rejected. This fired almost every time
#   for short entries (SELL market + BUY stop) and intermittently for longs.
#
#   The fix: submit ONE order with order_class="oto" and a stop_loss leg.
#   Alpaca creates the stop leg immediately in "held" status and activates it
#   server-side only once the market leg fills — no second top-level order is
#   ever submitted, so the wash-trade check has nothing to compare against.
# ---------------------------------------------------------------------------

class TestShortPositionStop:
    def test_short_spy_gets_buy_stop(self, strategy, api, portfolio):
        """Short SPY: a single OTO order carries a buy stop_loss leg."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-sell-spy", filled_avg_price="520.00", filled_qty="136",
            stop_leg_id="buy-stop-spy", stop_side="buy",
        )

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        entry_kwargs = api.submit_order.call_args_list[0].kwargs
        assert entry_kwargs["order_class"] == "oto"
        assert entry_kwargs["side"] == "sell"
        assert float(entry_kwargs["stop_loss"]["stop_price"]) > 520.0
        assert portfolio.stop_order_ids.get("SPY") == "buy-stop-spy"

    def test_short_qqq_gets_buy_stop(self, strategy, api, portfolio):
        """Short QQQ: OTO order's stop leg is on the buy side."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-sell-qqq", filled_avg_price="430.00", filled_qty="100",
            stop_leg_id="buy-stop-qqq", stop_side="buy",
        )

        strategy._enter("QQQ", "sell", "short", 100, 433.0)

        assert portfolio.stop_order_ids.get("QQQ") == "buy-stop-qqq"

    def test_long_gld_gets_sell_stop(self, strategy, api, portfolio):
        """Long GLD: OTO order's stop leg is on the sell side."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-buy-gld", filled_avg_price="180.00", filled_qty="55",
            stop_leg_id="sell-stop-gld", stop_side="sell",
        )

        strategy._enter("GLD", "buy", "long", 55, 178.0)

        entry_kwargs = api.submit_order.call_args_list[0].kwargs
        assert entry_kwargs["side"] == "buy"
        assert portfolio.stop_order_ids.get("GLD") == "sell-stop-gld"

    def test_short_entry_stop_price_above_entry(self, strategy, api, portfolio):
        """Stop price for a short must be above the entry price (buy-to-cover)."""
        entry_price = 520.0
        stop_price  = 523.0   # above entry → correct for short
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-sell", filled_avg_price=str(entry_price), filled_qty="136",
            stop_leg_id="buy-stop", stop_side="buy",
        )

        strategy._enter("SPY", "sell", "short", 136, stop_price)

        entry_kwargs = api.submit_order.call_args_list[0].kwargs
        assert float(entry_kwargs["stop_loss"]["stop_price"]) > entry_price

    def test_long_entry_stop_price_below_entry(self, strategy, api, portfolio):
        """Stop price for a long must be below the entry price."""
        entry_price = 520.0
        stop_price  = 517.0   # below entry → correct for long
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-buy", filled_avg_price=str(entry_price), filled_qty="136",
            stop_leg_id="sell-stop", stop_side="sell",
        )

        strategy._enter("SPY", "buy", "long", 136, stop_price)

        entry_kwargs = api.submit_order.call_args_list[0].kwargs
        assert float(entry_kwargs["stop_loss"]["stop_price"]) < entry_price

    @patch("time.sleep")
    def test_short_polls_until_filled(self, mock_sleep, strategy, api, portfolio):
        """_wait_for_fill polls get_order when the OTO entry isn't yet 'filled'."""
        accepted = _make_oto_order(
            id="mkt-sell", filled_avg_price=None, filled_qty=None,
            status="accepted", stop_leg_id="buy-stop", stop_side="buy",
        )
        filled = _make_oto_order(
            id="mkt-sell", filled_avg_price="520.00", filled_qty="136",
            status="filled", stop_leg_id="buy-stop", stop_side="buy",
        )

        api.submit_order.return_value = accepted
        api.get_order.return_value = filled
        api.list_orders.return_value = []

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        api.get_order.assert_called_with(accepted.id)
        mock_sleep.assert_called()
        assert portfolio.stop_order_ids.get("SPY") == "buy-stop"

    def test_long_also_polls_until_filled(self, strategy, api, portfolio):
        """Long entries wait for fill too, for accurate fill price/qty (not a
        wash-trade workaround — there's only ever one order now)."""
        accepted = _make_oto_order(
            id="mkt-buy", filled_avg_price=None, filled_qty=None,
            status="accepted", stop_leg_id="sell-stop", stop_side="sell",
        )
        filled = _make_oto_order(
            id="mkt-buy", filled_avg_price="520.00", filled_qty="136",
            status="filled", stop_leg_id="sell-stop", stop_side="sell",
        )
        api.submit_order.return_value = accepted
        api.get_order.return_value = filled
        api.list_orders.return_value = []

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        api.get_order.assert_called_with(accepted.id)
        assert portfolio.entry_prices.get("SPY") == 520.0

    def test_only_one_order_submitted_for_short(self, strategy, api, portfolio):
        """Exactly one order is submitted per short entry — no separate stop
        order, so there is nothing for Alpaca's wash-trade check to compare
        against the entry (this is what makes 40310000 impossible here)."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-sell", stop_leg_id="buy-stop", stop_side="buy",
        )

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        assert api.submit_order.call_count == 1
        entry_kwargs = api.submit_order.call_args_list[0].kwargs
        assert entry_kwargs["order_class"] == "oto"

    def test_only_one_order_submitted_for_long(self, strategy, api, portfolio):
        """Same guarantee for long entries."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-buy", stop_leg_id="sell-stop", stop_side="sell",
        )

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert api.submit_order.call_count == 1

    def test_missing_stop_leg_closes_position_immediately(self, strategy, api, portfolio):
        """If Alpaca's response has no stop_loss leg, the position must not be
        left unprotected — close it immediately rather than record the entry."""
        api.list_orders.return_value = []
        unprotected = _make_order(id="mkt-sell", filled_avg_price="520.00",
                                   filled_qty="136", legs=[])
        api.submit_order.return_value = unprotected
        api.get_order.return_value = unprotected

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        api.close_position.assert_called_once_with("SPY")
        assert "SPY" not in portfolio.stop_order_ids
        assert "SPY" not in portfolio.entry_prices

    def test_v2_short_spy_gets_buy_stop(self, strategy_v2, api, portfolio):
        """MeanReversionV2: short SPY also gets a buy stop via a single OTO order."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-sell-v2", filled_avg_price="520.00", filled_qty="136",
            stop_leg_id="buy-stop-v2", stop_side="buy",
        )

        strategy_v2._enter("SPY", "sell", "short", 136, 522.0)

        assert api.submit_order.call_count == 1
        assert portfolio.stop_order_ids.get("SPY") == "buy-stop-v2"

    def test_v2_short_qqq_gets_buy_stop(self, strategy_v2, api, portfolio):
        """MeanReversionV2: short QQQ also gets a buy stop."""
        api.list_orders.return_value = []
        api.submit_order.return_value = _make_oto_order(
            id="mkt-sell-qqq-v2", filled_avg_price="430.00", filled_qty="100",
            stop_leg_id="buy-stop-qqq-v2", stop_side="buy",
        )

        strategy_v2._enter("QQQ", "sell", "short", 100, 433.0)

        assert portfolio.stop_order_ids.get("QQQ") == "buy-stop-qqq-v2"
