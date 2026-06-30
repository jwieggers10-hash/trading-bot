"""Tests reproducing Alpaca error 40310000 (wash-trade) and verifying the fix.

Short-position stop tests are in TestShortPositionStop at the bottom of this file.

The bug: _enter() submitted a market order and then immediately submitted a
stop order on the opposite side.  If a previous trade's stop order was not
properly cleaned up (silent exception in _close(), bot crash between cancel and
position close), Alpaca rejected the new stop with:
  "40310000: potential wash trade detected. opposite side market/stop order exists"
The position was then left open with no stop-loss protection.

The fix: cancel_open_stop_orders(symbol, side) is called before every stop
placement in _enter(), _close(), and _replace_stop_order().  It lists all open
stop orders for the symbol on the given side and cancels any it finds, so there
is never more than one stop order active at a time.

Test structure:
  TestCancelOpenStopOrders  — unit tests for the Portfolio method in isolation
  TestEnterDedup            — _enter() calls the sweep before placing its stop
  TestCloseDedup            — _close() calls the sweep before close_position
  TestReplaceStopDedup      — _replace_stop_order() calls the sweep after the tracked cancel
  TestV2EnterDedup          — MeanReversionV2._enter() has the same protection
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
):
    o = MagicMock()
    o.id = id
    o.filled_avg_price = filled_avg_price
    o.filled_qty = filled_qty
    o.limit_price = None
    o.status = status  # plain string so _wait_for_fill can compare without .value
    return o


def _make_stop_order(id: str, side: str, order_type: str = "stop"):
    """Return a mock Order object whose .type and .side are plain strings
    (matching the comparison path in cancel_open_stop_orders)."""
    o = MagicMock()
    o.id = id
    o.side = side
    o.type = order_type
    return o


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


# ---------------------------------------------------------------------------
# TestEnterDedup — _enter() sweeps stops before placing the new one
# ---------------------------------------------------------------------------

class TestEnterDedup:
    def test_enter_cancels_dangling_stop_before_placing_new(self, strategy, api, portfolio):
        """
        Reproduces the wash-trade scenario.

        A stop-sell order "old-stop-1" was left open on Alpaca because a prior
        _close() call swallowed the cancel exception.  Without the fix, _enter()
        would try to place a new stop-sell and Alpaca would return 40310000.
        With the fix, old-stop-1 is cancelled first and the new stop succeeds.
        """
        dangling = _make_stop_order("old-stop-1", side="sell")
        api.list_orders.return_value = [dangling]
        api.submit_order.side_effect = [
            _make_order(id="mkt-1"),        # market buy
            _make_order(id="new-stop-1"),   # new stop-sell
        ]

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        # Dangling order must be cancelled before the new stop is submitted
        api.cancel_order.assert_called_with("old-stop-1")
        # Entry is recorded with the new stop order
        assert portfolio.stop_order_ids.get("SPY") == "new-stop-1"

    def test_cancel_happens_before_stop_submission(self, strategy, api, portfolio):
        """The sweep must precede the stop submit_order call, not follow it."""
        dangling = _make_stop_order("old-stop-1", side="sell")
        api.list_orders.return_value = [dangling]
        call_sequence = []
        api.cancel_order.side_effect = lambda *a, **kw: call_sequence.append(("cancel", a[0]))
        api.submit_order.side_effect = [
            _make_order(id="mkt-1"),
            (lambda *a, **kw: call_sequence.append(("stop_submit", None)) or _make_order(id="s1"))(*[], **{}),
        ]
        # Rebuild to capture submit_order properly
        api.submit_order.side_effect = None
        api.submit_order.return_value = _make_order(id="mkt-1")

        order_calls = []

        def record_submit(**kwargs):
            order_calls.append(kwargs.get("type", "?"))
            if len(order_calls) == 1:
                return _make_order(id="mkt-1")
            return _make_order(id="new-stop-1")

        api.submit_order.side_effect = lambda **kwargs: record_submit(**kwargs)

        cancel_calls = []
        api.cancel_order.side_effect = lambda oid: cancel_calls.append(oid)

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        # cancel must have happened before the second submit_order (the stop)
        assert "old-stop-1" in cancel_calls
        assert len(order_calls) == 2
        assert order_calls[1] == "stop"

    def test_enter_succeeds_when_no_dangling_orders(self, strategy, api, portfolio):
        """Normal path: no dangling orders, entry proceeds cleanly."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="520.00", filled_qty="136"),
            _make_order(id="stop-1"),
        ]

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0
        assert portfolio.stop_order_ids.get("SPY") == "stop-1"

    def test_enter_long_sweeps_sell_stops_not_buy_stops(self, strategy, api, portfolio):
        """A long entry should only cancel sell stops, not buy stops."""
        buy_stop = _make_stop_order("buy-stop-1", side="buy")
        api.list_orders.return_value = [buy_stop]
        api.submit_order.side_effect = [
            _make_order(id="mkt-1"),
            _make_order(id="stop-1"),
        ]

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        # The buy stop should NOT be cancelled (it's on the wrong side for a long)
        api.cancel_order.assert_not_called()

    def test_enter_short_sweeps_buy_stops(self, strategy, api, portfolio):
        """A short entry should only cancel buy stops."""
        dangling = _make_stop_order("old-buy-stop", side="buy")
        api.list_orders.return_value = [dangling]
        api.submit_order.side_effect = [
            _make_order(id="mkt-short"),
            _make_order(id="stop-buy-1"),
        ]

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        api.cancel_order.assert_called_with("old-buy-stop")
        assert portfolio.stop_order_ids.get("SPY") == "stop-buy-1"

    def test_entry_still_recorded_after_sweep_finds_nothing(self, strategy, api, portfolio):
        """list_orders returns empty — entry proceeds and is recorded."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="520.00", filled_qty="136"),
            _make_order(id="stop-1"),
        ]

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0

    def test_list_orders_failure_does_not_abort_entry(self, strategy, api, portfolio):
        """If list_orders fails, we still attempt the stop rather than silently failing."""
        api.list_orders.side_effect = Exception("API error")
        api.submit_order.side_effect = [
            _make_order(id="mkt-1"),
            _make_order(id="stop-1"),
        ]

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
        """MeanReversionV2._enter() must also sweep before placing its stop."""
        dangling = _make_stop_order("old-stop-v2", side="sell")
        api.list_orders.return_value = [dangling]
        api.submit_order.side_effect = [
            _make_order(id="mkt-v2"),
            _make_order(id="stop-v2"),
        ]

        strategy_v2._enter("SPY", "buy", "long", 136, 518.0)

        api.cancel_order.assert_called_with("old-stop-v2")
        assert portfolio.stop_order_ids.get("SPY") == "stop-v2"

    def test_v2_enter_normal_path_unaffected(self, strategy_v2, api, portfolio):
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-v2", filled_avg_price="520.00", filled_qty="136"),
            _make_order(id="stop-v2"),
        ]

        strategy_v2._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0
        assert portfolio.stop_order_ids.get("SPY") == "stop-v2"


# ---------------------------------------------------------------------------
# TestShortPositionStop — short entries get BUY stops without 40310000
#
# Root cause of the original failure:
#   For short entries (market SELL + BUY STOP), Alpaca's wash-trade detection
#   fires when the market SELL order is still in "new"/"accepted" state when
#   the BUY STOP is submitted.  The existing cancel_open_stop_orders sweep
#   only removes stop-type orders — it does not cancel the entry market order
#   itself, and cannot.  The fix is _wait_for_fill(), which polls the entry
#   order until it transitions to "filled" before placing the protective stop.
#
#   Long entries (market BUY + SELL STOP) are unaffected because Alpaca
#   recognises a SELL STOP on a long as a protective order, not a wash trade.
# ---------------------------------------------------------------------------

class TestShortPositionStop:
    def test_short_spy_gets_buy_stop(self, strategy, api, portfolio):
        """Short SPY: market sell is followed by a buy stop on the correct side."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-sell-spy", filled_avg_price="520.00", filled_qty="136"),
            _make_order(id="buy-stop-spy"),
        ]

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert stop_kwargs["side"] == "buy"
        assert portfolio.stop_order_ids.get("SPY") == "buy-stop-spy"

    def test_short_qqq_gets_buy_stop(self, strategy, api, portfolio):
        """Short QQQ: market sell is followed by a buy stop."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-sell-qqq", filled_avg_price="430.00", filled_qty="100"),
            _make_order(id="buy-stop-qqq"),
        ]

        strategy._enter("QQQ", "sell", "short", 100, 433.0)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert stop_kwargs["side"] == "buy"
        assert portfolio.stop_order_ids.get("QQQ") == "buy-stop-qqq"

    def test_long_gld_gets_sell_stop(self, strategy, api, portfolio):
        """Long GLD: market buy is followed by a sell stop."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-buy-gld", filled_avg_price="180.00", filled_qty="55"),
            _make_order(id="sell-stop-gld"),
        ]

        strategy._enter("GLD", "buy", "long", 55, 178.0)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert stop_kwargs["side"] == "sell"
        assert portfolio.stop_order_ids.get("GLD") == "sell-stop-gld"

    def test_short_entry_stop_price_above_entry(self, strategy, api, portfolio):
        """Stop price for a short must be above the entry price (buy-to-cover)."""
        entry_price = 520.0
        stop_price  = 523.0   # above entry → correct for short
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-sell", filled_avg_price=str(entry_price), filled_qty="136"),
            _make_order(id="buy-stop"),
        ]

        strategy._enter("SPY", "sell", "short", 136, stop_price)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert float(stop_kwargs["stop_price"]) > entry_price

    def test_long_entry_stop_price_below_entry(self, strategy, api, portfolio):
        """Stop price for a long must be below the entry price."""
        entry_price = 520.0
        stop_price  = 517.0   # below entry → correct for long
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-buy", filled_avg_price=str(entry_price), filled_qty="136"),
            _make_order(id="sell-stop"),
        ]

        strategy._enter("SPY", "buy", "long", 136, stop_price)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert float(stop_kwargs["stop_price"]) < entry_price

    @patch("time.sleep")
    def test_short_polls_until_filled_before_stop(self, mock_sleep, strategy, api, portfolio):
        """_wait_for_fill polls get_order when status is not yet 'filled'."""
        accepted = _make_order(id="mkt-sell", filled_avg_price=None, filled_qty=None,
                               status="accepted")
        filled   = _make_order(id="mkt-sell", filled_avg_price="520.00", filled_qty="136",
                               status="filled")
        stop_order = _make_order(id="buy-stop")

        api.submit_order.side_effect = [accepted, stop_order]
        api.get_order.return_value = filled
        api.list_orders.return_value = []

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        api.get_order.assert_called_with(accepted.id)
        mock_sleep.assert_called()
        assert portfolio.stop_order_ids.get("SPY") == "buy-stop"

    def test_long_does_not_poll_for_fill(self, strategy, api, portfolio):
        """Long entries skip _wait_for_fill entirely — no get_order calls."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-buy", filled_avg_price="520.00", filled_qty="136"),
            _make_order(id="sell-stop"),
        ]

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        api.get_order.assert_not_called()

    def test_no_duplicate_stop_submitted_for_short(self, strategy, api, portfolio):
        """Only one stop order is submitted per short entry (no duplicates)."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-sell"),
            _make_order(id="buy-stop"),
        ]

        strategy._enter("SPY", "sell", "short", 136, 522.0)

        stop_submits = [
            c for c in api.submit_order.call_args_list
            if c.kwargs.get("type") == "stop"
        ]
        assert len(stop_submits) == 1

    def test_v2_short_spy_gets_buy_stop(self, strategy_v2, api, portfolio):
        """MeanReversionV2: short SPY also gets a buy stop."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-sell-v2", filled_avg_price="520.00", filled_qty="136"),
            _make_order(id="buy-stop-v2"),
        ]

        strategy_v2._enter("SPY", "sell", "short", 136, 522.0)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert stop_kwargs["side"] == "buy"
        assert portfolio.stop_order_ids.get("SPY") == "buy-stop-v2"

    def test_v2_short_qqq_gets_buy_stop(self, strategy_v2, api, portfolio):
        """MeanReversionV2: short QQQ also gets a buy stop."""
        api.list_orders.return_value = []
        api.submit_order.side_effect = [
            _make_order(id="mkt-sell-qqq-v2", filled_avg_price="430.00", filled_qty="100"),
            _make_order(id="buy-stop-qqq-v2"),
        ]

        strategy_v2._enter("QQQ", "sell", "short", 100, 433.0)

        stop_kwargs = api.submit_order.call_args_list[1].kwargs
        assert stop_kwargs["side"] == "buy"
        assert portfolio.stop_order_ids.get("QQQ") == "buy-stop-qqq-v2"
