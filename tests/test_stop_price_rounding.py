"""Tests for equity stop-price rounding.

Alpaca rejects equity stop orders whose stop_price has more than 2 decimal
places (e.g. 730.5911).  round_stop_price() must floor equity sell stops and
ceil equity buy stops so the submitted price is always a valid 2-dp string.
"""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bot.portfolio import Portfolio
from bot.risk_manager import round_stop_price
from bot.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Helpers shared with test_failure_modes
# ---------------------------------------------------------------------------

def _make_order(
    filled_avg_price="520.00",
    filled_qty="136",
    limit_price=None,
    id="order-1",
):
    o = MagicMock()
    o.filled_avg_price = filled_avg_price
    o.filled_qty = filled_qty
    o.limit_price = limit_price
    o.id = id
    return o


def _decimal_places(price_str: str) -> int:
    if "." not in price_str:
        return 0
    return len(price_str.split(".")[1])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    m = MagicMock()
    m.list_positions.return_value = []
    return m


@pytest.fixture
def portfolio(api, tmp_path):
    with patch.object(Portfolio, "_init_log_files"):
        return Portfolio(api, state_file=str(tmp_path / "state.json"))


@pytest.fixture
def rm():
    m = MagicMock()
    m.get_account_equity.return_value = 400_000.0
    m.calculate_atr.return_value = 1.5
    m.integer_position_size.return_value = 136
    m.stop_price.return_value = 518.0
    return m


@pytest.fixture
def strategy(api, rm, portfolio):
    return MeanReversionStrategy(api, rm, portfolio)


# ---------------------------------------------------------------------------
# round_stop_price — unit tests
# ---------------------------------------------------------------------------

class TestRoundStopPrice:
    # SPY sell stop (protecting a long): floor to 2 dp
    def test_spy_sell_stop_floors_to_2dp(self):
        assert round_stop_price(730.5911, "sell", "SPY") == 730.59

    def test_spy_sell_stop_floors_when_third_decimal_nonzero(self):
        assert round_stop_price(729.4089, "sell", "SPY") == 729.40

    # SPY buy stop (protecting a short): ceil to 2 dp
    def test_spy_buy_stop_ceils_to_2dp(self):
        assert round_stop_price(730.5911, "buy", "SPY") == 730.60

    def test_spy_buy_stop_ceils_when_exact_cent(self):
        # 731.60 is already 2 dp; ceil should not add a cent
        assert round_stop_price(731.60, "buy", "SPY") == 731.60

    # QQQ sell stop
    def test_qqq_sell_stop_floors_to_2dp(self):
        assert round_stop_price(459.3711, "sell", "QQQ") == 459.37

    # QQQ buy stop
    def test_qqq_buy_stop_ceils_to_2dp(self):
        assert round_stop_price(461.1201, "buy", "QQQ") == 461.13

    # Exact 2-decimal input is unchanged
    def test_exact_2dp_sell_unchanged(self):
        assert round_stop_price(730.50, "sell", "SPY") == 730.50

    def test_exact_2dp_buy_unchanged(self):
        assert round_stop_price(730.50, "buy", "SPY") == 730.50

    # Equity results never exceed 2 decimal places
    def test_equity_result_has_at_most_2dp(self):
        result = round_stop_price(730.5911, "sell", "SPY")
        # Decimal comparison is exact
        assert Decimal(str(result)) == Decimal(str(result)).quantize(Decimal("0.01"))

    # Crypto uses 4 dp, not 2
    def test_crypto_rounds_to_4dp(self):
        assert round_stop_price(45123.56789, "sell", "BTC/USD") == 45123.5679

    def test_crypto_buy_stop_rounds_to_4dp(self):
        assert round_stop_price(45123.56781, "buy", "BTC/USD") == 45123.5678


# ---------------------------------------------------------------------------
# mean_reversion._enter — submitted stop_price must have ≤ 2 dp for equities
# ---------------------------------------------------------------------------

class TestEnterStopPriceSubmission:
    def test_spy_long_entry_submits_2dp_stop(self, strategy, api):
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="730.59", filled_qty="136"),
            _make_order(id="stop-1"),
        ]
        strategy._enter("SPY", "buy", "long", 136, 730.5911)

        stop_call = api.submit_order.call_args_list[1]
        submitted = stop_call.kwargs["stop_price"]
        assert _decimal_places(submitted) <= 2, f"stop_price={submitted!r} has more than 2 dp"

    def test_spy_long_entry_floors_stop_price(self, strategy, api):
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="730.59", filled_qty="136"),
            _make_order(id="stop-1"),
        ]
        strategy._enter("SPY", "buy", "long", 136, 730.5911)

        stop_call = api.submit_order.call_args_list[1]
        assert stop_call.kwargs["stop_price"] == "730.59"

    def test_qqq_short_entry_submits_2dp_stop(self, strategy, api):
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="460.20", filled_qty="136"),
            _make_order(id="stop-1"),
        ]
        strategy._enter("QQQ", "sell", "short", 136, 461.1234)

        stop_call = api.submit_order.call_args_list[1]
        submitted = stop_call.kwargs["stop_price"]
        assert _decimal_places(submitted) <= 2, f"stop_price={submitted!r} has more than 2 dp"

    def test_qqq_short_entry_ceils_stop_price(self, strategy, api):
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="460.20", filled_qty="136"),
            _make_order(id="stop-1"),
        ]
        strategy._enter("QQQ", "sell", "short", 136, 461.1201)

        stop_call = api.submit_order.call_args_list[1]
        assert stop_call.kwargs["stop_price"] == "461.13"

    def test_record_entry_receives_rounded_stop(self, strategy, api, portfolio):
        """The in-memory trailing stop must match the broker order price."""
        api.submit_order.side_effect = [
            _make_order(id="mkt-1", filled_avg_price="730.59", filled_qty="136"),
            _make_order(id="stop-1"),
        ]
        strategy._enter("SPY", "buy", "long", 136, 730.5911)

        stored = portfolio.trailing_stops.get("SPY")
        assert stored == 730.59


# ---------------------------------------------------------------------------
# portfolio._replace_stop_order — submitted stop_price must have ≤ 2 dp
# ---------------------------------------------------------------------------

class TestReplaceStopOrderSubmission:
    def test_spy_long_replace_submits_2dp_stop(self, portfolio, api):
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids["SPY"] = "old-stop"
        api.cancel_order.return_value = None
        api.submit_order.return_value = _make_order(id="new-stop")

        portfolio._replace_stop_order("SPY", 519.5678, "long")

        stop_call = api.submit_order.call_args
        submitted = stop_call.kwargs["stop_price"]
        assert _decimal_places(submitted) <= 2, f"stop_price={submitted!r} has more than 2 dp"

    def test_spy_long_replace_floors_stop(self, portfolio, api):
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids["SPY"] = "old-stop"
        api.cancel_order.return_value = None
        api.submit_order.return_value = _make_order(id="new-stop")

        portfolio._replace_stop_order("SPY", 519.5678, "long")

        stop_call = api.submit_order.call_args
        assert stop_call.kwargs["stop_price"] == "519.56"

    def test_qqq_short_replace_ceils_stop(self, portfolio, api):
        portfolio.entry_sizes["QQQ"] = 100
        portfolio.stop_order_ids["QQQ"] = "old-stop"
        api.cancel_order.return_value = None
        api.submit_order.return_value = _make_order(id="new-stop")

        portfolio._replace_stop_order("QQQ", 461.1201, "short")

        stop_call = api.submit_order.call_args
        assert stop_call.kwargs["stop_price"] == "461.13"


# ---------------------------------------------------------------------------
# portfolio.update_trailing_stop — stored candidate must have ≤ 2 dp
# ---------------------------------------------------------------------------

class TestUpdateTrailingStopRounding:
    def test_spy_long_stores_2dp_trailing_stop(self, portfolio, api):
        portfolio.trailing_stops["SPY"] = 518.50
        portfolio.entry_directions["SPY"] = "long"
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids["SPY"] = "stop-1"
        api.cancel_order.return_value = None
        api.submit_order.return_value = _make_order(id="stop-2")

        # current_price=525, atr=1.5, multiplier=1.5 → raw=522.75, floor→522.75 (exact 2dp)
        # Use values that produce a non-2dp raw candidate
        # current_price=525, atr=1.5, multiplier=2.0/3 → 524.0 (too clean); use atr=1.333
        # Easier: current_price=525, atr=1.333, multiplier=2.0 → 525 - 2.666 = 522.334 → floor→522.33
        portfolio.update_trailing_stop("SPY", 525.0, 1.333, 2.0, "long")

        stored = portfolio.trailing_stops["SPY"]
        assert Decimal(str(stored)) == Decimal(str(stored)).quantize(Decimal("0.01"))

    def test_spy_long_trailing_stop_submitted_with_2dp(self, portfolio, api):
        portfolio.trailing_stops["SPY"] = 518.50
        portfolio.entry_directions["SPY"] = "long"
        portfolio.entry_sizes["SPY"] = 136
        portfolio.stop_order_ids["SPY"] = "stop-1"
        api.cancel_order.return_value = None
        api.submit_order.return_value = _make_order(id="stop-2")

        portfolio.update_trailing_stop("SPY", 525.0, 1.333, 2.0, "long")

        if api.submit_order.called:
            submitted = api.submit_order.call_args.kwargs["stop_price"]
            assert _decimal_places(submitted) <= 2
