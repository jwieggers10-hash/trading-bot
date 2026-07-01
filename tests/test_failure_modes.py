"""Tests proving each failure-mode fix works correctly.

Covers:
  Fix 1 — PositionFetchError raised (never returns {}) on API failure
  Fix 2 — Stop order failure after market fill triggers emergency close
  Fix 3 — Partial fills: stop order uses actual filled qty, not requested qty
  Fix 4 — Position state persists to disk and is reconciled on restart
  Fix 5 — Stop-out cooldown prevents re-entry for STOP_COOLDOWN_SECONDS
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.portfolio import Portfolio, PositionFetchError
from bot.strategies.mean_reversion import MeanReversionStrategy, _filled_qty
from config import STOP_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(symbol: str, qty: float):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    return pos


def _make_order(
    filled_avg_price: str | None = "520.00",
    filled_qty: str | None = "136",
    limit_price=None,
    id: str = "order-1",
    status: str = "filled",
    legs=None,
):
    order = MagicMock()
    order.filled_avg_price = filled_avg_price
    order.filled_qty = filled_qty
    order.limit_price = limit_price
    order.id = id
    order.status = status  # plain string so _wait_for_fill doesn't poll for 10s
    order.legs = legs
    return order


def _make_oto_order(
    filled_avg_price: str | None = "520.00",
    filled_qty: str | None = "136",
    id: str = "mkt-1",
    stop_leg_id: str = "stop-1",
    stop_side: str = "sell",
    status: str = "filled",
):
    """A single OTO order response: market entry + a populated stop_loss leg.

    _enter() now submits one order_class="oto" order rather than a separate
    market order followed by a separate stop order."""
    leg = MagicMock()
    leg.id = stop_leg_id
    leg.side = stop_side
    leg.type = "stop"
    return _make_order(
        filled_avg_price=filled_avg_price, filled_qty=filled_qty, id=id,
        status=status, legs=[leg],
    )


def _make_df_bars(n: int = 25) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame long enough to pass the 22-bar check."""
    rng = np.random.default_rng(42)
    prices = 520.0 + rng.normal(0, 0.5, n).cumsum()
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": 1_000_000,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    mock = MagicMock()
    mock.list_positions.return_value = []
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


# ---------------------------------------------------------------------------
# Fix 1 — PositionFetchError on API failure
# ---------------------------------------------------------------------------

class TestPositionFetchError:
    def test_raises_when_list_positions_fails(self, api, portfolio):
        api.list_positions.side_effect = Exception("connection refused")
        with pytest.raises(PositionFetchError):
            portfolio.position_state("SPY")

    def test_position_state_returns_flat_when_symbol_absent(self, api, portfolio):
        api.list_positions.return_value = [_make_position("QQQ", 100)]
        assert portfolio.position_state("SPY") == "flat"

    def test_position_state_returns_long_for_positive_qty(self, api, portfolio):
        api.list_positions.return_value = [_make_position("SPY", 136)]
        assert portfolio.position_state("SPY") == "long"

    def test_position_state_returns_short_for_negative_qty(self, api, portfolio):
        api.list_positions.return_value = [_make_position("SPY", -136)]
        assert portfolio.position_state("SPY") == "short"

    def test_strategy_skips_tick_on_position_fetch_failure(self, strategy, api):
        api.list_positions.side_effect = Exception("network timeout")
        bars = _make_df_bars()
        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value={
                 "price": 514.0, "sma": 522.0, "upper": 526.0, "lower": 515.0,
             }):
            strategy.run("SPY")
        api.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2 — Missing protective stop triggers emergency close
#
# _enter() now submits a single order_class="oto" order (market entry +
# stop_loss leg) instead of two separate orders, so the old "stop order
# submission raises" scenario no longer exists as a code path. What still
# must hold: if the broker response has no protective stop leg for any
# reason, the position must not be left unprotected — close it immediately.
# ---------------------------------------------------------------------------

class TestStopOrderFailure:
    def test_emergency_close_when_stop_leg_missing(self, strategy, api, portfolio):
        """If the OTO order response has no stop_loss leg, close immediately."""
        api.submit_order.return_value = _make_order(id="mkt-1", legs=[])
        api.get_order.return_value = _make_order(id="mkt-1", legs=[])

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        api.close_position.assert_called_once_with("SPY")

    def test_record_entry_not_called_when_stop_leg_missing(self, strategy, api, portfolio):
        api.submit_order.return_value = _make_order(id="mkt-1", legs=[])
        api.get_order.return_value = _make_order(id="mkt-1", legs=[])

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert "SPY" not in portfolio.entry_prices
        assert "SPY" not in portfolio.stop_order_ids

    def test_market_order_failure_does_not_trigger_close(self, strategy, api, portfolio):
        api.submit_order.side_effect = Exception("market order rejected")
        strategy._enter("SPY", "buy", "long", 136, 518.0)

        api.close_position.assert_not_called()

    def test_successful_entry_calls_record_entry(self, strategy, api, portfolio):
        api.submit_order.return_value = _make_oto_order(
            id="mkt-1", filled_avg_price="520.00", filled_qty="136", stop_leg_id="stop-1",
        )
        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_prices.get("SPY") == 520.0
        assert portfolio.stop_order_ids.get("SPY") == "stop-1"


# ---------------------------------------------------------------------------
# Fix 3 — Partial fills use actual filled quantity
# ---------------------------------------------------------------------------

class TestPartialFillHandling:
    # Note: there is no longer a client-submitted "stop order qty" to assert
    # on — the stop_loss leg is part of the single OTO order, and Alpaca
    # sizes it to the parent's actual filled quantity server-side. What we
    # can and must still verify is that our own bookkeeping (entry_sizes,
    # used for position sizing / locked-notional accounting) reflects the
    # actual fill, not the originally requested size.
    def test_record_entry_uses_filled_qty(self, strategy, api, portfolio):
        api.submit_order.return_value = _make_oto_order(
            id="mkt-1", filled_avg_price="520.00", filled_qty="120", stop_leg_id="stop-1",
        )
        strategy._enter("SPY", "buy", "long", 136, 518.0)

        assert portfolio.entry_sizes.get("SPY") == 120

    def test_filled_qty_helper_uses_order_value(self):
        order = _make_order(filled_qty="120")
        assert _filled_qty(order, 136) == 120

    def test_filled_qty_helper_falls_back_to_requested_when_none(self):
        order = _make_order(filled_qty=None)
        assert _filled_qty(order, 136) == 136

    def test_filled_qty_helper_falls_back_when_zero(self):
        order = _make_order(filled_qty="0")
        assert _filled_qty(order, 136) == 136

    def test_filled_qty_helper_preserves_int_type(self):
        order = _make_order(filled_qty="120")
        result = _filled_qty(order, 136)
        assert isinstance(result, int)

    def test_filled_qty_helper_preserves_float_type_for_crypto(self):
        order = _make_order(filled_qty="0.123456")
        result = _filled_qty(order, 0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(0.123456)

    def test_partial_fill_with_missing_stop_leg_triggers_emergency_close(
        self, strategy, api, portfolio
    ):
        # Market leg partially fills, but the response carries no stop_loss leg
        order = _make_order(id="mkt-1", filled_qty="120", legs=[])
        api.submit_order.return_value = order
        api.get_order.return_value = order

        strategy._enter("SPY", "buy", "long", 136, 518.0)

        api.close_position.assert_called_once_with("SPY")
        assert "SPY" not in portfolio.entry_prices


# ---------------------------------------------------------------------------
# Fix 4 — Position state persists to disk and reconciles on restart
# ---------------------------------------------------------------------------

class TestPositionStatePersistence:
    def test_state_written_to_disk_on_record_entry(self, portfolio, tmp_path):
        portfolio.record_entry("SPY", "long", 520.0, 136, "stop-1", 518.0)

        data = json.loads((tmp_path / "state.json").read_text())
        assert "SPY" in data["positions"]
        pos = data["positions"]["SPY"]
        assert pos["entry_price"] == 520.0
        assert pos["entry_size"] == 136
        assert pos["direction"] == "long"
        assert pos["stop_order_id"] == "stop-1"

    def test_state_cleared_from_disk_on_clear_position(self, portfolio, tmp_path):
        portfolio.record_entry("SPY", "long", 520.0, 136, "stop-1", 518.0)
        portfolio.clear_position("SPY")

        data = json.loads((tmp_path / "state.json").read_text())
        assert "SPY" not in data["positions"]

    def test_state_loaded_on_restart_when_position_still_open(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {
                    "entry_price": 520.0, "entry_size": 136,
                    "direction": "long", "trailing_stop": 518.0,
                    "stop_order_id": "stop-1",
                }
            },
            "stop_cooldowns": {},
        }))
        api.list_positions.return_value = [_make_position("SPY", 136)]

        with patch.object(Portfolio, "_init_log_files"):
            p2 = Portfolio(api, state_file=str(state_file))

        assert p2.entry_prices.get("SPY") == 520.0
        assert p2.entry_sizes.get("SPY") == 136
        assert p2.entry_directions.get("SPY") == "long"
        assert p2.trailing_stops.get("SPY") == 518.0
        assert p2.stop_order_ids.get("SPY") == "stop-1"

    def test_reconcile_clears_position_closed_while_offline(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {
                    "entry_price": 520.0, "entry_size": 136,
                    "direction": "long", "trailing_stop": 518.0,
                    "stop_order_id": "stop-1",
                }
            },
            "stop_cooldowns": {},
        }))
        api.list_positions.return_value = []  # SPY closed while bot was offline

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        assert "SPY" not in p.entry_prices

    def test_reconcile_clears_direction_mismatch(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {
                    "entry_price": 520.0, "entry_size": 136,
                    "direction": "long", "trailing_stop": 518.0,
                    "stop_order_id": "stop-1",
                }
            },
            "stop_cooldowns": {},
        }))
        # Alpaca shows SPY as short (direction mismatch)
        api.list_positions.return_value = [_make_position("SPY", -50)]

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        assert "SPY" not in p.entry_prices

    def test_reconcile_skipped_gracefully_when_api_fails(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {
                    "entry_price": 520.0, "entry_size": 136,
                    "direction": "long", "trailing_stop": 518.0,
                    "stop_order_id": "stop-1",
                }
            },
            "stop_cooldowns": {},
        }))
        api.list_positions.side_effect = Exception("network error")

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        # State retained from disk when reconciliation cannot run
        assert p.entry_prices.get("SPY") == 520.0


# ---------------------------------------------------------------------------
# Fix 5 — Stop-out cooldown prevents re-entry
# ---------------------------------------------------------------------------

class TestStopOutCooldown:
    def test_cooldown_active_immediately_after_stop(self, portfolio):
        portfolio.record_stop_out("SPY")
        assert portfolio.in_stop_cooldown("SPY") is True

    def test_no_cooldown_before_any_stop_out(self, portfolio):
        assert portfolio.in_stop_cooldown("SPY") is False

    def test_cooldown_active_just_before_expiry(self, portfolio):
        recent = datetime.now(timezone.utc) - timedelta(seconds=STOP_COOLDOWN_SECONDS - 60)
        portfolio._stop_out_times["SPY"] = recent
        assert portfolio.in_stop_cooldown("SPY") is True

    def test_cooldown_expired_after_timeout(self, portfolio):
        expired = datetime.now(timezone.utc) - timedelta(seconds=STOP_COOLDOWN_SECONDS + 1)
        portfolio._stop_out_times["SPY"] = expired
        assert portfolio.in_stop_cooldown("SPY") is False

    def test_cooldown_persists_across_restart(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        api.list_positions.return_value = []

        with patch.object(Portfolio, "_init_log_files"):
            p1 = Portfolio(api, state_file=str(state_file))
        p1.record_stop_out("SPY")

        with patch.object(Portfolio, "_init_log_files"):
            p2 = Portfolio(api, state_file=str(state_file))

        assert p2.in_stop_cooldown("SPY") is True

    def test_cooldown_independence_between_symbols(self, portfolio):
        portfolio.record_stop_out("SPY")
        assert portfolio.in_stop_cooldown("SPY") is True
        assert portfolio.in_stop_cooldown("QQQ") is False

    def test_strategy_skips_entry_during_cooldown(self, strategy, api, portfolio):
        portfolio.record_stop_out("SPY")
        bars = _make_df_bars()

        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value={
                 "price": 514.0,   # below lower band — would trigger long
                 "sma": 522.0,
                 "upper": 526.0,
                 "lower": 515.0,
             }):
            strategy.run("SPY")

        # No order submitted because cooldown is active
        api.submit_order.assert_not_called()

    def test_stop_out_recorded_when_hard_stop_triggered(self, strategy, api, portfolio):
        # position_state() queries the live API; tell it SPY is long
        api.list_positions.return_value = [_make_position("SPY", 136)]
        portfolio.record_entry("SPY", "long", 525.0, 136, "stop-1", 520.0)

        bars = _make_df_bars()
        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value={
                 "price": 519.0,   # below stop at 520 — triggers trailing_stop_triggered
                 "sma": 522.0, "upper": 526.0, "lower": 515.0,
             }), \
             patch.object(portfolio, "log_trade"), \
             patch.object(api, "close_position"):
            strategy.run("SPY")

        assert portfolio.in_stop_cooldown("SPY") is True
