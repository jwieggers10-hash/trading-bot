"""Tests for the position_state.json corruption fix and startup rebuild-from-broker.

Root cause (production, VPS, since 2026-06-30): _find_child_stop_id() returned
a raw uuid.UUID (alpaca-py types Order.id as UUID, not str). Portfolio._save_state()
opened the state file in "w" mode — truncating it immediately — then json.dump()
raised TypeError("Object of type UUID is not JSON serializable") partway through
serializing the first symbol's stop_order_id, leaving the file permanently
truncated at ~166 bytes. Every subsequent save repeated the same failure
(531 occurrences logged), and the load path's `except Exception: starting
fresh` meant a future restart would silently discard all local tracking for
every open position, with no recovery path.

The fix has three parts:
  1. _find_child_stop_id() now returns str(leg.id); _save_state() also passes
     default=str to json.dump() as defense in depth.
  2. _save_state() writes to a temp file, fsyncs, then os.replace()s onto the
     real path — a failed serialization can never truncate the last good file.
  3. _load_state() quarantines a malformed file (timestamped .corrupt rename)
     instead of leaving it to repeatedly fail to parse, and
     _reconcile_with_broker() now rebuilds local tracking from the broker's
     live positions when local state is empty, instead of silently starting
     flat with open positions on the account.

Test structure:
  TestUuidSerialization        — UUIDs never crash _save_state()
  TestFailedSerializationSafety — a failed save leaves the prior file untouched
  TestMalformedJsonRecovery    — a corrupt file is quarantined, not repeatedly retried
  TestAtomicReplacement        — no partial/temp file is ever left as the real state file
  TestRebuildFromBroker        — startup reconciliation rebuilds from broker truth
"""
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from bot.portfolio import Portfolio
from bot.strategies.mean_reversion import _find_child_stop_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(symbol: str, qty: float, avg_entry_price: float):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.avg_entry_price = str(avg_entry_price)
    return pos


def _make_order_with_uuid_leg(stop_side: str = "sell"):
    """Simulates exactly what alpaca-py returns: leg.id is a real uuid.UUID,
    not a string."""
    leg = MagicMock()
    leg.id = uuid.uuid4()
    leg.side = stop_side
    leg.type = "stop"
    order = MagicMock()
    order.legs = [leg]
    return order, leg.id


def _make_stop_order(id: str, side: str, order_type: str = "stop", status: str = "new",
                      stop_price: float = 400.0):
    o = MagicMock()
    o.id = id
    o.side = side
    o.type = order_type
    o.status = status
    o.stop_price = stop_price
    return o


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


# ---------------------------------------------------------------------------
# TestUuidSerialization
# ---------------------------------------------------------------------------

class TestUuidSerialization:
    def test_find_child_stop_id_returns_str_not_uuid(self):
        """The actual fix: _find_child_stop_id() must never hand back a raw
        UUID object — that's what flowed into stop_order_ids and crashed
        json.dump() in production."""
        order, real_uuid = _make_order_with_uuid_leg(stop_side="sell")

        result = _find_child_stop_id(order, api=MagicMock(), symbol="GLD", stop_side="sell")

        assert isinstance(result, str)
        assert result == str(real_uuid)

    def test_save_state_survives_a_raw_uuid_in_stop_order_ids(self, portfolio, tmp_path):
        """Defense in depth: even if a UUID slipped into stop_order_ids some
        other way, _save_state() (default=str) must not crash or truncate
        the file."""
        portfolio.entry_prices["GLD"] = 408.78317
        portfolio.entry_sizes["GLD"] = 68
        portfolio.entry_directions["GLD"] = "long"
        portfolio.trailing_stops["GLD"] = 397.9329
        portfolio.stop_order_ids["GLD"] = uuid.uuid4()  # raw UUID, not str

        portfolio._save_state()

        data = json.loads((tmp_path / "state.json").read_text())
        assert "GLD" in data["positions"]
        assert isinstance(data["positions"]["GLD"]["stop_order_id"], str)

    def test_record_entry_with_uuid_stop_id_round_trips_through_disk(self, portfolio, tmp_path):
        raw_id = uuid.uuid4()
        portfolio.record_entry("GLD", "long", 408.78317, 68, raw_id, 397.9329)

        data = json.loads((tmp_path / "state.json").read_text())
        assert data["positions"]["GLD"]["stop_order_id"] == str(raw_id)


# ---------------------------------------------------------------------------
# TestFailedSerializationSafety
# ---------------------------------------------------------------------------

class TestFailedSerializationSafety:
    def test_failed_save_leaves_previous_good_file_untouched(self, portfolio, tmp_path, api):
        """Write a good state, then force the next save to fail partway
        through serialization. The file on disk must still contain the
        PREVIOUS good content — this is the exact scenario that corrupted
        position_state.json in production (the old code truncated the file
        by opening it in "w" mode before json.dump() ever ran)."""
        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)
        good_content = (tmp_path / "state.json").read_text()
        assert "SPY" in good_content

        portfolio.entry_prices["QQQ"] = 717.2
        portfolio.entry_sizes["QQQ"] = 38
        portfolio.entry_directions["QQQ"] = "long"
        portfolio.trailing_stops["QQQ"] = 700.0
        portfolio.stop_order_ids["QQQ"] = "stop-qqq"

        with patch("json.dump", side_effect=TypeError("Object of type UUID is not JSON serializable")):
            portfolio._save_state()

        # File must be exactly what it was before the failed save attempt.
        assert (tmp_path / "state.json").read_text() == good_content
        data = json.loads(good_content)
        assert "SPY" in data["positions"]
        assert "QQQ" not in data["positions"]

    def test_failed_save_does_not_leave_a_dangling_temp_file(self, portfolio, tmp_path):
        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)

        with patch("json.dump", side_effect=TypeError("boom")):
            portfolio._save_state()

        assert not (tmp_path / "state.json.tmp").exists()

    def test_save_state_logs_error_on_failure_without_raising(self, portfolio):
        """_save_state() must never propagate — callers (record_entry etc.)
        call it inline and must not crash the trading loop over a disk issue."""
        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)
        with patch("json.dump", side_effect=OSError("disk full")):
            portfolio._save_state()  # must not raise


# ---------------------------------------------------------------------------
# TestMalformedJsonRecovery
# ---------------------------------------------------------------------------

class TestMalformedJsonRecovery:
    def test_malformed_file_is_quarantined_on_load(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        # Exactly the production shape: truncated mid-value.
        state_file.write_text(
            '{\n  "positions": {\n    "USO": {\n      "entry_price": 130.7,\n'
            '      "entry_size": 210,\n      "direction": "long",\n'
            '      "trailing_stop": 130.28,\n      "stop_order_id": '
        )

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        # Started fresh in memory.
        assert p.entry_prices == {}
        # Original corrupt content preserved for forensics under a quarantine name.
        corrupt_backups = list(tmp_path.glob("state.json.*.corrupt"))
        assert len(corrupt_backups) == 1
        assert "stop_order_id" in corrupt_backups[0].read_text()
        # The live path is no longer the malformed file.
        assert not state_file.exists()

    def test_next_save_after_quarantine_produces_valid_json(self, api, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("{not valid json at all")

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        p.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)

        data = json.loads(state_file.read_text())
        assert data["positions"]["SPY"]["entry_size"] == 35

    def test_missing_file_does_not_attempt_quarantine(self, api, tmp_path):
        """No file at all (first-ever run) must not error or create a
        spurious .corrupt backup — quarantine only applies to a file that
        actually failed to parse."""
        state_file = tmp_path / "state.json"

        with patch.object(Portfolio, "_init_log_files"):
            Portfolio(api, state_file=str(state_file))

        assert list(tmp_path.glob("*.corrupt")) == []

    def test_partial_symbol_data_does_not_leave_half_populated_state(self, api, tmp_path):
        """A file where one symbol's record is malformed (missing a required
        key) must not leave entry_prices populated for that symbol while
        other dicts are missing it — the whole load either fully succeeds or
        fully resets to empty."""
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {
                    "entry_price": 520.0, "entry_size": 35,
                    "direction": "long", "trailing_stop": 518.0,
                    # stop_order_id missing entirely — KeyError during parse
                }
            },
            "stop_cooldowns": {},
        }))

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        assert p.entry_prices == {}
        assert p.entry_sizes == {}
        assert p.trailing_stops == {}


# ---------------------------------------------------------------------------
# TestAtomicReplacement
# ---------------------------------------------------------------------------

class TestAtomicReplacement:
    def test_successful_save_leaves_no_temp_file(self, portfolio, tmp_path):
        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)
        assert not (tmp_path / "state.json.tmp").exists()
        assert (tmp_path / "state.json").exists()

    def test_save_calls_os_replace_with_temp_and_real_paths(self, portfolio):
        """The real file handle must never be opened in write mode directly
        — os.replace must be the mechanism that publishes the new content."""
        with patch("os.replace") as mock_replace:
            portfolio.record_entry("QQQ", "long", 717.2, 38, "stop-qqq", 700.0)

        mock_replace.assert_called_once()
        src, dst = mock_replace.call_args.args
        assert src == f"{portfolio._state_file}.tmp"
        assert dst == portfolio._state_file

    def test_repeated_saves_each_replace_cleanly(self, portfolio, tmp_path):
        for i, sym in enumerate(["SPY", "QQQ", "GLD", "USO"]):
            portfolio.record_entry(sym, "long", 100.0 + i, 10 + i, f"stop-{sym}", 90.0 + i)

        assert not (tmp_path / "state.json.tmp").exists()
        data = json.loads((tmp_path / "state.json").read_text())
        assert set(data["positions"].keys()) == {"SPY", "QQQ", "GLD", "USO"}


# ---------------------------------------------------------------------------
# TestRebuildFromBroker — startup reconciliation with empty local state
# ---------------------------------------------------------------------------

class TestRebuildFromBroker:
    def test_restart_with_active_positions_recovers_symbol_side_qty_price(self, api, tmp_path):
        """After a restart with no usable local state (e.g. right after
        quarantining a corrupt file) but real open broker positions, the bot
        must not silently come back up believing it's flat."""
        api.list_positions.return_value = [
            _make_position("SPY", 35, 767.397143),
            _make_position("QQQ", 38, 717.219474),
            _make_position("GLD", 68, 408.78317),
            _make_position("USO", 210, 130.7),
        ]
        api.list_orders.return_value = []  # no live stop orders found for any symbol

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))

        for sym, qty, price in [
            ("SPY", 35, 767.397143), ("QQQ", 38, 717.219474),
            ("GLD", 68, 408.78317), ("USO", 210, 130.7),
        ]:
            assert p.entry_sizes[sym] == pytest.approx(qty)
            assert p.entry_prices[sym] == pytest.approx(price)
            assert p.entry_directions[sym] == "long"

    def test_recovers_short_direction_from_negative_qty(self, api, tmp_path):
        api.list_positions.return_value = [_make_position("GLD", -67, 405.4797)]
        api.list_orders.return_value = []

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))

        assert p.entry_directions["GLD"] == "short"
        assert p.entry_sizes["GLD"] == pytest.approx(67)

    def test_restores_stop_order_id_and_trailing_stop_from_live_stop_order(self, api, tmp_path):
        """When a live protective stop order exists on the broker for the
        symbol, its id and price must be recovered — not just qty/price."""
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]
        live_stop = _make_stop_order("real-stop-gld", side="sell", stop_price=397.9329)
        api.list_orders.return_value = [live_stop]

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))

        assert p.stop_order_ids["GLD"] == "real-stop-gld"
        assert p.trailing_stops["GLD"] == pytest.approx(397.9329)

    def test_no_live_stop_order_found_uses_conservative_placeholder(self, api, tmp_path, caplog):
        """If no live stop order can be found for a recovered position, the
        code must not fabricate a plausible-looking real stop — it uses the
        entry price as an explicit placeholder and logs CRITICAL so a human
        knows this position's stop tracking is unverified."""
        api.list_positions.return_value = [_make_position("GLD", 68, 408.78317)]
        api.list_orders.return_value = []  # nothing live found

        import logging
        with caplog.at_level(logging.CRITICAL, logger="bot.portfolio"):
            with patch.object(Portfolio, "_init_log_files"):
                p = Portfolio(api, state_file=str(tmp_path / "state.json"))

        assert p.trailing_stops["GLD"] == pytest.approx(408.78317)
        assert p.stop_order_ids["GLD"] == ""
        assert any("no live protective stop" in r.message.lower() for r in caplog.records)

    def test_rebuilt_state_is_persisted_to_disk(self, api, tmp_path):
        api.list_positions.return_value = [_make_position("SPY", 35, 767.397143)]
        api.list_orders.return_value = [_make_stop_order("stop-spy", side="sell", stop_price=760.0)]

        with patch.object(Portfolio, "_init_log_files"):
            Portfolio(api, state_file=str(tmp_path / "state.json"))

        data = json.loads((tmp_path / "state.json").read_text())
        assert data["positions"]["SPY"]["entry_size"] == 35
        assert data["positions"]["SPY"]["stop_order_id"] == "stop-spy"

    def test_no_broker_positions_and_no_local_state_stays_flat(self, api, tmp_path):
        """The common, boring case: nothing open anywhere — must not error
        or invent phantom positions."""
        api.list_positions.return_value = []

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))

        assert p.entry_prices == {}

    def test_existing_local_state_is_not_touched_by_rebuild_path(self, api, tmp_path):
        """Rebuild-from-broker must only trigger when local state is empty —
        normal reconciliation (already tested elsewhere) must still run when
        local state exists."""
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {
                    "entry_price": 520.0, "entry_size": 35,
                    "direction": "long", "trailing_stop": 518.0,
                    "stop_order_id": "stop-spy",
                }
            },
            "stop_cooldowns": {},
        }))
        api.list_positions.return_value = [_make_position("SPY", 35, 520.0)]

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(state_file))

        # Untouched — came from the loaded file, not a broker rebuild.
        assert p.entry_prices["SPY"] == 520.0
        assert p.stop_order_ids["SPY"] == "stop-spy"

    def test_broker_unreachable_with_empty_state_does_not_crash(self, api, tmp_path):
        api.list_positions.side_effect = Exception("network error")

        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))

        assert p.entry_prices == {}
