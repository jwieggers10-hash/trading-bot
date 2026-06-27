"""Tests for twice-daily P&L notification scheduling.

Covers:
  - _pnl_notify_due() fires at 09:00 and 22:00 Europe/Amsterdam only
  - Idempotency: same (date, hour) pair is not sent twice
  - CET (UTC+1) and CEST (UTC+2) offsets are handled transparently
  - portfolio.send_pnl_notification() calls notifier.daily_pnl correctly
"""
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from bot.main import _pnl_notify_due
from bot.portfolio import Portfolio
import bot.telegram_notifier as _tg

_AMS = ZoneInfo("Europe/Amsterdam")


def _ams(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_AMS)


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


# ---------------------------------------------------------------------------
# _pnl_notify_due — scheduling logic
# ---------------------------------------------------------------------------

class TestPnlNotifyDue:
    def test_fires_at_09_when_not_yet_notified(self):
        assert _pnl_notify_due(_ams(2026, 6, 27, 9), set()) is True

    def test_fires_at_22_when_not_yet_notified(self):
        assert _pnl_notify_due(_ams(2026, 6, 27, 22), set()) is True

    def test_does_not_fire_at_09_when_already_notified(self):
        notified = {(date(2026, 6, 27), 9)}
        assert _pnl_notify_due(_ams(2026, 6, 27, 9, 45), notified) is False

    def test_does_not_fire_at_22_when_already_notified(self):
        notified = {(date(2026, 6, 27), 22)}
        assert _pnl_notify_due(_ams(2026, 6, 27, 22, 30), notified) is False

    @pytest.mark.parametrize("hour", [0, 1, 8, 10, 12, 15, 21, 23])
    def test_does_not_fire_at_non_scheduled_hours(self, hour):
        assert _pnl_notify_due(_ams(2026, 6, 27, hour), set()) is False

    def test_fires_again_next_day_after_previous_day_notified(self):
        notified = {(date(2026, 6, 27), 9)}
        assert _pnl_notify_due(_ams(2026, 6, 28, 9), notified) is True

    def test_morning_and_evening_are_independent(self):
        # Morning sent — evening must still fire
        notified = {(date(2026, 6, 27), 9)}
        assert _pnl_notify_due(_ams(2026, 6, 27, 22), notified) is True

    def test_evening_sent_does_not_block_next_morning(self):
        notified = {(date(2026, 6, 27), 22)}
        assert _pnl_notify_due(_ams(2026, 6, 28, 9), notified) is True

    def test_fires_within_the_same_hour_until_added(self):
        # Simulate two loop ticks in the same minute before the set is updated
        now = _ams(2026, 6, 27, 9, 0)
        notified: set = set()
        assert _pnl_notify_due(now, notified) is True
        notified.add((now.date(), now.hour))
        assert _pnl_notify_due(now, notified) is False

    # ------------------------------------------------------------------
    # DST awareness: Amsterdam switches between CET (UTC+1) and CEST (UTC+2)
    # The function must use the Amsterdam hour, not the UTC hour.
    # ------------------------------------------------------------------

    def test_cest_summer_offset_is_utc_plus2(self):
        # 2026-06-27 is summer → CEST = UTC+2
        now = _ams(2026, 6, 27, 9, 0)
        assert now.utcoffset().total_seconds() == 7200

    def test_cet_winter_offset_is_utc_plus1(self):
        # 2026-01-01 is winter → CET = UTC+1
        now = _ams(2026, 1, 1, 9, 0)
        assert now.utcoffset().total_seconds() == 3600

    def test_fires_at_09_ams_in_summer_cest(self):
        # 09:00 AMS CEST = 07:00 UTC; function must use AMS hour (9), not UTC hour (7)
        now = _ams(2026, 6, 27, 9, 0)
        assert _pnl_notify_due(now, set()) is True

    def test_fires_at_09_ams_in_winter_cet(self):
        # 09:00 AMS CET = 08:00 UTC; function must use AMS hour (9), not UTC hour (8)
        now = _ams(2026, 1, 1, 9, 0)
        assert _pnl_notify_due(now, set()) is True

    def test_fires_at_22_ams_in_summer_cest(self):
        now = _ams(2026, 6, 27, 22, 0)
        assert _pnl_notify_due(now, set()) is True

    def test_fires_at_22_ams_in_winter_cet(self):
        now = _ams(2026, 1, 1, 22, 0)
        assert _pnl_notify_due(now, set()) is True


# ---------------------------------------------------------------------------
# Portfolio.send_pnl_notification
# ---------------------------------------------------------------------------

class TestSendPnlNotification:
    def test_calls_notifier_daily_pnl_with_correct_values(self, api, portfolio):
        account = MagicMock()
        account.equity = "200123.45"
        account.last_equity = "200000.00"
        api.get_account.return_value = account

        with patch.object(_tg.notifier, "daily_pnl") as mock_pnl:
            portfolio.send_pnl_notification()

        mock_pnl.assert_called_once()
        _, pnl, equity = mock_pnl.call_args[0]
        assert pnl == pytest.approx(123.45)
        assert equity == pytest.approx(200123.45)

    def test_date_string_format_is_iso(self, api, portfolio):
        account = MagicMock()
        account.equity = "100000.00"
        account.last_equity = "100000.00"
        api.get_account.return_value = account

        with patch.object(_tg.notifier, "daily_pnl") as mock_pnl:
            portfolio.send_pnl_notification()

        date_str = mock_pnl.call_args[0][0]
        # ISO 8601: YYYY-MM-DD
        assert len(date_str) == 10
        assert date_str[4] == "-" and date_str[7] == "-"

    def test_does_not_raise_on_api_failure(self, api, portfolio):
        api.get_account.side_effect = Exception("Alpaca unreachable")
        portfolio.send_pnl_notification()  # must not propagate

    def test_does_not_call_notifier_on_api_failure(self, api, portfolio):
        api.get_account.side_effect = Exception("timeout")

        with patch.object(_tg.notifier, "daily_pnl") as mock_pnl:
            portfolio.send_pnl_notification()

        mock_pnl.assert_not_called()

    def test_negative_pnl_handled_correctly(self, api, portfolio):
        account = MagicMock()
        account.equity = "199500.00"
        account.last_equity = "200000.00"
        api.get_account.return_value = account

        with patch.object(_tg.notifier, "daily_pnl") as mock_pnl:
            portfolio.send_pnl_notification()

        _, pnl, _ = mock_pnl.call_args[0]
        assert pnl == pytest.approx(-500.0)
