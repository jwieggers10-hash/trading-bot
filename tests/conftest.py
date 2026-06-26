"""Global pytest configuration.

Autouse fixture: replace notifier.send with a no-op MagicMock for every test.
This prevents any test from reaching the real Telegram API regardless of what
credentials are present in .env.

Tests that want to assert on what would have been sent can request the
`telegram_mock` fixture directly.
"""
from unittest.mock import MagicMock

import pytest

import bot.telegram_notifier as _tg


@pytest.fixture(autouse=True)
def telegram_mock(monkeypatch):
    """Patch notifier.send so no test ever calls the real Telegram API."""
    mock = MagicMock(return_value=True)
    monkeypatch.setattr(_tg.notifier, "send", mock)
    return mock
