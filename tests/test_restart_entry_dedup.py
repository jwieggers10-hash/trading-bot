"""Tests proving a restart cannot cause a duplicate entry for the same candle.

Context: main.py's per-strategy interval timer (`last_mean_rev` etc.) is an
in-memory float that resets to 0 on every process start, so a restart
immediately re-evaluates every symbol's latest candle — regardless of how
much of that candle's interval had already elapsed before the restart.
position_state() is a live broker query, so a restart alone can't duplicate
an entry once the position is visible on Alpaca. But there is a real window
between submit_order() returning and that fill becoming visible via
list_positions(): a crash inside that window followed by an immediate
restart would see the symbol as still "flat" and, without a guard, could
submit a second market order for the same signal — doubling the position.

The fix: Portfolio persists the (candle_ts, direction) of the last entry
*attempt* per symbol to disk, written before submit_order() is ever called.
entry_signal_already_processed() lets the entry decision short-circuit even
across a process restart that reloads state from the same file.

Test structure:
  TestPortfolioEntrySignalGuard — unit tests for the Portfolio methods
  TestWithinProcessDedup        — run() called twice for the same candle
  TestRestartDedup              — simulates a crash + restart via two
                                   independent Portfolio instances sharing
                                   one state file
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.portfolio import Portfolio
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.mean_reversion_v2 import MeanReversionV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(id: str = "mkt-1", filled_avg_price="520.00", filled_qty="136",
                 status="filled", stop_leg_id="stop-1", stop_side="sell"):
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


def _make_df_bars(n: int = 25, last_ts: str = "2026-01-01 10:00", base_price: float = 520.0):
    """Bars ending exactly at *last_ts* — this is the "candle" run() will key on."""
    rng = np.random.default_rng(42)
    prices = base_price + rng.normal(0, 0.5, n).cumsum()
    end = pd.Timestamp(last_ts)
    idx = pd.date_range(end=end, periods=n, freq="15min")
    return pd.DataFrame(
        {"open": prices, "high": prices + 0.3, "low": prices - 0.3,
         "close": prices, "volume": 1_000_000},
        index=idx,
    )


_LONG_SIGNAL = {"price": 514.0, "sma": 522.0, "upper": 526.0, "lower": 515.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    m = MagicMock()
    m.list_positions.return_value = []   # flat by default
    m.list_orders.return_value = []
    return m


@pytest.fixture
def rm():
    m = MagicMock()
    m.get_account_equity.return_value = 400_000.0
    m.calculate_atr.return_value = 1.5
    m.integer_position_size.return_value = 136
    m.stop_price.return_value = 518.0
    return m


@pytest.fixture
def rm_v2():
    m = MagicMock()
    m.get_account_equity.return_value = 400_000.0
    m.calculate_atr.return_value = 1.5
    m.v2_position_size.return_value = 136
    m.stop_price.return_value = 518.0
    return m


# ---------------------------------------------------------------------------
# TestPortfolioEntrySignalGuard — unit tests for the Portfolio methods
# ---------------------------------------------------------------------------

class TestPortfolioEntrySignalGuard:
    def test_not_processed_by_default(self, api, tmp_path):
        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))
        assert p.entry_signal_already_processed("SPY", "2026-01-01T10:00", "long") is False

    def test_processed_after_recording(self, api, tmp_path):
        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))
        p.record_entry_signal("SPY", "2026-01-01T10:00", "long")
        assert p.entry_signal_already_processed("SPY", "2026-01-01T10:00", "long") is True

    def test_different_candle_not_blocked(self, api, tmp_path):
        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))
        p.record_entry_signal("SPY", "2026-01-01T10:00", "long")
        assert p.entry_signal_already_processed("SPY", "2026-01-01T10:15", "long") is False

    def test_different_direction_same_candle_not_blocked(self, api, tmp_path):
        """A genuine reversal (exit then opposite entry) within the same
        candle must still be allowed — only the exact (candle, direction)
        that was already attempted is blocked."""
        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))
        p.record_entry_signal("SPY", "2026-01-01T10:00", "long")
        assert p.entry_signal_already_processed("SPY", "2026-01-01T10:00", "short") is False

    def test_different_symbol_not_blocked(self, api, tmp_path):
        with patch.object(Portfolio, "_init_log_files"):
            p = Portfolio(api, state_file=str(tmp_path / "state.json"))
        p.record_entry_signal("SPY", "2026-01-01T10:00", "long")
        assert p.entry_signal_already_processed("QQQ", "2026-01-01T10:00", "long") is False

    def test_persists_to_disk_and_reloads(self, api, tmp_path):
        """The core restart guarantee: a fresh Portfolio instance reading the
        same state file must see the previously recorded attempt."""
        state_file = str(tmp_path / "state.json")
        with patch.object(Portfolio, "_init_log_files"):
            p1 = Portfolio(api, state_file=state_file)
        p1.record_entry_signal("SPY", "2026-01-01T10:00", "long")

        with patch.object(Portfolio, "_init_log_files"):
            p2 = Portfolio(api, state_file=state_file)  # simulates a restart

        assert p2.entry_signal_already_processed("SPY", "2026-01-01T10:00", "long") is True


# ---------------------------------------------------------------------------
# TestWithinProcessDedup — run() called twice for the same candle
# ---------------------------------------------------------------------------

class TestWithinProcessDedup:
    def test_second_run_same_candle_does_not_resubmit(self, api, rm, tmp_path):
        with patch.object(Portfolio, "_init_log_files"):
            portfolio = Portfolio(api, state_file=str(tmp_path / "state.json"))
        strategy = MeanReversionStrategy(api, rm, portfolio)

        bars = _make_df_bars(last_ts="2026-01-01 10:00")
        api.submit_order.return_value = _make_order()

        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")   # first tick within the candle: enters
            strategy.run("SPY")   # 60s later, still same candle, still flat
                                   # (position mock unchanged) — must NOT resubmit

        assert api.submit_order.call_count == 1

    def test_next_candle_is_allowed(self, api, rm, tmp_path):
        """A genuinely new candle must still be tradeable after a position
        was closed through the bot's own normal path (SMA exit / hard stop —
        both call Portfolio.clear_position()).

        This is deliberately NOT simulated by leaving api.list_positions() at
        [] with the local entry still tracked: that combination is
        indistinguishable from an external exit (e.g. the broker-side stop
        firing while the bot wasn't looking), and Portfolio.recover_external_exit()
        now — correctly, by design — treats any such transition as a stop-out
        and applies the re-entry cooldown (see TestExternalExitDefersReentry
        below for the dedicated test of that behavior). This test simulates a
        clean, already-closed position instead, so it actually exercises what
        it's meant to: the entry_signal_already_processed() guard allowing a
        fresh candle through, not the external-exit/cooldown path.
        """
        with patch.object(Portfolio, "_init_log_files"):
            portfolio = Portfolio(api, state_file=str(tmp_path / "state.json"))
        strategy = MeanReversionStrategy(api, rm, portfolio)

        api.submit_order.return_value = _make_order()

        bars1 = _make_df_bars(last_ts="2026-01-01 10:00")
        with patch.object(strategy, "_get_bars", return_value=bars1), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        # Position closed through the bot's own normal path (not externally) —
        # broker is flat AND local tracking has already been cleared, so the
        # next tick sees a clean flat state rather than an unexplained one.
        portfolio.clear_position("SPY")
        api.list_positions.return_value = []
        bars2 = _make_df_bars(last_ts="2026-01-01 10:15")
        with patch.object(strategy, "_get_bars", return_value=bars2), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        assert api.submit_order.call_count == 2


# ---------------------------------------------------------------------------
# TestRestartDedup — simulates a crash + restart via two independent
# Portfolio instances that share one on-disk state file.
# ---------------------------------------------------------------------------

class TestRestartDedup:
    def test_restart_after_entry_attempt_does_not_duplicate(self, api, rm, tmp_path):
        """
        Reproduces the exact race: the entry order is submitted and filled on
        Alpaca's side, but the process crashes before that fill is queried
        again — list_positions() still returns [] on the very next call (as
        it would if queried again immediately after a fast restart, before
        Alpaca's fill has propagated). Without the guard, main.py's reset
        interval timer would let the restarted process see "flat" + a still
        valid signal and submit a second market order for the same candle.
        """
        state_file = str(tmp_path / "state.json")

        with patch.object(Portfolio, "_init_log_files"):
            portfolio1 = Portfolio(api, state_file=state_file)
        strategy1 = MeanReversionStrategy(api, rm, portfolio1)

        bars = _make_df_bars(last_ts="2026-01-01 10:00")
        api.submit_order.return_value = _make_order()
        api.list_positions.return_value = []  # still "flat" from the broker's perspective

        with patch.object(strategy1, "_get_bars", return_value=bars), \
             patch.object(strategy1, "_signals", return_value=_LONG_SIGNAL):
            strategy1.run("SPY")

        assert api.submit_order.call_count == 1

        # --- simulate crash + restart: fresh Portfolio/strategy instances,
        # loading state from the same file, broker still reports "flat" ---
        with patch.object(Portfolio, "_init_log_files"):
            portfolio2 = Portfolio(api, state_file=state_file)
        strategy2 = MeanReversionStrategy(api, rm, portfolio2)

        with patch.object(strategy2, "_get_bars", return_value=bars), \
             patch.object(strategy2, "_signals", return_value=_LONG_SIGNAL):
            strategy2.run("SPY")

        # No second order — the restart-safe guard blocked the re-attempt
        assert api.submit_order.call_count == 1

    def test_restart_on_a_new_candle_still_trades(self, api, rm, tmp_path):
        """The guard must not become a permanent block — a genuinely new
        candle after restart must still be tradeable.

        Like test_next_candle_is_allowed above, the position must be closed
        through the bot's own normal path before "restart" so the persisted
        state file has no stale open-position record for SPY. If it did,
        portfolio2's startup reconciliation would (correctly) treat a
        tracked-but-broker-flat SPY as an external exit / stop-out and defer
        re-entry via the cooldown — a different, deliberately-tested behavior
        (see TestExternalExitDefersReentry), not what this test is about.
        """
        state_file = str(tmp_path / "state.json")

        with patch.object(Portfolio, "_init_log_files"):
            portfolio1 = Portfolio(api, state_file=state_file)
        strategy1 = MeanReversionStrategy(api, rm, portfolio1)

        bars1 = _make_df_bars(last_ts="2026-01-01 10:00")
        api.submit_order.return_value = _make_order()
        api.list_positions.return_value = []

        with patch.object(strategy1, "_get_bars", return_value=bars1), \
             patch.object(strategy1, "_signals", return_value=_LONG_SIGNAL):
            strategy1.run("SPY")

        # Position closed through the bot's own normal path before "restart".
        portfolio1.clear_position("SPY")

        with patch.object(Portfolio, "_init_log_files"):
            portfolio2 = Portfolio(api, state_file=state_file)
        strategy2 = MeanReversionStrategy(api, rm, portfolio2)

        bars2 = _make_df_bars(last_ts="2026-01-01 10:15")   # new candle
        with patch.object(strategy2, "_get_bars", return_value=bars2), \
             patch.object(strategy2, "_signals", return_value=_LONG_SIGNAL):
            strategy2.run("SPY")

        assert api.submit_order.call_count == 2

    def test_v2_restart_after_entry_attempt_does_not_duplicate(self, api, rm_v2, tmp_path):
        """Same guarantee for MeanReversionV2."""
        state_file = str(tmp_path / "state.json")

        with patch.object(Portfolio, "_init_log_files"):
            portfolio1 = Portfolio(api, state_file=state_file)
        strategy1 = MeanReversionV2(api, rm_v2, portfolio1)

        bars = _make_df_bars(last_ts="2026-01-01 10:00")
        api.submit_order.return_value = _make_order()
        api.list_positions.return_value = []

        with patch.object(strategy1, "_get_bars", return_value=bars), \
             patch.object(strategy1, "_signals", return_value=_LONG_SIGNAL):
            strategy1.run("SPY")

        assert api.submit_order.call_count == 1

        with patch.object(Portfolio, "_init_log_files"):
            portfolio2 = Portfolio(api, state_file=state_file)
        strategy2 = MeanReversionV2(api, rm_v2, portfolio2)

        with patch.object(strategy2, "_get_bars", return_value=bars), \
             patch.object(strategy2, "_signals", return_value=_LONG_SIGNAL):
            strategy2.run("SPY")

        assert api.submit_order.call_count == 1

    def test_signal_recorded_before_order_submitted(self, api, rm, tmp_path):
        """The guard must be persisted BEFORE submit_order() is called —
        otherwise a crash during/after submit_order() but before the flag is
        written would reopen the exact race the guard exists to close."""
        with patch.object(Portfolio, "_init_log_files"):
            portfolio = Portfolio(api, state_file=str(tmp_path / "state.json"))
        strategy = MeanReversionStrategy(api, rm, portfolio)

        bars = _make_df_bars(last_ts="2026-01-01 10:00")
        candle_ts = bars.index[-1]

        def check_flag_set_before_submit(**kwargs):
            assert portfolio.entry_signal_already_processed("SPY", candle_ts, "long") is True
            return _make_order()

        api.submit_order.side_effect = check_flag_set_before_submit

        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        api.submit_order.assert_called_once()


# ---------------------------------------------------------------------------
# TestExternalExitDefersReentry — the deliberate design decision behind
# Portfolio.recover_external_exit()
#
# Decision: when a symbol is tracked locally as open but the broker reports
# it flat with no local record of the bot's own close() path having run, the
# safest assumption is that the broker-side hard stop fired externally (it
# fills the instant price crosses it, faster than the bot's own poll-interval
# check can ever react — see recover_external_exit()'s docstring in
# portfolio.py). recover_external_exit() therefore always calls
# record_stop_out() (starting the standard STOP_COOLDOWN_SECONDS cooldown)
# before clearing local state, and run() returns immediately after — it does
# NOT fall through to same-tick re-entry evaluation.
#
# This was evaluated against the alternative (fall through to same-tick
# entry evaluation) and rejected: even without the early return, the
# just-started cooldown would block re-entry anyway via in_stop_cooldown(),
# so removing the return buys nothing while making the control flow harder
# to reason about. Treating every unexplained external flat-transition as a
# stop-out — never as a silent "someone closed it, no cooldown needed" — is
# consistent with how every other stop-out in this codebase is handled, and
# errs toward not immediately re-risking capital on a symbol that may have
# just been stopped out by real, adverse price action.
# ---------------------------------------------------------------------------

class TestExternalExitDefersReentry:
    def test_external_exit_starts_cooldown_and_blocks_same_tick_reentry(self, api, rm, tmp_path):
        """A symbol tracked locally as open, found flat on the broker with no
        local record of a normal close, must not re-enter on the very same
        tick even if the current candle's signal would otherwise trigger one."""
        with patch.object(Portfolio, "_init_log_files"):
            portfolio = Portfolio(api, state_file=str(tmp_path / "state.json"))
        strategy = MeanReversionStrategy(api, rm, portfolio)

        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)
        api.list_positions.return_value = []  # broker shows flat — never told the bot why

        bars = _make_df_bars(last_ts="2026-01-01 10:00")
        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        api.submit_order.assert_not_called()
        assert portfolio.in_stop_cooldown("SPY") is True
        assert "SPY" not in portfolio.entry_prices

    def test_external_exit_blocks_reentry_on_the_very_next_candle_too(self, api, rm, tmp_path):
        """The deferral isn't just same-tick — the cooldown persists across
        the next candle as well, exactly like a real stop-out."""
        with patch.object(Portfolio, "_init_log_files"):
            portfolio = Portfolio(api, state_file=str(tmp_path / "state.json"))
        strategy = MeanReversionStrategy(api, rm, portfolio)

        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)
        api.list_positions.return_value = []

        bars1 = _make_df_bars(last_ts="2026-01-01 10:00")
        with patch.object(strategy, "_get_bars", return_value=bars1), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        bars2 = _make_df_bars(last_ts="2026-01-01 10:15")
        with patch.object(strategy, "_get_bars", return_value=bars2), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        # Cooldown (60 min by default) spans well past one 15-min candle.
        api.submit_order.assert_not_called()

    def test_v2_external_exit_also_defers_reentry(self, api, rm_v2, tmp_path):
        """Same guarantee for MeanReversionV2."""
        with patch.object(Portfolio, "_init_log_files"):
            portfolio = Portfolio(api, state_file=str(tmp_path / "state.json"))
        strategy = MeanReversionV2(api, rm_v2, portfolio)

        portfolio.record_entry("SPY", "long", 520.0, 35, "stop-spy", 518.0)
        api.list_positions.return_value = []

        bars = _make_df_bars(last_ts="2026-01-01 10:00")
        with patch.object(strategy, "_get_bars", return_value=bars), \
             patch.object(strategy, "_signals", return_value=_LONG_SIGNAL):
            strategy.run("SPY")

        api.submit_order.assert_not_called()
        assert portfolio.in_stop_cooldown("SPY") is True
