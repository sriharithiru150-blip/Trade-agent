"""Unit tests for market hours utilities."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from trade_agent.utils.market_hours import (
    _IST,
    is_market_open,
    is_pre_open,
    now_ist,
    seconds_until_market_open,
)


def _make_ist(weekday: int, hour: int, minute: int) -> datetime:
    """Build a datetime in IST for a given weekday and time.

    weekday: 0=Mon … 6=Sun
    """
    # 2024-01-01 is a Monday
    base = datetime(2024, 1, 1, tzinfo=_IST)
    days = (weekday - base.weekday()) % 7
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days)


class TestIsMarketOpen:
    def test_open_during_session(self) -> None:
        dt = _make_ist(0, 10, 0)  # Monday 10:00
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is True

    def test_closed_before_session(self) -> None:
        dt = _make_ist(0, 9, 10)  # Monday 09:10 — before 09:15
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is False

    def test_closed_after_session(self) -> None:
        dt = _make_ist(0, 15, 31)  # Monday 15:31
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is False

    def test_closed_on_saturday(self) -> None:
        dt = _make_ist(5, 10, 0)  # Saturday 10:00
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is False

    def test_closed_on_sunday(self) -> None:
        dt = _make_ist(6, 10, 0)  # Sunday
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is False

    def test_open_at_exactly_open_time(self) -> None:
        dt = _make_ist(1, 9, 15)  # Tuesday 09:15
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is True

    def test_open_at_exactly_close_time(self) -> None:
        dt = _make_ist(1, 15, 30)  # Tuesday 15:30
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_market_open() is True


class TestIsPreOpen:
    def test_pre_open_in_window(self) -> None:
        dt = _make_ist(0, 9, 5)  # Monday 09:05
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_pre_open() is True

    def test_not_pre_open_after_window(self) -> None:
        dt = _make_ist(0, 9, 15)
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_pre_open() is False

    def test_not_pre_open_on_weekend(self) -> None:
        dt = _make_ist(5, 9, 5)
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert is_pre_open() is False


class TestSecondsUntilMarketOpen:
    def test_returns_zero_when_open(self) -> None:
        dt = _make_ist(0, 10, 0)
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            assert seconds_until_market_open() == 0.0

    def test_positive_when_closed(self) -> None:
        dt = _make_ist(0, 8, 0)  # Before open
        with patch("trade_agent.utils.market_hours.now_ist", return_value=dt):
            secs = seconds_until_market_open()
            assert secs > 0
            # Should be approximately 75 minutes
            assert 60 * 60 < secs < 2 * 60 * 60
