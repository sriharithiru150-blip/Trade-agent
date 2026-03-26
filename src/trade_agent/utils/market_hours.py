"""IST market hours helpers for NSE / BSE."""

from __future__ import annotations

from datetime import datetime, time, timezone, timedelta

# Indian Standard Time offset
_IST = timezone(timedelta(hours=5, minutes=30))

# NSE/BSE trading session
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)

# Pre-open session (for reference, we don't trade here)
_PRE_OPEN_START = time(9, 0)
_PRE_OPEN_END = time(9, 8)


def now_ist() -> datetime:
    """Return the current datetime in IST."""
    return datetime.now(_IST)


def is_market_open() -> bool:
    """Return True if NSE/BSE is currently in its regular trading session.

    This checks day-of-week and session time. It does NOT account for
    exchange holidays — add a holiday calendar for production use.
    """
    now = now_ist()
    # Monday=0 … Friday=4
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return _MARKET_OPEN <= current_time <= _MARKET_CLOSE


def is_pre_open() -> bool:
    """Return True if we are in the NSE pre-open session (09:00–09:08 IST)."""
    now = now_ist()
    if now.weekday() >= 5:
        return False
    return _PRE_OPEN_START <= now.time() <= _PRE_OPEN_END


def seconds_until_market_open() -> float:
    """Return seconds until the next market open.

    Returns 0 if the market is already open.
    """
    if is_market_open():
        return 0.0
    now = now_ist()
    # Build today's open datetime in IST
    open_today = now.replace(
        hour=_MARKET_OPEN.hour,
        minute=_MARKET_OPEN.minute,
        second=0,
        microsecond=0,
    )
    if now >= open_today:
        # After close — target tomorrow (skip weekend)
        days_ahead = 1
        while (now.weekday() + days_ahead) % 7 >= 5:
            days_ahead += 1
        open_today = open_today + timedelta(days=days_ahead)
    elif now.weekday() >= 5:
        # Weekend — find next Monday
        days_ahead = 7 - now.weekday()
        open_today = open_today + timedelta(days=days_ahead)
    delta = open_today - now
    return max(0.0, delta.total_seconds())


def market_close_ist() -> datetime:
    """Return today's market close datetime in IST."""
    now = now_ist()
    return now.replace(
        hour=_MARKET_CLOSE.hour,
        minute=_MARKET_CLOSE.minute,
        second=0,
        microsecond=0,
    )
