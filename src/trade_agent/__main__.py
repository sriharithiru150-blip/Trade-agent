"""Entry point — run the trading agent as a scheduled loop.

Usage:
    python -m trade_agent           # uses .env in current directory
    trade-agent                     # if installed via pip install -e .
"""

from __future__ import annotations

import signal
import sys
import time

import schedule

from trade_agent.config import get_settings
from trade_agent.utils.logger import get_logger, setup_logging
from trade_agent.utils.market_hours import (
    is_market_open,
    market_close_ist,
    now_ist,
    seconds_until_market_open,
)

log = get_logger(__name__)

_SHUTDOWN = False


def _handle_signal(signum: int, frame: object) -> None:
    global _SHUTDOWN
    log.info("shutdown_signal_received", signum=signum)
    _SHUTDOWN = True


def main() -> None:
    """Main entry point — parse environment, configure logging, and start loop."""
    from trade_agent.agent import TradingAgent  # lazy import for faster startup

    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)

    log.info(
        "trade_agent_starting",
        model=settings.claude_model,
        dry_run=settings.dry_run,
        intraday_mode=settings.intraday_mode,
        eod_square_off_minutes=settings.eod_square_off_minutes,
        watchlist=settings.get_watchlist(),
        interval_minutes=settings.analysis_interval_minutes,
    )

    if settings.dry_run:
        print("┌─────────────────────────────────────────────┐")
        print("│  PAPER TRADE MODE (DRY_RUN=true)            │")
        print("│  No real orders will be placed.             │")
        print("│  Set DRY_RUN=false in .env for live trading │")
        print("└─────────────────────────────────────────────┘")

    # Graceful shutdown on Ctrl-C / SIGTERM
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Build the agent ONCE — reused across every cycle (avoids reconnect overhead)
    agent = TradingAgent(settings)

    def _run_cycle() -> None:
        """Single analysis + execution cycle — called by the scheduler."""
        try:
            summary = agent.run_once()
            log.info("cycle_summary", text=summary)
            print("\n" + "=" * 72)
            print(summary)
            print("=" * 72 + "\n")
        except Exception as exc:
            log.error("cycle_failed", error=str(exc), exc_info=True)

        # End-of-day: square off all positions N min before market close
        now = now_ist()
        close = market_close_ist()
        minutes_to_close = (close - now).total_seconds() / 60
        if 0 < minutes_to_close <= settings.eod_square_off_minutes:
            log.info("eod_squareoff_trigger", minutes_to_close=round(minutes_to_close, 1))
            result = agent.square_off_all()
            log.info("eod_result", **result)

    interval = settings.analysis_interval_minutes
    schedule.every(interval).minutes.do(_run_cycle)
    log.info("scheduler_set", interval_minutes=interval)

    # Run immediately on startup if market is open
    if is_market_open():
        log.info("market_open_running_immediately")
        _run_cycle()
    else:
        wait_secs = seconds_until_market_open()
        log.info("waiting_for_market_open", wait_minutes=round(wait_secs / 60, 1))
        print(f"Market is closed. Next open in {wait_secs / 60:.0f} minutes.")

    # Poll every 5 seconds so scheduled jobs fire on time
    while not _SHUTDOWN:
        schedule.run_pending()
        time.sleep(5)

    log.info("trade_agent_stopped")
    sys.exit(0)


if __name__ == "__main__":
    main()
