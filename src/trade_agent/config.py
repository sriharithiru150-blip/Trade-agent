"""Configuration management — all settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AI ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    claude_model: str = Field("claude-sonnet-4-6", description="Claude model ID")

    # ── News ─────────────────────────────────────────────────────────────────
    news_api_key: str = Field("", description="NewsAPI.org key (optional)")

    # ── Broker ───────────────────────────────────────────────────────────────
    kite_api_key: str = Field("", description="Zerodha Kite API key")
    kite_api_secret: str = Field("", description="Zerodha Kite API secret")
    kite_access_token: str = Field("", description="Zerodha Kite access token")

    # ── Safety ───────────────────────────────────────────────────────────────
    dry_run: bool = Field(True, description="Paper-trade mode; no real orders placed")
    kill_switch: bool = Field(False, description="Emergency halt for all trading")
    max_trade_amount: Annotated[float, Field(gt=0)] = Field(
        10_000.0, description="Max INR per single trade"
    )
    initial_capital: Annotated[float, Field(ge=0)] = Field(
        0.0,
        description=(
            "Your starting capital in INR. When set, the portfolio tracker shows "
            "capital recovery progress and flags when you are trading on profits only. "
            "Set to 0 to disable tracking."
        ),
    )
    max_open_positions: Annotated[int, Field(ge=1, le=50)] = Field(
        5, description="Max simultaneous open positions"
    )
    max_sector_positions: Annotated[int, Field(ge=1, le=20)] = Field(
        2, description="Max simultaneous positions in any single sector"
    )
    stop_loss_pct: Annotated[float, Field(gt=0, lt=1)] = Field(
        0.02, description="Stop-loss as fraction of entry price (fallback when ATR unavailable)"
    )
    take_profit_pct: Annotated[float, Field(gt=0, lt=1)] = Field(
        0.04, description="Take-profit as fraction of entry price (fallback when ATR unavailable)"
    )

    # ── Intraday mode ─────────────────────────────────────────────────────────
    intraday_mode: bool = Field(
        False,
        description=(
            "When True: use 5-min ATR, tighter freshness windows (5/15/30 min), "
            "₹50Cr ADV floor, 2-min options staleness, and exit by EOD_SQUARE_OFF_MINUTES"
        ),
    )
    eod_square_off_minutes: Annotated[int, Field(ge=1, le=90)] = Field(
        30,
        description=(
            "Minutes before market close to trigger end-of-day square-off. "
            "30 min (15:00 IST) is recommended for intraday to avoid 15:15 rush."
        ),
    )
    intraday_adv_floor_cr: Annotated[float, Field(gt=0)] = Field(
        50.0,
        description=(
            "Minimum ADV in crores for intraday trading. "
            "Intraday needs tighter spreads than swing — ₹50Cr vs ₹20Cr default."
        ),
    )
    intraday_options_staleness_minutes: Annotated[float, Field(gt=0)] = Field(
        2.0,
        description=(
            "Options snapshot age (minutes) above which it's considered stale in intraday mode. "
            "NSE refreshes the public feed every 3-5 min; for intraday 2 min is the effective limit."
        ),
    )

    # ── Watchlist ─────────────────────────────────────────────────────────────
    watchlist: str = Field(
        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,AXISBANK,BAJFINANCE,TATAMOTORS,ITC",
        description="Comma-separated NSE symbols",
    )

    # ── Agent Behaviour ───────────────────────────────────────────────────────
    analysis_interval_minutes: Annotated[int, Field(ge=1)] = Field(
        5, description="Minutes between analysis runs"
    )
    min_signal_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        0.65, description="Minimum combined score to open a trade"
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Logging verbosity")
    log_file: str = Field("logs/trade_agent.log", description="Log file path")

    @field_validator("watchlist")
    @classmethod
    def _validate_watchlist(cls, v: str) -> str:
        symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
        if not symbols:
            raise ValueError("WATCHLIST must contain at least one symbol")
        return ",".join(symbols)

    def get_watchlist(self) -> list[str]:
        """Return the watchlist as a list of clean uppercase NSE symbols."""
        return [s.strip().upper() for s in self.watchlist.split(",") if s.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()  # type: ignore[call-arg]
