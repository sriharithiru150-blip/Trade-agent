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
    max_open_positions: Annotated[int, Field(ge=1, le=50)] = Field(
        5, description="Max simultaneous open positions"
    )
    stop_loss_pct: Annotated[float, Field(gt=0, lt=1)] = Field(
        0.02, description="Stop-loss as fraction of entry price"
    )
    take_profit_pct: Annotated[float, Field(gt=0, lt=1)] = Field(
        0.04, description="Take-profit as fraction of entry price"
    )

    # ── Watchlist ─────────────────────────────────────────────────────────────
    watchlist: str = Field(
        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,AXISBANK,BAJFINANCE,TATAMOTORS,ITC",
        description="Comma-separated NSE symbols",
    )

    # ── Agent Behaviour ───────────────────────────────────────────────────────
    analysis_interval_minutes: Annotated[int, Field(ge=1)] = Field(
        30, description="Minutes between analysis runs"
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
