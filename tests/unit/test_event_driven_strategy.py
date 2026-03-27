"""Unit tests for the event-driven positional strategy."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from trade_agent.analysis.event_scorer import EventEvidence, EventScorer
from trade_agent.analysis.liquidity import LiquidityProfile
from trade_agent.analysis.signals import CombinedSignal
from trade_agent.data.events import BulkDeal, CorporateEvent, EventType
from trade_agent.data.options_chain import OptionsChainSnapshot, StrikeData
from trade_agent.strategies.event_driven import EventDrivenStrategy


def _history(n: int = 60, price: float = 2500.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = price + np.cumsum(rng.normal(0.2, 1.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    high = close + 10
    low = close - 10
    vol = rng.integers(400_000, 600_000, n).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _liquid() -> LiquidityProfile:
    return LiquidityProfile(
        symbol="RELIANCE",
        avg_daily_volume=500_000,
        avg_daily_value_cr=125.0,
        adv_20=500_000,
        atr_pct=1.0,
        is_liquid=True,
        liquidity_tier="high",
        estimated_spread_pct=0.05,
        round_trip_cost_pct=0.15,
        max_position_size=50_000.0,
    )


def _illiquid() -> LiquidityProfile:
    liq = _liquid()
    liq.is_liquid = False
    liq.liquidity_tier = "illiquid"
    liq.max_position_size = 0.0
    return liq


def _fresh_event(event_type: EventType = EventType.BUYBACK, age: float = 15.0) -> CorporateEvent:
    filed = datetime.now(timezone.utc) - timedelta(minutes=age)
    return CorporateEvent(
        symbol="RELIANCE",
        exchange="NSE",
        event_type=event_type,
        headline=f"{event_type.value} approved",
        detail="",
        filing_time=filed,
        source_url="",
        event_id=f"{event_type.value}_{age}",
    )


def _bullish_evidence(conviction: float = 0.72) -> EventEvidence:
    evt = _fresh_event()
    return EventEvidence(
        symbol="RELIANCE",
        event_score=0.8,
        flow_score=0.5,
        options_score=0.6,
        conviction=conviction,
        bias="bullish",
        event_summary="Buyback at premium",
        flow_summary="FII buying ₹25 Cr",
        options_summary="PCR=1.3",
        key_risks=[],
        events=[evt],
        deals=[],
        options=None,
    )


class TestEventDrivenStrategy:
    def setup_method(self) -> None:
        self.strategy = EventDrivenStrategy(
            min_conviction=0.60,
            stop_loss_pct=0.03,
            take_profit_pct=0.09,
            max_trade_amount=10_000.0,
        )
        self.hist = _history()

    def test_long_on_strong_evidence(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(conviction=0.75),
            liquidity=_liquid(),
        )
        assert result.direction == "LONG"
        assert result.quantity > 0

    def test_skip_on_illiquid(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(),
            liquidity=_illiquid(),
        )
        assert result.direction == "SKIP"
        assert "lliquid" in result.rationale

    def test_skip_on_low_conviction(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(conviction=0.40),
            liquidity=_liquid(),
        )
        assert result.direction == "SKIP"

    def test_skip_with_no_evidence(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=None,
            liquidity=_liquid(),
        )
        assert result.direction == "SKIP"

    def test_skip_on_stale_event(self) -> None:
        # Event filed 95 minutes ago — past the 90-min freshness window
        stale_evt = _fresh_event(age=95.0)
        evidence = _bullish_evidence()
        evidence.events = [stale_evt]
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=evidence,
            liquidity=_liquid(),
        )
        assert result.direction == "SKIP"

    def test_stop_loss_below_entry(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(),
            liquidity=_liquid(),
        )
        if result.direction == "LONG":
            assert result.stop_loss < result.entry_price

    def test_take_profit_above_entry(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(),
            liquidity=_liquid(),
        )
        if result.direction == "LONG":
            assert result.take_profit > result.entry_price

    def test_minimum_rr_ratio(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(),
            liquidity=_liquid(),
        )
        if result.direction == "LONG":
            risk = result.entry_price - result.stop_loss
            reward = result.take_profit - result.entry_price
            assert (reward / risk) >= 2.0

    def test_should_trade_property(self) -> None:
        result = self.strategy.evaluate(
            "RELIANCE", None, self.hist, [],
            evidence=_bullish_evidence(conviction=0.75),
            liquidity=_liquid(),
        )
        assert result.should_trade is True
