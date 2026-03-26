"""Unit tests for signal aggregation."""

from __future__ import annotations

import pytest

from trade_agent.analysis.sentiment import AggregateSentiment
from trade_agent.analysis.signals import CombinedSignal, compute_signal
from trade_agent.analysis.technical import TechnicalSnapshot


def _make_bullish_technical() -> TechnicalSnapshot:
    return TechnicalSnapshot(
        last_close=2500.0,
        sma_20=2400.0,
        sma_50=2300.0,
        ema_20=2420.0,
        rsi_14=55.0,
        stoch_k=60.0,
        stoch_d=58.0,
        macd=5.0,
        macd_signal=3.0,
        macd_diff=2.0,
        bb_upper=2600.0,
        bb_lower=2400.0,
        bb_pct=0.5,
        atr_14=30.0,
        obv=1_000_000.0,
        volume_ratio=1.2,
        trend_direction="bullish",
        momentum_signal="neutral",
        volatility_signal="normal",
    )


def _make_bearish_technical() -> TechnicalSnapshot:
    snap = _make_bullish_technical()
    snap.trend_direction = "bearish"
    snap.macd_diff = -2.0
    snap.sma_20 = 2600.0  # price below SMA
    return snap


def _make_bullish_sentiment() -> AggregateSentiment:
    return AggregateSentiment(
        scores=[],
        avg_sentiment=0.6,
        avg_relevance=0.7,
        weighted_sentiment=0.6,
        article_count=5,
        bullish_count=4,
        bearish_count=1,
    )


def _make_neutral_sentiment() -> AggregateSentiment:
    return AggregateSentiment(
        scores=[],
        avg_sentiment=0.0,
        avg_relevance=0.0,
        weighted_sentiment=0.0,
        article_count=0,
        bullish_count=0,
        bearish_count=0,
    )


class TestComputeSignal:
    def test_returns_combined_signal(self) -> None:
        sig = compute_signal(
            "RELIANCE",
            _make_bullish_technical(),
            _make_bullish_sentiment(),
            nifty_change_pct=0.5,
        )
        assert isinstance(sig, CombinedSignal)

    def test_bullish_inputs_give_buy_action(self) -> None:
        sig = compute_signal(
            "TCS",
            _make_bullish_technical(),
            _make_bullish_sentiment(),
            nifty_change_pct=0.8,
            min_score_to_act=0.55,
        )
        assert sig.action == "BUY"
        assert sig.score > 0.55

    def test_score_clamped_between_zero_and_one(self) -> None:
        sig = compute_signal(
            "INFY",
            _make_bullish_technical(),
            _make_neutral_sentiment(),
        )
        assert 0.0 <= sig.score <= 1.0

    def test_neutral_sentiment_gives_hold_or_buy(self) -> None:
        sig = compute_signal(
            "WIPRO",
            _make_bullish_technical(),
            _make_neutral_sentiment(),
            nifty_change_pct=0.0,
        )
        assert sig.action in ("BUY", "HOLD")

    def test_to_dict_serialisable(self) -> None:
        sig = compute_signal("AXISBANK", _make_bullish_technical(), _make_neutral_sentiment())
        d = sig.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "AXISBANK"
        assert "action" in d
        assert "score" in d

    def test_confidence_field_valid(self) -> None:
        sig = compute_signal("ITC", _make_bullish_technical(), _make_bullish_sentiment())
        assert sig.confidence in ("high", "medium", "low")
