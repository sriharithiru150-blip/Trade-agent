"""Unit tests for technical analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_agent.analysis.technical import (
    TechnicalSnapshot,
    _derive_momentum,
    _derive_trend,
    _derive_volatility,
    compute_technical_snapshot,
)


def _make_ohlcv(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with an upward trend."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.3, 1.5, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close - rng.normal(0, 0.5, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


class TestComputeTechnicalSnapshot:
    def test_returns_snapshot_instance(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        assert isinstance(snap, TechnicalSnapshot)

    def test_last_close_matches_dataframe(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        assert snap.last_close == pytest.approx(float(df["Close"].iloc[-1]))

    def test_rsi_in_valid_range(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        if snap.rsi_14 is not None:
            assert 0 <= snap.rsi_14 <= 100

    def test_bb_pct_in_valid_range(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        if snap.bb_pct is not None:
            assert -0.5 <= snap.bb_pct <= 1.5  # can exceed 0-1 in extreme moves

    def test_raises_on_too_few_rows(self) -> None:
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="at least 20"):
            compute_technical_snapshot(df)

    def test_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"Close": [100, 101]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_technical_snapshot(df)

    def test_trend_direction_is_valid_value(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        assert snap.trend_direction in ("bullish", "bearish", "neutral")

    def test_momentum_signal_is_valid_value(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        assert snap.momentum_signal in ("neutral", "overbought", "oversold")

    def test_to_dict_returns_dict(self) -> None:
        df = _make_ohlcv(60)
        snap = compute_technical_snapshot(df)
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "rsi_14" in d
        assert "trend_direction" in d


class TestDeriveSignals:
    def test_bullish_trend_above_smas(self) -> None:
        assert _derive_trend(110.0, sma_20=100.0, sma_50=95.0, macd_diff=0.5) == "bullish"

    def test_bearish_trend_below_smas(self) -> None:
        assert _derive_trend(90.0, sma_20=100.0, sma_50=95.0, macd_diff=-0.5) == "bearish"

    def test_oversold_momentum(self) -> None:
        assert _derive_momentum(25.0, None, None) == "oversold"

    def test_overbought_momentum(self) -> None:
        assert _derive_momentum(75.0, None, None) == "overbought"

    def test_neutral_momentum(self) -> None:
        assert _derive_momentum(50.0, None, None) == "neutral"

    def test_high_volatility(self) -> None:
        # bb_pct outside squeeze band (0.4-0.6), ATR 4% of close → high volatility
        assert _derive_volatility(0.8, 4.0, 100.0) == "high"

    def test_normal_volatility(self) -> None:
        # bb_pct outside squeeze band, ATR 1% of close → normal
        assert _derive_volatility(0.8, 1.0, 100.0) == "normal"
