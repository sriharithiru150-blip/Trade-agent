"""Unit tests for the liquidity filter and slippage model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_agent.analysis.liquidity import (
    LiquidityProfile,
    SlippageModel,
    _calc_round_trip_cost,
    compute_liquidity_profile,
    compute_slippage,
)


def _make_history(n: int = 30, avg_price: float = 2500.0, avg_vol: int = 500_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = avg_price + rng.normal(0, 20, n)
    vol = rng.integers(int(avg_vol * 0.7), int(avg_vol * 1.3), n).astype(float)
    high = close + rng.uniform(10, 30, n)
    low = close - rng.uniform(10, 30, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


class TestComputeLiquidityProfile:
    def test_high_liquid_stock(self) -> None:
        # 500k shares × ₹2500 = ₹125 Cr ADV → "high" tier
        hist = _make_history(avg_price=2500.0, avg_vol=500_000)
        prof = compute_liquidity_profile("RELIANCE", hist, 2500.0, 10_000.0)
        assert prof.liquidity_tier == "high"
        assert prof.is_liquid is True

    def test_illiquid_stock(self) -> None:
        # 500 shares × ₹100 = ₹0.05 Cr ADV → "illiquid"
        hist = _make_history(avg_price=100.0, avg_vol=500)
        prof = compute_liquidity_profile("SMALL", hist, 100.0, 10_000.0)
        assert prof.liquidity_tier == "illiquid"
        assert prof.is_liquid is False

    def test_max_position_size_capped_for_illiquid(self) -> None:
        hist = _make_history(avg_price=100.0, avg_vol=500)
        prof = compute_liquidity_profile("SMALL", hist, 100.0, 10_000.0)
        assert prof.max_position_size == 0.0

    def test_returns_profile_instance(self) -> None:
        hist = _make_history()
        prof = compute_liquidity_profile("TCS", hist, 2500.0)
        assert isinstance(prof, LiquidityProfile)

    def test_spread_pct_positive(self) -> None:
        hist = _make_history()
        prof = compute_liquidity_profile("TCS", hist, 2500.0)
        assert prof.estimated_spread_pct > 0

    def test_too_few_rows_returns_illiquid(self) -> None:
        hist = _make_history(n=3)
        prof = compute_liquidity_profile("X", hist, 100.0)
        assert prof.is_liquid is False


class TestComputeSlippage:
    def _liquid_profile(self) -> LiquidityProfile:
        return LiquidityProfile(
            symbol="RELIANCE",
            avg_daily_volume=500_000,
            avg_daily_value_cr=125.0,
            adv_20=500_000,
            atr_pct=1.5,
            is_liquid=True,
            liquidity_tier="high",
            estimated_spread_pct=0.05,
            round_trip_cost_pct=0.15,
            max_position_size=50_000.0,
        )

    def test_returns_slippage_model(self) -> None:
        liq = self._liquid_profile()
        sm = compute_slippage(2500.0, 4, liq, stop_loss_raw=2450.0)
        assert isinstance(sm, SlippageModel)

    def test_total_cost_positive(self) -> None:
        liq = self._liquid_profile()
        sm = compute_slippage(2500.0, 4, liq, stop_loss_raw=2450.0)
        assert sm.total_cost_inr > 0

    def test_break_even_pct_small_for_liquid(self) -> None:
        liq = self._liquid_profile()
        sm = compute_slippage(2500.0, 4, liq, stop_loss_raw=2450.0)
        # All-in round-trip for a liquid large-cap should be < 0.5%
        assert sm.break_even_pct < 0.005

    def test_adjusted_stop_loss_below_raw(self) -> None:
        liq = self._liquid_profile()
        sm = compute_slippage(2500.0, 4, liq, stop_loss_raw=2450.0)
        assert sm.adjusted_stop_loss <= 2450.0


class TestRoundTripCost:
    def test_higher_value_means_lower_pct(self) -> None:
        """Brokerage flat fee means higher value = lower cost %."""
        small = _calc_round_trip_cost(1_000.0, 0.05, 0.1)
        large = _calc_round_trip_cost(100_000.0, 0.05, 0.1)
        assert small > large

    def test_returns_positive(self) -> None:
        assert _calc_round_trip_cost(10_000.0, 0.05, 0.1) > 0
