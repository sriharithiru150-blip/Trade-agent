"""Unit tests for the options chain analyser."""

from __future__ import annotations

import pytest

from trade_agent.data.options_chain import (
    OptionsChainSnapshot,
    StrikeData,
    _compute_max_pain,
)


def _make_strikes(
    base_price: float = 2500.0,
    n: int = 5,
) -> list[StrikeData]:
    """Generate synthetic strike data around base_price."""
    strikes = []
    step = base_price * 0.02
    for i in range(n):
        k = base_price + (i - n // 2) * step
        # Puts heavier near ATM and below; calls heavier above ATM
        call_oi = int(50_000 * (1 + i * 0.3))
        put_oi = int(80_000 * (1 + (n - i) * 0.2))
        strikes.append(StrikeData(
            strike=round(k, 0),
            call_oi=call_oi,
            call_oi_change=call_oi // 10,
            call_iv=18.0 + i * 0.5,
            call_ltp=max(1.0, (base_price - k) * 0.8),
            put_oi=put_oi,
            put_oi_change=put_oi // 15,
            put_iv=20.0 + (n - i) * 0.5,
            put_ltp=max(1.0, (k - base_price) * 0.8),
        ))
    return strikes


class TestStrikeData:
    def test_pcr_with_positive_call_oi(self) -> None:
        s = StrikeData(
            strike=2500, call_oi=100_000, call_oi_change=5000,
            call_iv=18.0, call_ltp=50.0,
            put_oi=130_000, put_oi_change=8000,
            put_iv=20.0, put_ltp=60.0,
        )
        assert s.pcr == pytest.approx(1.3)

    def test_pcr_none_when_call_oi_zero(self) -> None:
        s = StrikeData(
            strike=2500, call_oi=0, call_oi_change=0,
            call_iv=0.0, call_ltp=0.0,
            put_oi=10_000, put_oi_change=0,
            put_iv=20.0, put_ltp=30.0,
        )
        assert s.pcr is None


class TestOptionsChainSnapshot:
    def _make_snapshot(self, pcr: float) -> OptionsChainSnapshot:
        # Build a snapshot with the desired aggregate PCR
        strikes = _make_strikes()
        total_call = sum(s.call_oi for s in strikes) or 1
        total_put = int(total_call * pcr)
        # Scale put OI uniformly
        factor = total_put / max(sum(s.put_oi for s in strikes), 1)
        for s in strikes:
            s.put_oi = int(s.put_oi * factor)
        snap = OptionsChainSnapshot(
            symbol="RELIANCE",
            underlying_price=2500.0,
            expiry="30-Jan-2025",
            strikes=strikes,
            pcr_oi=pcr,
            max_pain=2480.0,
            call_oi_wall=2600.0,
            put_oi_wall=2400.0,
            iv_skew=2.0,
        )
        return snap

    def test_high_pcr_is_bullish(self) -> None:
        snap = self._make_snapshot(pcr=1.4)
        assert snap.sentiment == "bullish"

    def test_low_pcr_is_bearish(self) -> None:
        snap = self._make_snapshot(pcr=0.6)
        assert snap.sentiment == "bearish"

    def test_neutral_pcr(self) -> None:
        snap = self._make_snapshot(pcr=1.0)
        assert snap.sentiment == "neutral"

    def test_signal_score_in_range(self) -> None:
        snap = self._make_snapshot(pcr=1.0)
        assert 0.0 <= snap.signal_score <= 1.0

    def test_to_dict_structure(self) -> None:
        snap = self._make_snapshot(pcr=1.2)
        d = snap.to_dict()
        assert "pcr_oi" in d
        assert "max_pain" in d
        assert "sentiment" in d
        assert "signal_score" in d


class TestComputeMaxPain:
    def test_max_pain_returns_a_strike(self) -> None:
        strikes = _make_strikes(base_price=2500.0, n=7)
        mp = _compute_max_pain(strikes)
        all_strike_values = [s.strike for s in strikes]
        assert mp in all_strike_values

    def test_empty_strikes(self) -> None:
        assert _compute_max_pain([]) == 0.0
