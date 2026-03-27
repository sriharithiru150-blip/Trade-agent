"""Unit tests for the EventScorer."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from trade_agent.analysis.event_scorer import EventEvidence, EventScorer, _bias
from trade_agent.data.events import BulkDeal, CorporateEvent, EventType
from trade_agent.data.options_chain import OptionsChainSnapshot, StrikeData


def _event(
    event_type: EventType = EventType.BUYBACK,
    age_minutes: float = 5.0,   # default within 15-min Nifty50 window
    symbol: str = "RELIANCE",
) -> CorporateEvent:
    filed = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return CorporateEvent(
        symbol=symbol,
        exchange="NSE",
        event_type=event_type,
        headline=f"{event_type.value} announcement",
        detail="Details",
        filing_time=filed,
        source_url="https://nseindia.com",
        event_id=f"id_{event_type.value}_{age_minutes}",
    )


def _deal(transaction: str = "BUY", value_cr: float = 20.0, is_inst: bool = True) -> BulkDeal:
    price = 2500.0
    qty = int(value_cr * 1e7 / price)
    return BulkDeal(
        symbol="RELIANCE",
        deal_type="BULK",
        client_name="FII CLIENT",
        transaction=transaction,
        quantity=qty,
        price=price,
        trade_date="01-Jan-2025",
        is_institutional=is_inst,
        institution_type="FII" if is_inst else "OTHER",
    )


def _options_snap(pcr: float = 1.2) -> OptionsChainSnapshot:
    strikes = [
        StrikeData(2400, 50000, 0, 18.0, 0.0, 80000, 0, 22.0, 0.0),
        StrikeData(2500, 60000, 0, 17.0, 0.0, 70000, 0, 21.0, 0.0),
        StrikeData(2600, 80000, 0, 16.0, 0.0, 40000, 0, 19.0, 0.0),
    ]
    return OptionsChainSnapshot(
        symbol="RELIANCE",
        underlying_price=2500.0,
        expiry="30-Jan-2025",
        strikes=strikes,
        pcr_oi=pcr,
        max_pain=2480.0,
        call_oi_wall=2600.0,
        put_oi_wall=2400.0,
        iv_skew=3.0,
    )


class TestEventScorer:
    def setup_method(self) -> None:
        self.scorer = EventScorer()

    def test_returns_event_evidence(self) -> None:
        ev = self.scorer.score("RELIANCE", [_event()], [_deal()], _options_snap())
        assert isinstance(ev, EventEvidence)

    def test_bullish_inputs_give_high_conviction(self) -> None:
        ev = self.scorer.score(
            "RELIANCE",
            [_event(EventType.BUYBACK, age_minutes=5)],  # well within 15-min window
            [_deal("BUY", 30.0)],
            _options_snap(pcr=1.3),
        )
        assert ev.conviction >= 0.60
        assert ev.bias == "bullish"

    def test_bearish_events_give_low_conviction(self) -> None:
        ev = self.scorer.score(
            "RELIANCE",
            [_event(EventType.DEBT_DEFAULT, age_minutes=5)],
            [_deal("SELL", 40.0)],
            _options_snap(pcr=0.5),
        )
        assert ev.conviction < 0.50

    def test_no_events_neutral(self) -> None:
        ev = self.scorer.score("TCS", [], [], None)
        assert ev.bias == "neutral"
        assert ev.conviction == pytest.approx(0.5, abs=0.1)

    def test_risks_populated_for_default(self) -> None:
        ev = self.scorer.score(
            "RELIANCE",
            [_event(EventType.DEBT_DEFAULT, age_minutes=5)],
            [],
            None,
        )
        assert len(ev.key_risks) > 0

    def test_adv_normalised_flow_score_high_deal(self) -> None:
        # ₹30Cr deal on ₹100Cr ADV → 30% of ADV → capped at 1.0
        ev = self.scorer.score(
            "RELIANCE",
            [],
            [_deal("BUY", 30.0)],
            None,
            adv_cr=100.0,
        )
        assert ev.flow_score == pytest.approx(1.0)

    def test_adv_normalised_flow_score_small_deal_on_large_adv(self) -> None:
        # ₹2Cr deal on ₹400Cr ADV → 0.5% of ADV → exactly at ADV floor, score near zero
        ev = self.scorer.score(
            "RELIANCE",
            [],
            [_deal("BUY", 2.0)],
            None,
            adv_cr=400.0,
        )
        # 0.5% of ADV = minimum threshold; deal is filtered (below 0.5% * 400 = 2.0 Cr minimum)
        # Actually 2.0/400 = 0.005 = _MIN_DEAL_ADV_FRACTION exactly, so NOT filtered
        # score = (2.0/400)/0.05 = 0.005/0.05 = 0.1
        assert 0.0 < ev.flow_score < 0.5

    def test_small_deal_below_floor_filtered(self) -> None:
        # ₹0.5Cr deal — below the ₹1Cr absolute floor
        tiny_deal = _deal("BUY", 0.5)
        ev = self.scorer.score("RELIANCE", [], [tiny_deal], None)
        # Filtered out → neutral flow
        assert ev.flow_score == pytest.approx(0.0)

    def test_stale_events_outside_window_ignored(self) -> None:
        # RELIANCE is Nifty50, window = 15 min; age 20 min → outside window
        stale_evt = _event(EventType.BUYBACK, age_minutes=20.0)
        ev = self.scorer.score("RELIANCE", [stale_evt], [], None)
        assert ev.event_score == pytest.approx(0.0)

    def test_to_dict_serialisable(self) -> None:
        ev = self.scorer.score("INFY", [_event()], [], None)
        d = ev.to_dict()
        assert "conviction" in d
        assert "bias" in d
        assert "event_summary" in d


class TestBias:
    def test_above_0_65_is_bullish(self) -> None:
        assert _bias(0.70) == "bullish"

    def test_below_0_35_is_bearish(self) -> None:
        assert _bias(0.30) == "bearish"

    def test_middle_is_neutral(self) -> None:
        assert _bias(0.50) == "neutral"
