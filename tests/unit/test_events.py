"""Unit tests for the corporate event scraper and data models."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from trade_agent.data.events import (
    BulkDeal,
    CorporateEvent,
    EventScraper,
    EventType,
    MarketCapTier,
    _classify_event,
    _classify_institution,
    _event_id,
    get_freshness_window,
    get_market_cap_tier,
    score_event_direction,
)


class TestClassifyEvent:
    def test_results_keyword(self) -> None:
        assert _classify_event("Q3 Financial Results declared") == EventType.RESULTS

    def test_dividend_keyword(self) -> None:
        assert _classify_event("Board declares final dividend") == EventType.DIVIDEND

    def test_buyback_keyword(self) -> None:
        assert _classify_event("Company announces buyback of shares") == EventType.BUYBACK

    def test_qip_keyword(self) -> None:
        assert _classify_event("QIP allotment completed at ₹450") == EventType.QIP

    def test_merger_keyword(self) -> None:
        assert _classify_event("Acquisition of XYZ subsidiary approved") == EventType.MERGER_ACQUISITION

    def test_sebi_order(self) -> None:
        assert _classify_event("SEBI order against company directors") == EventType.SEBI_ORDER

    def test_management_change(self) -> None:
        assert _classify_event("Appointment of new CEO announced") == EventType.MANAGEMENT_CHANGE

    def test_pledge_keyword(self) -> None:
        assert _classify_event("Promoter pledging of shares disclosed") == EventType.PROMOTER_PLEDGE

    def test_unknown_falls_back_to_other(self) -> None:
        assert _classify_event("General update about operations") == EventType.OTHER

    def test_case_insensitive(self) -> None:
        assert _classify_event("BUYBACK ANNOUNCED") == EventType.BUYBACK


class TestClassifyInstitution:
    def test_fii_detected(self) -> None:
        is_inst, kind = _classify_institution("CLSA SINGAPORE PTE LTD")
        assert is_inst is True
        assert kind == "FII"

    def test_mutual_fund_detected(self) -> None:
        is_inst, kind = _classify_institution("SBI FUND MANAGEMENT LTD")
        assert is_inst is True
        assert kind == "MF"

    def test_lic_is_dii(self) -> None:
        is_inst, kind = _classify_institution("LIC OF INDIA")
        assert is_inst is True
        assert kind == "DII"

    def test_promoter_detected(self) -> None:
        is_inst, kind = _classify_institution("AMBANI PROMOTER GROUP")
        assert is_inst is True
        assert kind == "PROMOTER"

    def test_retail_not_institutional(self) -> None:
        is_inst, kind = _classify_institution("RAMESH KUMAR")
        assert is_inst is False
        assert kind == "OTHER"


class TestCorporateEvent:
    def _make_event(self, age_minutes: float = 30.0) -> CorporateEvent:
        filed = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
        return CorporateEvent(
            symbol="RELIANCE",
            exchange="NSE",
            event_type=EventType.BUYBACK,
            headline="Board approves buyback at premium",
            detail="Details of the buyback scheme",
            filing_time=filed,
            source_url="https://nseindia.com",
            event_id="abc123",
        )

    def test_age_minutes_approx(self) -> None:
        evt = self._make_event(age_minutes=45.0)
        assert 44 <= evt.age_minutes <= 46

    def test_urgency_buyback(self) -> None:
        evt = self._make_event()
        assert evt.urgency >= 0.8

    def test_to_dict_has_required_keys(self) -> None:
        evt = self._make_event()
        d = evt.to_dict()
        assert "event_type" in d
        assert "filing_time" in d
        assert "direction_score" in d
        assert "urgency" in d
        assert "freshness_window_minutes" in d
        assert "is_fresh" in d

    def test_freshness_window_reliance_is_15(self) -> None:
        # RELIANCE is Nifty50 → 15-minute window
        evt = self._make_event(age_minutes=10.0)
        assert evt.freshness_window == 15

    def test_is_fresh_within_window(self) -> None:
        evt = self._make_event(age_minutes=10.0)
        assert evt.is_fresh is True

    def test_is_stale_outside_window(self) -> None:
        evt = self._make_event(age_minutes=20.0)  # > 15 min window for Nifty50
        assert evt.is_fresh is False

    def test_direction_score_buyback_is_positive(self) -> None:
        evt = self._make_event()
        assert evt.direction_score == pytest.approx(1.0)

    def test_direction_backward_compat(self) -> None:
        evt = self._make_event()
        assert evt.direction == 1


class TestBulkDeal:
    def _make_deal(self, qty: int = 100_000, price: float = 2500.0) -> BulkDeal:
        return BulkDeal(
            symbol="RELIANCE",
            deal_type="BULK",
            client_name="CLSA SINGAPORE",
            transaction="BUY",
            quantity=qty,
            price=price,
            trade_date="01-Jan-2025",
            is_institutional=True,
            institution_type="FII",
        )

    def test_value_cr_calculation(self) -> None:
        deal = self._make_deal(qty=100_000, price=2500.0)
        assert deal.value_cr == pytest.approx(25.0)

    def test_to_dict_structure(self) -> None:
        d = self._make_deal().to_dict()
        assert "value_cr" in d
        assert "institution_type" in d
        assert d["is_institutional"] is True


class TestEventId:
    def test_same_inputs_produce_same_id(self) -> None:
        id1 = _event_id("RELIANCE", "Buyback approved", "01-Jan-2025")
        id2 = _event_id("RELIANCE", "Buyback approved", "01-Jan-2025")
        assert id1 == id2

    def test_different_inputs_produce_different_id(self) -> None:
        id1 = _event_id("RELIANCE", "Buyback approved", "01-Jan-2025")
        id2 = _event_id("TCS", "Buyback approved", "01-Jan-2025")
        assert id1 != id2


class TestMarketCapTier:
    def test_nifty50_symbol_is_large(self) -> None:
        assert get_market_cap_tier("RELIANCE") == MarketCapTier.LARGE
        assert get_market_cap_tier("TCS") == MarketCapTier.LARGE

    def test_nifty_next50_symbol_is_mid(self) -> None:
        assert get_market_cap_tier("ZOMATO") == MarketCapTier.MID

    def test_midcap_symbol_is_mid(self) -> None:
        assert get_market_cap_tier("IRCTC") == MarketCapTier.MID

    def test_unknown_symbol_is_small(self) -> None:
        assert get_market_cap_tier("UNKNOWNSYM") == MarketCapTier.SMALL

    def test_freshness_window_large_is_15(self) -> None:
        assert get_freshness_window("RELIANCE") == 15

    def test_freshness_window_mid_is_45(self) -> None:
        assert get_freshness_window("ZOMATO") == 45

    def test_freshness_window_small_is_90(self) -> None:
        assert get_freshness_window("UNKNOWNSYM") == 90

    def test_lowercase_input_handled(self) -> None:
        assert get_market_cap_tier("reliance") == MarketCapTier.LARGE


class TestScoreEventDirection:
    def test_buyback_is_bullish(self) -> None:
        score = score_event_direction(EventType.BUYBACK, "Company announces buyback")
        assert score == pytest.approx(1.0)

    def test_debt_default_is_bearish(self) -> None:
        score = score_event_direction(EventType.DEBT_DEFAULT, "Company defaults on NCD payment")
        assert score == pytest.approx(-1.0)

    def test_qip_capex_is_mildly_positive(self) -> None:
        score = score_event_direction(EventType.QIP, "QIP for capex and expansion")
        assert score == pytest.approx(0.3)

    def test_qip_distress_is_negative(self) -> None:
        score = score_event_direction(EventType.QIP, "QIP to repay debt and NPA resolution")
        assert score == pytest.approx(-0.8)

    def test_qip_default_is_mildly_negative(self) -> None:
        score = score_event_direction(EventType.QIP, "QIP allotment completed")
        assert score == pytest.approx(-0.3)

    def test_credit_rating_upgrade_is_positive(self) -> None:
        score = score_event_direction(EventType.CREDIT_RATING, "CRISIL upgrade to AA+")
        assert score == pytest.approx(0.8)

    def test_credit_rating_downgrade_is_negative(self) -> None:
        score = score_event_direction(EventType.CREDIT_RATING, "Moody's downgrade to B3")
        assert score == pytest.approx(-0.9)

    def test_management_resign_is_negative(self) -> None:
        score = score_event_direction(EventType.MANAGEMENT_CHANGE, "CFO resigns with immediate effect")
        assert score == pytest.approx(-0.5)

    def test_management_appoint_is_mildly_positive(self) -> None:
        score = score_event_direction(EventType.MANAGEMENT_CHANGE, "New CEO appointed from industry")
        assert score == pytest.approx(0.2)

    def test_ma_target_is_positive(self) -> None:
        score = score_event_direction(EventType.MERGER_ACQUISITION, "Company acquired via takeover bid")
        assert score == pytest.approx(0.8)

    def test_ma_acquirer_is_slightly_negative(self) -> None:
        score = score_event_direction(EventType.MERGER_ACQUISITION, "Company acquires rival for ₹500Cr")
        assert score == pytest.approx(-0.2)
