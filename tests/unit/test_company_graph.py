"""Unit tests for the company relationship graph."""

from __future__ import annotations

from trade_agent.data.company_graph import CompanyGraph, CompanyProfile, RelatedEntity


class TestCompanyGraph:
    def setup_method(self) -> None:
        self.graph = CompanyGraph()

    def test_known_symbol_returns_profile(self) -> None:
        profile = self.graph.get_profile("RELIANCE")
        assert profile.symbol == "RELIANCE"
        assert profile.full_name == "Reliance Industries"

    def test_case_insensitive(self) -> None:
        p1 = self.graph.get_profile("reliance")
        p2 = self.graph.get_profile("RELIANCE")
        assert p1.symbol == p2.symbol

    def test_unknown_symbol_returns_stub(self) -> None:
        profile = self.graph.get_profile("UNKNOWNSYMBOL")
        assert profile.symbol == "UNKNOWNSYMBOL"
        assert profile.sector == "Unknown"
        assert profile.related_names() == []

    def test_related_entities_not_empty_for_known(self) -> None:
        entities = self.graph.get_related_entities("TCS")
        assert len(entities) > 0

    def test_competitors_subset_of_related(self) -> None:
        all_related = set(self.graph.get_related_entities("HDFCBANK"))
        competitors = set(self.graph.get_competitors("HDFCBANK"))
        assert competitors.issubset(all_related)

    def test_subsidiaries_for_reliance(self) -> None:
        subs = self.graph.get_subsidiaries("RELIANCE")
        assert "Jio Platforms" in subs

    def test_add_custom_profile(self) -> None:
        custom = CompanyProfile(
            symbol="CUSTOM",
            full_name="Custom Corp",
            sector="Test",
            related=[RelatedEntity("Rival Corp", "competitor")],
        )
        self.graph.add_profile(custom)
        assert self.graph.get_profile("CUSTOM").full_name == "Custom Corp"
        assert "Rival Corp" in self.graph.get_competitors("CUSTOM")
