"""Company relationship graph — maps NSE/BSE symbols to related entities.

Related entities include:
- Subsidiaries
- Joint-venture partners
- Key suppliers / customers
- Direct competitors

The agent uses these relationships to cast a wide net when searching for
market-moving news — e.g. bad news about a Reliance JIO competitor (Airtel)
may signal opportunity for Reliance, while supply-chain issues at a
Tata Motors supplier directly affect TATAMOTORS.

In production this data should be loaded from a database or an external
data provider (e.g. Refinitiv, Bloomberg, or a curated YAML/JSON file).
The built-in seed data covers the Nifty 50 most actively traded names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RelationshipType = Literal["subsidiary", "partner", "competitor", "supplier", "customer"]


@dataclass
class RelatedEntity:
    """A company related to a listed stock."""

    name: str
    relationship: RelationshipType
    notes: str = ""


@dataclass
class CompanyProfile:
    """Minimal profile for a listed company."""

    symbol: str
    full_name: str
    sector: str
    related: list[RelatedEntity] = field(default_factory=list)

    def related_names(self) -> list[str]:
        """Return all related entity names as a flat list."""
        return [r.name for r in self.related]

    def related_by_type(self, rel_type: RelationshipType) -> list[str]:
        """Return related entity names filtered by relationship type."""
        return [r.name for r in self.related if r.relationship == rel_type]


# ── Seed data ─────────────────────────────────────────────────────────────────
# Coverage: major Nifty 50 / Nifty 100 components.
# Add or update entries as needed.

_SEED: list[CompanyProfile] = [
    CompanyProfile(
        symbol="RELIANCE",
        full_name="Reliance Industries",
        sector="Conglomerate",
        related=[
            RelatedEntity("Jio Platforms", "subsidiary"),
            RelatedEntity("Reliance Retail", "subsidiary"),
            RelatedEntity("Reliance Jio Infocomm", "subsidiary"),
            RelatedEntity("Network18", "subsidiary"),
            RelatedEntity("Bharti Airtel", "competitor"),
            RelatedEntity("Vodafone Idea", "competitor"),
            RelatedEntity("Amazon India", "competitor"),
            RelatedEntity("Flipkart", "competitor"),
            RelatedEntity("Saudi Aramco", "partner"),
        ],
    ),
    CompanyProfile(
        symbol="TCS",
        full_name="Tata Consultancy Services",
        sector="IT Services",
        related=[
            RelatedEntity("Tata Sons", "partner"),
            RelatedEntity("Infosys", "competitor"),
            RelatedEntity("Wipro", "competitor"),
            RelatedEntity("HCL Technologies", "competitor"),
            RelatedEntity("Accenture", "competitor"),
            RelatedEntity("Cognizant", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="INFY",
        full_name="Infosys",
        sector="IT Services",
        related=[
            RelatedEntity("TCS", "competitor"),
            RelatedEntity("Wipro", "competitor"),
            RelatedEntity("HCL Technologies", "competitor"),
            RelatedEntity("Accenture", "competitor"),
            RelatedEntity("EdgeVerve Systems", "subsidiary"),
            RelatedEntity("Infosys BPM", "subsidiary"),
        ],
    ),
    CompanyProfile(
        symbol="HDFCBANK",
        full_name="HDFC Bank",
        sector="Banking",
        related=[
            RelatedEntity("HDFC Life Insurance", "subsidiary"),
            RelatedEntity("HDFC Securities", "subsidiary"),
            RelatedEntity("ICICI Bank", "competitor"),
            RelatedEntity("Axis Bank", "competitor"),
            RelatedEntity("SBI", "competitor"),
            RelatedEntity("Kotak Mahindra Bank", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="ICICIBANK",
        full_name="ICICI Bank",
        sector="Banking",
        related=[
            RelatedEntity("ICICI Prudential Life Insurance", "subsidiary"),
            RelatedEntity("ICICI Lombard", "subsidiary"),
            RelatedEntity("ICICI Securities", "subsidiary"),
            RelatedEntity("HDFC Bank", "competitor"),
            RelatedEntity("Axis Bank", "competitor"),
            RelatedEntity("SBI", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="WIPRO",
        full_name="Wipro",
        sector="IT Services",
        related=[
            RelatedEntity("TCS", "competitor"),
            RelatedEntity("Infosys", "competitor"),
            RelatedEntity("HCL Technologies", "competitor"),
            RelatedEntity("Capgemini", "competitor"),
            RelatedEntity("Wipro GE Healthcare", "subsidiary"),
        ],
    ),
    CompanyProfile(
        symbol="AXISBANK",
        full_name="Axis Bank",
        sector="Banking",
        related=[
            RelatedEntity("Axis Securities", "subsidiary"),
            RelatedEntity("Axis Capital", "subsidiary"),
            RelatedEntity("HDFC Bank", "competitor"),
            RelatedEntity("ICICI Bank", "competitor"),
            RelatedEntity("Kotak Mahindra Bank", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="BAJFINANCE",
        full_name="Bajaj Finance",
        sector="NBFC",
        related=[
            RelatedEntity("Bajaj Finserv", "partner"),
            RelatedEntity("Bajaj Allianz", "partner"),
            RelatedEntity("Muthoot Finance", "competitor"),
            RelatedEntity("Cholamandalam Investment", "competitor"),
            RelatedEntity("HDFC Bank", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="TATAMOTORS",
        full_name="Tata Motors",
        sector="Automobile",
        related=[
            RelatedEntity("Jaguar Land Rover", "subsidiary"),
            RelatedEntity("Tata Daewoo", "subsidiary"),
            RelatedEntity("Tata AutoComp Systems", "subsidiary"),
            RelatedEntity("Maruti Suzuki", "competitor"),
            RelatedEntity("Hyundai Motor India", "competitor"),
            RelatedEntity("Mahindra & Mahindra", "competitor"),
            RelatedEntity("Ola Electric", "competitor"),
            RelatedEntity("Tesla", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="ITC",
        full_name="ITC",
        sector="FMCG",
        related=[
            RelatedEntity("ITC Hotels", "subsidiary"),
            RelatedEntity("ITC Agro Tech", "subsidiary"),
            RelatedEntity("Hindustan Unilever", "competitor"),
            RelatedEntity("Nestle India", "competitor"),
            RelatedEntity("Dabur", "competitor"),
            RelatedEntity("Marico", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="MARUTI",
        full_name="Maruti Suzuki India",
        sector="Automobile",
        related=[
            RelatedEntity("Suzuki Motor Corporation", "partner"),
            RelatedEntity("Hyundai Motor India", "competitor"),
            RelatedEntity("Tata Motors", "competitor"),
            RelatedEntity("Mahindra & Mahindra", "competitor"),
            RelatedEntity("Kia India", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="SUNPHARMA",
        full_name="Sun Pharmaceutical Industries",
        sector="Pharma",
        related=[
            RelatedEntity("Taro Pharmaceutical", "subsidiary"),
            RelatedEntity("Sun Pharma Advanced Research", "subsidiary"),
            RelatedEntity("Dr. Reddy's Laboratories", "competitor"),
            RelatedEntity("Cipla", "competitor"),
            RelatedEntity("Lupin", "competitor"),
            RelatedEntity("Aurobindo Pharma", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="ONGC",
        full_name="Oil and Natural Gas Corporation",
        sector="Energy",
        related=[
            RelatedEntity("ONGC Videsh", "subsidiary"),
            RelatedEntity("Hindustan Petroleum", "subsidiary"),
            RelatedEntity("MRPL", "subsidiary"),
            RelatedEntity("Reliance Industries", "competitor"),
            RelatedEntity("Cairn India", "competitor"),
            RelatedEntity("OPEC", "partner"),
        ],
    ),
    CompanyProfile(
        symbol="NTPC",
        full_name="NTPC",
        sector="Power",
        related=[
            RelatedEntity("NTPC Renewable Energy", "subsidiary"),
            RelatedEntity("Ratnagiri Gas and Power", "subsidiary"),
            RelatedEntity("Adani Power", "competitor"),
            RelatedEntity("Tata Power", "competitor"),
        ],
    ),
    CompanyProfile(
        symbol="LT",
        full_name="Larsen & Toubro",
        sector="Engineering & Construction",
        related=[
            RelatedEntity("L&T Finance Holdings", "subsidiary"),
            RelatedEntity("L&T Infotech", "subsidiary"),
            RelatedEntity("L&T Technology Services", "subsidiary"),
            RelatedEntity("L&T Hydrocarbon Engineering", "subsidiary"),
            RelatedEntity("Siemens India", "competitor"),
            RelatedEntity("ABB India", "competitor"),
        ],
    ),
]

_INDEX: dict[str, CompanyProfile] = {cp.symbol.upper(): cp for cp in _SEED}


class CompanyGraph:
    """Registry of company profiles with relationship look-ups.

    Profiles not in the seed data return a minimal auto-generated profile
    with no related entities rather than raising an error, so the agent
    can still function for unlisted or newly added symbols.
    """

    def get_profile(self, symbol: str) -> CompanyProfile:
        """Return the company profile for ``symbol``.

        Args:
            symbol: NSE/BSE symbol (case-insensitive).

        Returns:
            :class:`CompanyProfile` — either from seed data or a stub.
        """
        key = symbol.upper().strip()
        return _INDEX.get(key, CompanyProfile(symbol=key, full_name=key, sector="Unknown"))

    def get_related_entities(self, symbol: str) -> list[str]:
        """Return all related entity names for ``symbol``.

        Args:
            symbol: NSE/BSE symbol.

        Returns:
            List of entity names (company names, not symbols) for news search.
        """
        return self.get_profile(symbol).related_names()

    def get_competitors(self, symbol: str) -> list[str]:
        """Return competitor names for ``symbol``."""
        return self.get_profile(symbol).related_by_type("competitor")

    def get_subsidiaries(self, symbol: str) -> list[str]:
        """Return subsidiary names for ``symbol``."""
        return self.get_profile(symbol).related_by_type("subsidiary")

    def add_profile(self, profile: CompanyProfile) -> None:
        """Register a custom company profile (overwrites existing entry).

        Args:
            profile: A :class:`CompanyProfile` instance.
        """
        _INDEX[profile.symbol.upper()] = profile
