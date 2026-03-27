"""Corporate event scraper — NSE and BSE announcements, SEBI filings.

This is the primary alpha source in the new architecture.  We poll the
exchange announcement feeds every minute and surface events *before* the
retail crowd picks them up via news aggregators.

Event categories that carry genuine edge
-----------------------------------------
- Board meeting outcomes (dividend, bonus, split, buyback)
- Quarterly / annual results (beats/misses relative to estimates)
- QIP / rights issue / FPO announcements (dilution events)
- Merger, demerger, acquisition announcements
- Promoter pledging or un-pledging of shares
- SEBI order / regulatory action
- Debt default / credit downgrade
- Management change (CEO/CFO/MD departure or appointment)
- NSE/BSE bulk deal & block deal filings (institutional conviction signal)

Sources used (no paid subscription required)
---------------------------------------------
- NSE corporate announcements API  (requires browser-like headers + session)
- BSE corporate announcements API
- NSE bulk deal / block deal endpoints
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

# ── NSE / BSE endpoints ───────────────────────────────────────────────────────

_NSE_BASE = "https://www.nseindia.com"
_BSE_BASE = "https://api.bseindia.com"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}


# ── Event taxonomy ────────────────────────────────────────────────────────────


class EventType(str, Enum):
    BOARD_MEETING = "board_meeting"
    RESULTS = "results"
    DIVIDEND = "dividend"
    BONUS = "bonus"
    SPLIT = "split"
    BUYBACK = "buyback"
    QIP = "qip"
    RIGHTS_ISSUE = "rights_issue"
    MERGER_ACQUISITION = "merger_acquisition"
    DEMERGER = "demerger"
    PROMOTER_PLEDGE = "promoter_pledge"
    PROMOTER_BUY = "promoter_buy"
    PROMOTER_SELL = "promoter_sell"
    SEBI_ORDER = "sebi_order"
    CREDIT_RATING = "credit_rating"
    DEBT_DEFAULT = "debt_default"
    MANAGEMENT_CHANGE = "management_change"
    BULK_DEAL = "bulk_deal"
    BLOCK_DEAL = "block_deal"
    OTHER = "other"


# Keywords that map headline text → EventType
_TYPE_KEYWORDS: list[tuple[list[str], EventType]] = [
    (["board meeting", "board of directors"], EventType.BOARD_MEETING),
    (["financial results", "quarterly results", "q1", "q2", "q3", "q4", "annual results"], EventType.RESULTS),
    (["dividend"], EventType.DIVIDEND),
    (["bonus shares", "bonus issue"], EventType.BONUS),
    (["stock split", "sub-division"], EventType.SPLIT),
    (["buyback", "buy-back", "share repurchase"], EventType.BUYBACK),
    (["qip", "qualified institutional placement"], EventType.QIP),
    (["rights issue", "rights entitlement"], EventType.RIGHTS_ISSUE),
    (["acquisition", "merger", "amalgamation", "takeover"], EventType.MERGER_ACQUISITION),
    (["demerger", "spin-off", "spinoff", "hive-off"], EventType.DEMERGER),
    (["pledge", "pledging", "encumbrance"], EventType.PROMOTER_PLEDGE),
    (["sebi order", "sebi directive", "sebi notice", "adjudication"], EventType.SEBI_ORDER),
    (["credit rating", "rating upgrade", "rating downgrade", "outlook"], EventType.CREDIT_RATING),
    (["default", "npa", "debt restructuring", "moratorium"], EventType.DEBT_DEFAULT),
    (["ceo", "cfo", "coo", "md ", "managing director", "director resign", "director appoint"], EventType.MANAGEMENT_CHANGE),
]

# ── Market cap tiers for freshness windows ────────────────────────────────────
# Nifty 50 algos reprice in seconds; mid-caps take longer.
# Freshness window = maximum event age (minutes) that still carries edge.

class MarketCapTier(str, Enum):
    LARGE = "large"    # Nifty 50 — window 15 min
    MID   = "mid"      # Nifty 51-500 — window 45 min
    SMALL = "small"    # Below Nifty 500 / unknown — window 90 min

FRESHNESS_WINDOW: dict[MarketCapTier, int] = {
    MarketCapTier.LARGE: 15,
    MarketCapTier.MID:   45,
    MarketCapTier.SMALL: 90,
}

# Intraday freshness windows — tighter because algos reprice Nifty50 within
# seconds; even mid-caps are efficiently priced within 15 min during market hours.
INTRADAY_FRESHNESS_WINDOW: dict[MarketCapTier, int] = {
    MarketCapTier.LARGE: 5,
    MarketCapTier.MID:   15,
    MarketCapTier.SMALL: 30,
}

# Static Nifty 50 membership (as of 2025)
_NIFTY50: frozenset[str] = frozenset({
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "TECHM", "HCLTECH",
    "TATAMOTORS", "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS",
    "BAJAJFINSV", "BAJAJ-AUTO", "HEROMOTOCO", "DRREDDY", "CIPLA",
    "DIVISLAB", "EICHERMOT", "COALINDIA", "BPCL", "HINDALCO",
    "APOLLOHOSP", "TATACONSUM", "BRITANNIA", "SBILIFE", "HDFCLIFE",
    "M&M", "GRASIM", "INDUSINDBK", "UPL", "SHREECEM",
})

# Nifty 100 additions (51-100)
_NIFTY_NEXT50: frozenset[str] = frozenset({
    "DMART", "VEDL", "SIEMENS", "HAVELLS", "PIDILITIND", "BERGEPAINT",
    "MCDOWELL-N", "COLPAL", "GODREJCP", "DABUR", "MARICO", "MUTHOOTFIN",
    "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "FEDERALBNK", "IDFCFIRSTB",
    "BANDHANBNK", "RBLBANK", "CHOLAFIN", "BAJAJHLDNG", "LICHSGFIN",
    "ABCAPITAL", "MANAPPURAM", "PFC", "RECLTD", "IRFC",
    "TATAPOWER", "ADANIGREEN", "ADANITRANS", "TORNTPOWER", "CESC",
    "GAIL", "IOC", "HINDPETRO", "MGL", "IGL",
    "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM", "LALPATHLAB",
    "FORTIS", "MAXHEALTH", "METROPOLIS",
    "ZOMATO", "PAYTM", "NYKAA", "POLICYBZR",
})

_NIFTY_MIDCAP: frozenset[str] = frozenset({
    "VOLTAS", "WHIRLPOOL", "BLUESTAR", "CROMPTON", "POLYCAB", "KEI",
    "ASTRAL", "SUPREMIND", "AARTIIND", "DEEPAKNI", "SRF", "ALKYLAMINE",
    "LAXMIMACH", "ELGIEQUIP", "GREAVESCOT", "SCHAEFFLER", "TIMKEN",
    "PAGEIND", "RELAXO", "BATAINDIA", "VIPIND",
    "PERSISTENT", "MPHASIS", "LTTS", "COFORGE", "KPITTECH",
    "IRCTC", "CONCOR", "BLUEDART", "MAHLOG",
    "APLAPOLLO", "JSPL", "SAIL", "NMDC", "MOIL",
    "CAMS", "CDSL", "BSE", "MCX",
    "ABSLAMC", "HDFCAMC", "ICICIGI", "SBICARD",
})


def get_market_cap_tier(symbol: str) -> MarketCapTier:
    """Return the market cap tier for a symbol (used for freshness windowing).

    Args:
        symbol: NSE stock symbol (uppercase).

    Returns:
        :class:`MarketCapTier` — LARGE, MID, or SMALL.
    """
    sym = symbol.upper().strip()
    if sym in _NIFTY50:
        return MarketCapTier.LARGE
    if sym in _NIFTY_NEXT50 or sym in _NIFTY_MIDCAP:
        return MarketCapTier.MID
    return MarketCapTier.SMALL


def get_freshness_window(symbol: str, intraday: bool = False) -> int:
    """Return the maximum event age in minutes that still carries edge.

    Swing mode: Nifty50=15 min, mid=45 min, small=90 min.
    Intraday mode: Nifty50=5 min, mid=15 min, small=30 min.

    Args:
        symbol: NSE stock symbol.
        intraday: If True, return the tighter intraday window.

    Returns:
        Maximum event age in minutes.
    """
    tier = get_market_cap_tier(symbol)
    windows = INTRADAY_FRESHNESS_WINDOW if intraday else FRESHNESS_WINDOW
    return windows[tier]


# ── Contextual direction scoring ──────────────────────────────────────────────
# Base direction per event type.  Some types (QIP, credit rating, management
# change) need headline context to determine actual direction — see
# score_event_direction() below for the full logic.

_BASE_DIRECTION: dict[EventType, float] = {
    EventType.BOARD_MEETING:      0.0,
    EventType.RESULTS:            0.0,   # requires actual numbers
    EventType.DIVIDEND:           0.8,   # mostly positive; sometimes signals mature/no-growth
    EventType.BONUS:              0.8,
    EventType.SPLIT:              0.6,
    EventType.BUYBACK:            1.0,   # clear positive: management buying at premium
    EventType.QIP:                0.0,   # context-dependent — see score_event_direction()
    EventType.RIGHTS_ISSUE:      -0.5,   # usually dilutive, occasionally growth signal
    EventType.MERGER_ACQUISITION: 0.0,   # acquirer often dips; target rises
    EventType.DEMERGER:           0.7,   # unlocks value typically
    EventType.PROMOTER_PLEDGE:   -0.9,   # near-always bearish
    EventType.PROMOTER_BUY:       0.9,
    EventType.PROMOTER_SELL:     -0.7,
    EventType.SEBI_ORDER:        -0.9,
    EventType.CREDIT_RATING:      0.0,   # context-dependent — see score_event_direction()
    EventType.DEBT_DEFAULT:      -1.0,
    EventType.MANAGEMENT_CHANGE:  0.0,   # context-dependent — see score_event_direction()
    EventType.BULK_DEAL:          0.0,   # scored separately via flow_score
    EventType.BLOCK_DEAL:         0.0,
    EventType.OTHER:              0.0,
}

# Retained for backward-compat (used in to_dict as integer approximation)
EVENT_DIRECTION: dict[EventType, int] = {k: int(round(v)) for k, v in _BASE_DIRECTION.items()}


def score_event_direction(event_type: EventType, headline: str) -> float:
    """Return a contextual direction score in [-1.0, 1.0].

    For event types where context matters (QIP, credit rating, management
    change), this parses the headline to refine the direction beyond the
    flat base value.

    Args:
        event_type: The classified event type.
        headline: Raw headline text from the filing.

    Returns:
        Directional score: positive = bullish, negative = bearish.
    """
    base = _BASE_DIRECTION.get(event_type, 0.0)
    text = headline.lower()

    if event_type == EventType.QIP:
        # Capex / expansion QIP → mildly positive; distress / debt QIP → negative
        if any(k in text for k in ["capex", "expansion", "growth", "acquisition fund"]):
            return 0.3
        if any(k in text for k in ["debt", "npa", "repay", "loss", "distress"]):
            return -0.8
        return -0.3   # default: mild dilution concern

    if event_type == EventType.CREDIT_RATING:
        if any(k in text for k in ["upgrade", "positive outlook", "watch positive"]):
            return 0.8
        if any(k in text for k in ["downgrade", "negative outlook", "watch negative", "default"]):
            return -0.9
        if any(k in text for k in ["affirm", "reaffirm", "stable"]):
            return 0.1
        return 0.0

    if event_type == EventType.MANAGEMENT_CHANGE:
        if any(k in text for k in ["resign", "stepped down", "vacates", "dismissed"]):
            return -0.5
        if any(k in text for k in ["appoint", "joins", "takes charge", "elevated"]):
            return 0.2
        return 0.0

    if event_type == EventType.RESULTS:
        # Without actual numbers we can't score; caller must use Claude
        return 0.0

    if event_type == EventType.MERGER_ACQUISITION:
        # Acquiring company (buyer) typically dips; target spikes
        if any(k in text for k in ["acquires", "acquirer", "buys", "purchase"]):
            return -0.2   # buyer — uncertain
        if any(k in text for k in ["acquired", "takeover", "bid received"]):
            return 0.8    # target — premium expected
        return 0.0

    return base

# Urgency weight — how quickly price typically reacts
EVENT_URGENCY: dict[EventType, float] = {
    EventType.RESULTS: 1.0,
    EventType.MERGER_ACQUISITION: 1.0,
    EventType.DEBT_DEFAULT: 1.0,
    EventType.SEBI_ORDER: 0.9,
    EventType.BUYBACK: 0.9,
    EventType.QIP: 0.85,
    EventType.DIVIDEND: 0.7,
    EventType.BONUS: 0.7,
    EventType.SPLIT: 0.65,
    EventType.DEMERGER: 0.8,
    EventType.MANAGEMENT_CHANGE: 0.75,
    EventType.PROMOTER_PLEDGE: 0.8,
    EventType.PROMOTER_BUY: 0.7,
    EventType.PROMOTER_SELL: 0.75,
    EventType.CREDIT_RATING: 0.8,
    EventType.BULK_DEAL: 0.6,
    EventType.BLOCK_DEAL: 0.55,
    EventType.BOARD_MEETING: 0.4,
    EventType.RIGHTS_ISSUE: 0.6,
    EventType.OTHER: 0.2,
}


@dataclass
class CorporateEvent:
    """A single exchange-filed corporate event."""

    symbol: str
    exchange: str            # "NSE" or "BSE"
    event_type: EventType
    headline: str
    detail: str
    filing_time: datetime
    source_url: str
    event_id: str            # dedup key derived from content hash

    @property
    def age_minutes(self) -> float:
        """Minutes elapsed since the event was filed."""
        return (datetime.now(timezone.utc) - self.filing_time).total_seconds() / 60

    @property
    def direction(self) -> int:
        """Integer direction for backward compat (-1/0/1)."""
        return int(round(self.direction_score))

    @property
    def direction_score(self) -> float:
        """Contextual direction score [-1.0, 1.0], headline-aware."""
        return score_event_direction(self.event_type, self.headline)

    @property
    def urgency(self) -> float:
        return EVENT_URGENCY.get(self.event_type, 0.2)

    @property
    def freshness_window(self) -> int:
        """Maximum age in minutes for this event to still carry edge (swing mode)."""
        return get_freshness_window(self.symbol, intraday=False)

    @property
    def intraday_freshness_window(self) -> int:
        """Maximum age in minutes for this event to still carry edge (intraday mode)."""
        return get_freshness_window(self.symbol, intraday=True)

    @property
    def is_fresh(self) -> bool:
        """True if the event is within the tier-appropriate freshness window (swing)."""
        return self.age_minutes <= self.freshness_window

    @property
    def is_fresh_intraday(self) -> bool:
        """True if the event is within the tighter intraday freshness window."""
        return self.age_minutes <= self.intraday_freshness_window

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "event_type": self.event_type.value,
            "headline": self.headline,
            "detail": self.detail[:300],
            "filing_time": self.filing_time.isoformat(),
            "age_minutes": round(self.age_minutes, 1),
            "direction_score": round(self.direction_score, 2),
            "urgency": self.urgency,
            "freshness_window_minutes": self.freshness_window,
            "intraday_freshness_window_minutes": self.intraday_freshness_window,
            "is_fresh": self.is_fresh,
            "is_fresh_intraday": self.is_fresh_intraday,
        }


@dataclass
class BulkDeal:
    """A single bulk/block deal filing."""

    symbol: str
    deal_type: str       # "BULK" or "BLOCK"
    client_name: str
    transaction: str     # "BUY" or "SELL"
    quantity: int
    price: float
    trade_date: str
    # Derived
    is_institutional: bool = False
    institution_type: str = ""  # "FII", "MF", "DII", "PROMOTER", "OTHER"

    @property
    def value_cr(self) -> float:
        """Trade value in crores."""
        return (self.quantity * self.price) / 1e7

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "deal_type": self.deal_type,
            "client_name": self.client_name,
            "transaction": self.transaction,
            "quantity": self.quantity,
            "price": round(self.price, 2),
            "value_cr": round(self.value_cr, 2),
            "trade_date": self.trade_date,
            "is_institutional": self.is_institutional,
            "institution_type": self.institution_type,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _classify_event(headline: str) -> EventType:
    """Map a headline string to an EventType via keyword matching."""
    text = headline.lower()
    for keywords, event_type in _TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return event_type
    return EventType.OTHER


def _event_id(symbol: str, headline: str, filing_time: str) -> str:
    """Deterministic ID for deduplication."""
    raw = f"{symbol}|{headline}|{filing_time}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _classify_institution(client_name: str) -> tuple[bool, str]:
    """Heuristically classify a bulk/block deal client as institutional."""
    name = client_name.upper()
    if any(k in name for k in ["FII", "FPI", "FOREIGN", "OVERSEAS", "INDIA FUND", "CAYMAN",
                                "MAURITIUS", "SINGAPORE", "LLOYDS", "MERRILL", "MORGAN",
                                "GOLDMAN", "CITIBANK", "HSBC", "NOMURA", "CLSA", "MACQUARIE"]):
        return True, "FII"
    if any(k in name for k in ["MUTUAL FUND", "MF ", " MF", "AMC", "SBI FUND", "HDFC AMC",
                                "ICICI PRUD", "NIPPON", "AXIS AMC", "KOTAK AMC", "UTI AMC",
                                "ADITYA BIRLA", "FRANKLIN", "MIRAE", "MOTILAL"]):
        return True, "MF"
    if any(k in name for k in ["LIC", "GIC", "NPS", "EPFO", "PENSION", "INSURANCE"]):
        return True, "DII"
    if any(k in name for k in ["PROMOTER", "FOUNDER", "FAMILY", "HOLDING", "PROMOTER GROUP"]):
        return True, "PROMOTER"
    return False, "OTHER"


# ── NSE session (required for cookie-based API) ───────────────────────────────


class _NSESession:
    """Manages an httpx session with NSE-compatible cookies."""

    def __init__(self) -> None:
        self._client = httpx.Client(
            headers=_NSE_HEADERS, timeout=15, follow_redirects=True
        )
        self._cookie_valid = False

    def _refresh_cookies(self) -> None:
        """Hit the NSE homepage to get a valid session cookie."""
        try:
            self._client.get(_NSE_BASE)
            self._cookie_valid = True
        except Exception as exc:
            log.warning("nse_cookie_refresh_failed", error=str(exc))

    def get(self, url: str) -> dict[str, Any]:
        """GET a JSON endpoint, refreshing cookies on first call."""
        if not self._cookie_valid:
            self._refresh_cookies()
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                # Cookie stale — try once more
                self._cookie_valid = False
                self._refresh_cookies()
                resp = self._client.get(url)
                resp.raise_for_status()
                return resp.json()
            raise


# ── Event scraper ─────────────────────────────────────────────────────────────


class EventScraper:
    """Scrape NSE/BSE for corporate announcements and institutional deals.

    Maintains a set of seen event IDs to avoid re-surfacing old events.

    Args:
        max_age_minutes: Ignore events older than this (default 120 min).
    """

    def __init__(self, max_age_minutes: int = 120) -> None:
        self._nse = _NSESession()
        self._seen: set[str] = set()
        self._max_age = max_age_minutes

    def get_announcements(
        self,
        symbol: str | None = None,
        new_only: bool = True,
    ) -> list[CorporateEvent]:
        """Fetch recent NSE corporate announcements.

        Args:
            symbol: Filter to a specific symbol; None returns all.
            new_only: If True, skip events already seen this session.

        Returns:
            List of :class:`CorporateEvent` objects, newest first.
        """
        events: list[CorporateEvent] = []

        # NSE announcements endpoint
        url = f"{_NSE_BASE}/api/corporate-announcements?index=equities"
        if symbol:
            url += f"&symbol={symbol.upper()}"

        try:
            data = self._nse.get(url)
            raw_list = data if isinstance(data, list) else data.get("data", [])
        except Exception as exc:
            log.warning("nse_announcements_failed", symbol=symbol, error=str(exc))
            raw_list = []

        for item in raw_list:
            evt = self._parse_nse_announcement(item)
            if evt is None:
                continue
            if evt.age_minutes > self._max_age:
                continue
            if new_only and evt.event_id in self._seen:
                continue
            self._seen.add(evt.event_id)
            events.append(evt)

        events.sort(key=lambda e: e.filing_time, reverse=True)
        log.info(
            "announcements_fetched",
            symbol=symbol or "all",
            count=len(events),
        )
        return events

    def get_bulk_deals(self) -> list[BulkDeal]:
        """Fetch today's NSE bulk deals.

        Returns:
            List of :class:`BulkDeal` objects.
        """
        deals: list[BulkDeal] = []
        url = f"{_NSE_BASE}/api/bulk-deals"
        try:
            data = self._nse.get(url)
            raw_list = data if isinstance(data, list) else data.get("data", [])
            for item in raw_list:
                deal = self._parse_bulk_deal(item, "BULK")
                if deal:
                    deals.append(deal)
        except Exception as exc:
            log.warning("bulk_deals_failed", error=str(exc))

        return deals

    def get_block_deals(self) -> list[BulkDeal]:
        """Fetch today's NSE block deals.

        Returns:
            List of :class:`BulkDeal` objects.
        """
        deals: list[BulkDeal] = []
        url = f"{_NSE_BASE}/api/block-deals"
        try:
            data = self._nse.get(url)
            raw_list = data if isinstance(data, list) else data.get("data", [])
            for item in raw_list:
                deal = self._parse_bulk_deal(item, "BLOCK")
                if deal:
                    deals.append(deal)
        except Exception as exc:
            log.warning("block_deals_failed", error=str(exc))

        return deals

    def get_all_deals(self) -> list[BulkDeal]:
        """Fetch both bulk and block deals for today."""
        return self.get_bulk_deals() + self.get_block_deals()

    def get_deals_for_symbol(self, symbol: str) -> list[BulkDeal]:
        """Filter bulk+block deals to a specific symbol."""
        sym = symbol.upper()
        return [d for d in self.get_all_deals() if d.symbol == sym]

    # ── Parsers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_nse_announcement(item: dict[str, Any]) -> CorporateEvent | None:
        """Parse a raw NSE announcement dict."""
        try:
            symbol = (item.get("symbol") or item.get("sm_symbol") or "").upper().strip()
            headline = (item.get("desc") or item.get("subject") or "").strip()
            detail = (item.get("attchmntText") or item.get("body") or "").strip()
            filing_raw = item.get("an_dt") or item.get("date") or ""

            if not symbol or not headline:
                return None

            # Parse filing datetime
            filing_time: datetime
            for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S"):
                try:
                    filing_time = datetime.strptime(filing_raw, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
            else:
                filing_time = datetime.now(timezone.utc)

            event_type = _classify_event(headline)
            event_id = _event_id(symbol, headline, filing_raw)
            source_url = (
                f"https://www.nseindia.com/companies-listing/corporate-filings-announcements"
            )

            return CorporateEvent(
                symbol=symbol,
                exchange="NSE",
                event_type=event_type,
                headline=headline,
                detail=detail[:500],
                filing_time=filing_time,
                source_url=source_url,
                event_id=event_id,
            )
        except Exception as exc:
            log.debug("announcement_parse_error", error=str(exc))
            return None

    @staticmethod
    def _parse_bulk_deal(item: dict[str, Any], deal_type: str) -> BulkDeal | None:
        """Parse a raw NSE bulk/block deal dict."""
        try:
            symbol = (item.get("symbol") or item.get("Symbol") or "").upper().strip()
            client = (item.get("client_name") or item.get("ClientName") or "").strip()
            txn = (item.get("buySell") or item.get("BuySell") or "BUY").upper().strip()
            qty = int(float(item.get("quantityTraded") or item.get("Qty") or 0))
            price = float(item.get("wgtAvgPrice") or item.get("Price") or 0)
            date = str(item.get("trade_date") or item.get("TradeDate") or "")

            if not symbol or qty <= 0 or price <= 0:
                return None

            is_inst, inst_type = _classify_institution(client)
            return BulkDeal(
                symbol=symbol,
                deal_type=deal_type,
                client_name=client,
                transaction=txn if txn in ("BUY", "SELL") else "BUY",
                quantity=qty,
                price=price,
                trade_date=date,
                is_institutional=is_inst,
                institution_type=inst_type,
            )
        except Exception as exc:
            log.debug("bulk_deal_parse_error", error=str(exc))
            return None
