"""News fetcher — aggregates news from multiple sources.

Sources:
1. NewsAPI (requires API key)
2. Google News RSS feed (no key required)
3. Economic Times / Moneycontrol RSS (no key required)

The fetcher is entity-aware: for a given company it also searches news
for known subsidiaries, partner organisations, and competitors — giving
the agent a 360° picture of market-relevant events.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

try:
    import feedparser  # type: ignore[import]
    _FEEDPARSER_OK = True
except Exception:  # pragma: no cover — old sgmllib-based feedparser on Python 3.11
    feedparser = None  # type: ignore[assignment]
    _FEEDPARSER_OK = False

import httpx

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Article:
    """A single news article."""

    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    entities_matched: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Title + description combined for analysis."""
        return f"{self.title}. {self.description}"


# ── RSS helpers ───────────────────────────────────────────────────────────────

_RSS_SOURCES = {
    "google_news": "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en",
    "economic_times": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "nse_announcements": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}


def _parse_rss_date(entry: Any) -> datetime:
    """Parse a feedparser entry's published date, falling back to now."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _fetch_rss(url: str, timeout: int = 10) -> list[Any]:
    """Fetch and parse an RSS feed, returning a list of entries."""
    if not _FEEDPARSER_OK:
        return []
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "TradeAgent/0.1"})
        return list(feed.entries)
    except Exception as exc:
        log.warning("rss_fetch_failed", url=url, error=str(exc))
        return []


# ── NewsAPI helper ────────────────────────────────────────────────────────────


def _fetch_newsapi(
    query: str,
    api_key: str,
    page_size: int = 10,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Fetch articles from NewsAPI.

    Args:
        query: Search query string.
        api_key: NewsAPI key.
        page_size: Number of articles to retrieve.
        language: Language code.

    Returns:
        List of raw article dicts from the API.
    """
    if not api_key:
        return []
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={quote_plus(query)}"
        f"&language={language}"
        f"&pageSize={page_size}"
        "&sortBy=publishedAt"
    )
    try:
        resp = httpx.get(url, headers={"X-Api-Key": api_key}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception as exc:
        log.warning("newsapi_fetch_failed", query=query, error=str(exc))
        return []


# ── Main fetcher ──────────────────────────────────────────────────────────────


class NewsFetcher:
    """Aggregate news for a company and its related entities.

    Args:
        news_api_key: Optional NewsAPI key; RSS feeds are used regardless.
        max_articles_per_entity: Cap on articles fetched per search term.
    """

    def __init__(
        self,
        news_api_key: str = "",
        max_articles_per_entity: int = 5,
    ) -> None:
        self._api_key = news_api_key
        self._max = max_articles_per_entity

    def fetch_for_symbol(
        self,
        symbol: str,
        company_name: str,
        related_entities: list[str] | None = None,
    ) -> list[Article]:
        """Fetch news for a stock and all related entities.

        Args:
            symbol: NSE/BSE symbol (used as a fallback search term).
            company_name: Full company name (e.g. ``"Reliance Industries"``).
            related_entities: Additional search terms — subsidiaries, partners,
                competitors — supplied by :mod:`trade_agent.data.company_graph`.

        Returns:
            Deduplicated list of :class:`Article` objects sorted newest-first.
        """
        entities = [company_name, symbol]
        if related_entities:
            entities.extend(related_entities)

        articles: list[Article] = []
        seen_urls: set[str] = set()

        for entity in entities:
            batch = self._fetch_entity(entity)
            for art in batch:
                if art.url not in seen_urls:
                    art.entities_matched.append(entity)
                    seen_urls.add(art.url)
                    articles.append(art)

        articles.sort(key=lambda a: a.published_at, reverse=True)
        log.info(
            "news_fetched",
            symbol=symbol,
            entities_searched=len(entities),
            articles_found=len(articles),
        )
        return articles

    def fetch_market_news(self) -> list[Article]:
        """Fetch broad Indian market news (not symbol-specific).

        Returns:
            List of recent market/economy news articles.
        """
        articles: list[Article] = []
        seen_urls: set[str] = set()

        for source_key in ("economic_times", "moneycontrol"):
            url = _RSS_SOURCES[source_key]
            for entry in _fetch_rss(url)[: self._max * 2]:
                art = self._rss_entry_to_article(entry, source=source_key)
                if art and art.url not in seen_urls:
                    seen_urls.add(art.url)
                    articles.append(art)

        # NewsAPI general Indian market query
        for raw in _fetch_newsapi("India stock market NSE BSE Nifty Sensex", self._api_key, page_size=10):
            art = self._newsapi_to_article(raw)
            if art and art.url not in seen_urls:
                seen_urls.add(art.url)
                articles.append(art)

        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles

    # ── private ───────────────────────────────────────────────────────────────

    def _fetch_entity(self, entity: str) -> list[Article]:
        """Fetch articles for a single search entity."""
        articles: list[Article] = []

        # Google News RSS
        rss_url = _RSS_SOURCES["google_news"].format(query=quote_plus(entity))
        for entry in _fetch_rss(rss_url)[: self._max]:
            art = self._rss_entry_to_article(entry, source="google_news")
            if art:
                articles.append(art)
            time.sleep(0.05)  # gentle rate limiting

        # NewsAPI
        for raw in _fetch_newsapi(entity, self._api_key, page_size=self._max):
            art = self._newsapi_to_article(raw)
            if art:
                articles.append(art)

        return articles

    @staticmethod
    def _rss_entry_to_article(entry: Any, source: str) -> Article | None:
        """Convert a feedparser entry dict to an :class:`Article`."""
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        # Strip HTML tags from summary
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        link = getattr(entry, "link", "") or ""
        if not title or not link:
            return None
        return Article(
            title=title.strip(),
            description=summary[:500],
            url=link,
            source=source,
            published_at=_parse_rss_date(entry),
        )

    @staticmethod
    def _newsapi_to_article(raw: dict[str, Any]) -> Article | None:
        """Convert a NewsAPI article dict to an :class:`Article`."""
        title = (raw.get("title") or "").strip()
        description = (raw.get("description") or "").strip()
        url = (raw.get("url") or "").strip()
        if not title or not url or url == "https://removed.com":
            return None
        source_name = (raw.get("source") or {}).get("name", "newsapi")
        published_raw = raw.get("publishedAt") or ""
        try:
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except Exception:
            published_at = datetime.now(timezone.utc)
        return Article(
            title=title,
            description=description[:500],
            url=url,
            source=source_name,
            published_at=published_at,
        )
