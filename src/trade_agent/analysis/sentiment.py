"""News sentiment analyser — uses Claude to score articles.

Each article is scored on two dimensions:
- **relevance** (0.0–1.0): how directly the article affects the stock's price.
- **sentiment** (−1.0–1.0): negative → bearish, positive → bullish.

A batch of articles is sent to Claude in a single API call to minimise
latency and token usage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from trade_agent.data.news_fetcher import Article
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a professional financial analyst specialising in Indian equity markets (NSE/BSE).

Your task is to score news articles for a specific stock.

For each article you will return:
- "relevance": float 0.0-1.0 — how directly this news affects the stock price
  (1.0 = direct material impact, 0.5 = moderate indirect impact, 0.0 = unrelated)
- "sentiment": float -1.0 to 1.0 — market sentiment
  (-1.0 = strongly bearish, 0.0 = neutral, 1.0 = strongly bullish)
- "reasoning": string — one sentence explaining the score

Consider the following when scoring:
- Regulatory actions, government policy, budget announcements
- Earnings beats/misses and guidance revisions
- M&A activity, fundraising, stake sales
- Management changes
- Subsidiary or competitor news that has knock-on effects
- Macro events (RBI rate decisions, INR moves, FII flows)

Return ONLY a valid JSON array of objects, one per article, in the same order provided.
Do not add any explanation outside the JSON.
"""


@dataclass
class SentimentScore:
    """Sentiment result for a single article."""

    article: Article
    relevance: float    # 0–1
    sentiment: float    # −1 to 1
    reasoning: str

    @property
    def weighted_score(self) -> float:
        """Sentiment weighted by relevance (−1 to 1)."""
        return self.relevance * self.sentiment


@dataclass
class AggregateSentiment:
    """Rolled-up sentiment across all scored articles."""

    scores: list[SentimentScore]
    avg_sentiment: float         # −1 to 1
    avg_relevance: float         # 0–1
    weighted_sentiment: float    # relevance-weighted sentiment
    article_count: int
    bullish_count: int
    bearish_count: int

    @property
    def signal(self) -> str:
        """Human-readable signal: 'bullish' | 'bearish' | 'neutral'."""
        if self.weighted_sentiment > 0.2:
            return "bullish"
        if self.weighted_sentiment < -0.2:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict[str, float | int | str | list[dict]]:  # type: ignore[type-arg]
        """Return serialisable representation."""
        return {
            "avg_sentiment": round(self.avg_sentiment, 4),
            "avg_relevance": round(self.avg_relevance, 4),
            "weighted_sentiment": round(self.weighted_sentiment, 4),
            "signal": self.signal,
            "article_count": self.article_count,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "top_articles": [
                {
                    "title": s.article.title,
                    "source": s.article.source,
                    "sentiment": round(s.sentiment, 3),
                    "relevance": round(s.relevance, 3),
                    "reasoning": s.reasoning,
                }
                for s in sorted(self.scores, key=lambda x: abs(x.weighted_score), reverse=True)[:5]
            ],
        }


class SentimentAnalyser:
    """Score news articles using Claude.

    Args:
        api_key: Anthropic API key.
        model: Claude model ID.
        max_articles: Max articles to send per API call (to control cost).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_articles: int = 15,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_articles = max_articles

    def analyse(
        self,
        symbol: str,
        company_name: str,
        articles: list[Article],
    ) -> AggregateSentiment:
        """Score a list of articles and return aggregate sentiment.

        Args:
            symbol: NSE/BSE symbol for logging context.
            company_name: Full company name for context.
            articles: News articles fetched by :class:`NewsFetcher`.

        Returns:
            An :class:`AggregateSentiment` instance.
        """
        if not articles:
            return _empty_aggregate()

        # Truncate to budget
        batch = articles[: self._max_articles]

        article_list = [
            {"index": i, "title": a.title, "description": a.description, "source": a.source}
            for i, a in enumerate(batch)
        ]
        user_message = (
            f"Stock: {company_name} ({symbol})\n\n"
            f"Articles to score:\n{json.dumps(article_list, indent=2)}"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            # Claude might wrap in markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scored = json.loads(raw)
        except Exception as exc:
            log.error("sentiment_analysis_failed", symbol=symbol, error=str(exc))
            return _empty_aggregate()

        score_objects: list[SentimentScore] = []
        for item in scored:
            idx = item.get("index", 0)
            if idx >= len(batch):
                continue
            score_objects.append(
                SentimentScore(
                    article=batch[idx],
                    relevance=float(item.get("relevance", 0.0)),
                    sentiment=float(item.get("sentiment", 0.0)),
                    reasoning=str(item.get("reasoning", "")),
                )
            )

        return _aggregate(score_objects)


def _empty_aggregate() -> AggregateSentiment:
    return AggregateSentiment(
        scores=[],
        avg_sentiment=0.0,
        avg_relevance=0.0,
        weighted_sentiment=0.0,
        article_count=0,
        bullish_count=0,
        bearish_count=0,
    )


def _aggregate(scores: list[SentimentScore]) -> AggregateSentiment:
    if not scores:
        return _empty_aggregate()
    avg_sent = sum(s.sentiment for s in scores) / len(scores)
    avg_rel = sum(s.relevance for s in scores) / len(scores)
    # Relevance-weighted sentiment
    total_weight = sum(s.relevance for s in scores) or 1.0
    weighted_sent = sum(s.weighted_score for s in scores) / total_weight
    bullish = sum(1 for s in scores if s.sentiment > 0.1)
    bearish = sum(1 for s in scores if s.sentiment < -0.1)
    return AggregateSentiment(
        scores=scores,
        avg_sentiment=avg_sent,
        avg_relevance=avg_rel,
        weighted_sentiment=weighted_sent,
        article_count=len(scores),
        bullish_count=bullish,
        bearish_count=bearish,
    )
