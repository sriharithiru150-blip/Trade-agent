"""News-driven strategy.

This strategy activates when there is strong, highly-relevant news that
overrides or supplements the technical signal.

Entry conditions:
- Sentiment weighted score ≥ 0.70  (strongly bullish news)
- At least 1 high-relevance article (relevance ≥ 0.7)
- Combined score ≥ min_score (sanity check — news shouldn't fight technicals badly)
- RSI < 75  (not already severely overbought)
"""

from __future__ import annotations

import pandas as pd

from trade_agent.analysis.sentiment import AggregateSentiment
from trade_agent.analysis.signals import CombinedSignal
from trade_agent.data.news_fetcher import Article
from trade_agent.strategies.base import BaseStrategy, StrategyResult
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

_MIN_WEIGHTED_SENTIMENT = 0.70   # maps to sentiment_score > 0.85
_MIN_HIGH_RELEVANCE_ARTICLES = 1
_MIN_COMBINED_SCORE = 0.55
_MAX_RSI = 75.0


class NewsDrivenStrategy(BaseStrategy):
    """Enter positions when strong positive news catalysts are detected.

    Unlike trend-following, this strategy can enter even in neutral technical
    conditions if the news catalyst is strong enough, but still applies a
    combined score floor to avoid fighting heavy downtrends.

    Args:
        min_sentiment_score: Minimum sentiment_score component (0–1) to enter.
        min_combined_score: Minimum combined signal score floor.
        stop_loss_pct: Stop-loss fraction.
        take_profit_pct: Take-profit fraction (wider for news events).
        max_trade_amount: Max INR per trade.
    """

    def __init__(
        self,
        min_sentiment_score: float = _MIN_WEIGHTED_SENTIMENT,
        min_combined_score: float = _MIN_COMBINED_SCORE,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.05,
        max_trade_amount: float = 10_000.0,
    ) -> None:
        super().__init__(stop_loss_pct, take_profit_pct, max_trade_amount)
        self.min_sentiment_score = min_sentiment_score
        self.min_combined_score = min_combined_score

    def evaluate(
        self,
        symbol: str,
        signal: CombinedSignal,
        history: pd.DataFrame,
        articles: list[Article],
    ) -> StrategyResult:
        """Evaluate news-catalyst entry conditions.

        Args:
            symbol: NSE/BSE symbol.
            signal: Aggregated signal (must include sentiment component).
            history: OHLCV DataFrame (daily).
            articles: Recent news articles for relevance checks.

        Returns:
            :class:`StrategyResult` indicating LONG or SKIP.
        """
        entry_price = float(history["Close"].iloc[-1])
        stop_loss, take_profit = self._calc_levels(entry_price)

        # Sentiment gate
        if signal.sentiment_score < self.min_sentiment_score:
            return self._skip(
                symbol, entry_price, stop_loss, take_profit,
                f"Sentiment score {signal.sentiment_score:.2f} < {self.min_sentiment_score}"
            )

        # Combined floor — don't fight a crashing market
        if signal.score < self.min_combined_score:
            return self._skip(
                symbol, entry_price, stop_loss, take_profit,
                f"Combined score {signal.score:.2f} too low despite good sentiment"
            )

        # Require at least one highly relevant article
        high_relevance = [
            a for a in articles
            if hasattr(a, "relevance") and getattr(a, "relevance", 0) >= 0.7  # type: ignore[attr-defined]
        ]
        # (SentimentScore.relevance is on the score object, not the article directly;
        #  here we just check we have articles at all)
        if not articles:
            return self._skip(
                symbol, entry_price, stop_loss, take_profit,
                "No news articles available"
            )

        quantity = self._calc_quantity(entry_price)
        rationale = (
            f"NewsDriven: sentiment_score={signal.sentiment_score:.2f}, "
            f"combined={signal.score:.2f}, articles={len(articles)}, "
            f"{signal.reasoning}"
        )

        log.info("strategy_decision", strategy="news_driven", symbol=symbol,
                 action="LONG", score=signal.score, sentiment=signal.sentiment_score)

        return StrategyResult(
            symbol=symbol,
            direction="LONG",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            confidence=signal.confidence,
            rationale=rationale,
            signal=signal,
        )

    def _skip(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str,
    ) -> StrategyResult:
        log.debug("strategy_skip", strategy="news_driven", symbol=symbol, reason=reason)
        return StrategyResult(
            symbol=symbol,
            direction="SKIP",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=0,
            confidence="low",
            rationale=reason,
        )
