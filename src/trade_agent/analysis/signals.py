"""Signal aggregation — combines technical and sentiment scores into a
single normalised trade signal.

Signal pipeline
---------------
1. Technical score  (0–1) from :class:`TechnicalSnapshot`
2. Sentiment score  (0–1) from :class:`AggregateSentiment`
3. Market breadth   (0–1) from Nifty/Sensex trend
4. Weighted combination → final score in [0, 1]

Score interpretation:
  ≥ 0.65  → BUY signal
  ≤ 0.35  → SELL / SHORT signal (not used for day-trading unless allowed)
  else    → HOLD / no action
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trade_agent.analysis.sentiment import AggregateSentiment
from trade_agent.analysis.technical import TechnicalSnapshot
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

Action = Literal["BUY", "SELL", "HOLD"]

# Weights must sum to 1.0
_WEIGHT_TECHNICAL = 0.50
_WEIGHT_SENTIMENT = 0.35
_WEIGHT_MARKET_BREADTH = 0.15


@dataclass
class CombinedSignal:
    """Aggregated trading signal for a single stock."""

    symbol: str
    action: Action
    score: float                 # 0–1 (higher = more bullish)
    technical_score: float       # 0–1
    sentiment_score: float       # 0–1
    market_breadth_score: float  # 0–1
    confidence: str              # "high" | "medium" | "low"
    reasoning: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "score": round(self.score, 4),
            "technical_score": round(self.technical_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "market_breadth_score": round(self.market_breadth_score, 4),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


def _technical_to_score(snap: TechnicalSnapshot) -> float:
    """Convert a TechnicalSnapshot to a 0–1 score."""
    score = 0.5  # neutral baseline

    # Trend direction
    if snap.trend_direction == "bullish":
        score += 0.15
    elif snap.trend_direction == "bearish":
        score -= 0.15

    # Momentum
    if snap.momentum_signal == "oversold":
        score += 0.15  # potential bounce
    elif snap.momentum_signal == "overbought":
        score -= 0.10

    # MACD crossover
    if snap.macd_diff is not None:
        if snap.macd_diff > 0:
            score += 0.10
        else:
            score -= 0.10

    # Bollinger Band position
    if snap.bb_pct is not None:
        if snap.bb_pct < 0.20:
            score += 0.10  # near lower band — possible mean reversion up
        elif snap.bb_pct > 0.80:
            score -= 0.10

    # Volume confirmation
    if snap.volume_ratio is not None:
        if snap.volume_ratio > 1.5 and snap.trend_direction == "bullish":
            score += 0.05  # high volume bull — strong confirmation
        elif snap.volume_ratio > 1.5 and snap.trend_direction == "bearish":
            score -= 0.05

    # Clamp
    return max(0.0, min(1.0, score))


def _sentiment_to_score(sentiment: AggregateSentiment) -> float:
    """Convert an AggregateSentiment to a 0–1 score."""
    if sentiment.article_count == 0:
        return 0.5  # no news → neutral
    # weighted_sentiment is in [-1, 1]; map to [0, 1]
    return max(0.0, min(1.0, (sentiment.weighted_sentiment + 1.0) / 2.0))


def _market_breadth_to_score(nifty_change_pct: float | None) -> float:
    """Map the Nifty 50 day change % to a breadth score (0–1)."""
    if nifty_change_pct is None:
        return 0.5
    # +2% → 0.9, 0% → 0.5, -2% → 0.1
    return max(0.0, min(1.0, 0.5 + (nifty_change_pct / 4.0)))


def compute_signal(
    symbol: str,
    technical: TechnicalSnapshot,
    sentiment: AggregateSentiment,
    nifty_change_pct: float | None = None,
    min_score_to_act: float = 0.65,
) -> CombinedSignal:
    """Compute a combined trading signal.

    Args:
        symbol: NSE/BSE stock symbol.
        technical: Output from :func:`compute_technical_snapshot`.
        sentiment: Output from :class:`SentimentAnalyser`.
        nifty_change_pct: Nifty 50 day change percentage (optional market context).
        min_score_to_act: Minimum score to trigger a BUY action.

    Returns:
        A :class:`CombinedSignal` with a final score and recommended action.
    """
    tech_score = _technical_to_score(technical)
    sent_score = _sentiment_to_score(sentiment)
    breadth_score = _market_breadth_to_score(nifty_change_pct)

    final_score = (
        _WEIGHT_TECHNICAL * tech_score
        + _WEIGHT_SENTIMENT * sent_score
        + _WEIGHT_MARKET_BREADTH * breadth_score
    )

    # Action
    if final_score >= min_score_to_act:
        action: Action = "BUY"
    elif final_score <= (1.0 - min_score_to_act):
        action = "SELL"
    else:
        action = "HOLD"

    # Confidence based on agreement between components
    spread = max(tech_score, sent_score, breadth_score) - min(tech_score, sent_score, breadth_score)
    if spread < 0.2:
        confidence = "high"
    elif spread < 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    reasoning = (
        f"Technical={tech_score:.2f} ({technical.trend_direction}, "
        f"RSI={technical.rsi_14 or 'N/A'}), "
        f"Sentiment={sent_score:.2f} ({sentiment.signal}, "
        f"{sentiment.article_count} articles), "
        f"Breadth={breadth_score:.2f}"
    )

    log.info(
        "signal_computed",
        symbol=symbol,
        action=action,
        score=round(final_score, 4),
        confidence=confidence,
    )

    return CombinedSignal(
        symbol=symbol,
        action=action,
        score=round(final_score, 4),
        technical_score=round(tech_score, 4),
        sentiment_score=round(sent_score, 4),
        market_breadth_score=round(breadth_score, 4),
        confidence=confidence,
        reasoning=reasoning,
    )
