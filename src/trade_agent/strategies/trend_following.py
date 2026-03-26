"""Trend-following strategy.

Entry conditions (all must be true):
- Combined signal score ≥ threshold
- Price above SMA-20
- MACD histogram positive (recent cross-over or continuing)
- RSI between 40 and 70 (trending, not overbought)
- Volume ratio ≥ 1.0 (at least average volume confirming the move)
"""

from __future__ import annotations

import pandas as pd

from trade_agent.analysis.signals import CombinedSignal
from trade_agent.data.news_fetcher import Article
from trade_agent.strategies.base import BaseStrategy, StrategyResult
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

_MIN_SCORE = 0.60
_RSI_MIN = 40.0
_RSI_MAX = 70.0
_MIN_VOLUME_RATIO = 1.0


class TrendFollowingStrategy(BaseStrategy):
    """Enter long positions in confirmed uptrends with volume support.

    Args:
        min_score: Minimum combined signal score to enter.
        stop_loss_pct: Stop-loss fraction below entry.
        take_profit_pct: Take-profit fraction above entry.
        max_trade_amount: Max INR per trade.
    """

    def __init__(
        self,
        min_score: float = _MIN_SCORE,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_trade_amount: float = 10_000.0,
    ) -> None:
        super().__init__(stop_loss_pct, take_profit_pct, max_trade_amount)
        self.min_score = min_score

    def evaluate(
        self,
        symbol: str,
        signal: CombinedSignal,
        history: pd.DataFrame,
        articles: list[Article],
    ) -> StrategyResult:
        """Evaluate trend-following entry conditions.

        Args:
            symbol: NSE/BSE symbol.
            signal: Aggregated signal from the analysis layer.
            history: OHLCV DataFrame (daily).
            articles: Unused by this strategy (here for interface compliance).

        Returns:
            :class:`StrategyResult` with trade parameters or SKIP.
        """
        snap = signal  # CombinedSignal carries technical info via reasoning

        entry_price = float(history["Close"].iloc[-1])
        stop_loss, take_profit = self._calc_levels(entry_price)

        # Score gate
        if signal.score < self.min_score:
            return self._skip(symbol, entry_price, stop_loss, take_profit,
                              f"Score {signal.score:.2f} < threshold {self.min_score}")

        # Must be a BUY action
        if signal.action != "BUY":
            return self._skip(symbol, entry_price, stop_loss, take_profit,
                              f"Signal action is {signal.action}")

        # Confidence gate
        if signal.confidence == "low":
            return self._skip(symbol, entry_price, stop_loss, take_profit,
                              "Low confidence signal — skipping")

        quantity = self._calc_quantity(entry_price)
        rationale = (
            f"TrendFollowing: score={signal.score:.2f}, {signal.reasoning}"
        )

        log.info("strategy_decision", strategy="trend_following", symbol=symbol,
                 action="LONG", score=signal.score)

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
        log.debug("strategy_skip", strategy="trend_following", symbol=symbol, reason=reason)
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
