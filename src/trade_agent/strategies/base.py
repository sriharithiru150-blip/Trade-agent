"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trade_agent.analysis.signals import CombinedSignal
from trade_agent.data.news_fetcher import Article

TradeDirection = Literal["LONG", "SKIP"]

# Default R:R ratio used when widening take-profit to meet minimum
_DEFAULT_RR_RATIO = 2.5


@dataclass
class StrategyResult:
    """Output from a strategy evaluation."""

    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    confidence: str           # "high" | "medium" | "low"
    rationale: str
    signal: CombinedSignal | None = None

    @property
    def should_trade(self) -> bool:
        """Return True if the strategy recommends opening a position."""
        return self.direction == "LONG" and self.quantity > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "quantity": self.quantity,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


class BaseStrategy(ABC):
    """All strategies must implement :meth:`evaluate`.

    Args:
        stop_loss_pct: Fraction below entry for stop-loss (e.g. 0.02 = 2%).
        take_profit_pct: Fraction above entry for take-profit.
        max_trade_amount: Maximum INR to spend per trade.
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_trade_amount: float = 10_000.0,
    ) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trade_amount = max_trade_amount

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        signal: CombinedSignal,
        history: pd.DataFrame,
        articles: list[Article],
    ) -> StrategyResult:
        """Evaluate whether to trade and at what parameters.

        Args:
            symbol: NSE/BSE stock symbol.
            signal: Aggregated signal from the analysis layer.
            history: Daily OHLCV DataFrame (at least 50 rows preferred).
            articles: Recent news articles for final rule checks.

        Returns:
            A :class:`StrategyResult` with full trade parameters.
        """

    def _calc_levels(
        self, entry_price: float
    ) -> tuple[float, float]:
        """Return (stop_loss, take_profit) prices for a LONG position."""
        stop_loss = round(entry_price * (1.0 - self.stop_loss_pct), 2)
        take_profit = round(entry_price * (1.0 + self.take_profit_pct), 2)
        return stop_loss, take_profit

    def _calc_atr_levels(
        self,
        history: pd.DataFrame,
        entry_price: float,
        atr_multiplier: float = 1.5,
        rr_ratio: float = _DEFAULT_RR_RATIO,
    ) -> tuple[float, float, float]:
        """Return (stop_loss, take_profit, atr14) using ATR-based sizing.

        Computes ATR-14 from the history DataFrame and places the stop-loss
        at ``entry_price - atr_multiplier * ATR14``.  Take-profit uses
        ``rr_ratio`` times the risk distance.  Falls back to
        ``_calc_levels()`` if history is too short or ATR is zero.

        Args:
            history: Daily OHLCV DataFrame with at least 15 rows.
            entry_price: Intended entry price.
            atr_multiplier: How many ATR units below entry to place the stop.
            rr_ratio: Risk-to-reward ratio for take-profit placement.

        Returns:
            (stop_loss, take_profit, atr14) — atr14 is 0.0 on fallback.
        """
        if len(history) < 15 or entry_price <= 0:
            sl, tp = self._calc_levels(entry_price)
            return sl, tp, 0.0

        high = history["High"]
        low = history["Low"]
        prev_close = history["Close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr14 = float(tr.rolling(14).mean().iloc[-1])

        if atr14 <= 0:
            sl, tp = self._calc_levels(entry_price)
            return sl, tp, 0.0

        stop_loss = round(entry_price - atr_multiplier * atr14, 2)
        take_profit = round(entry_price + atr_multiplier * atr14 * rr_ratio, 2)
        return stop_loss, take_profit, atr14

    def _calc_quantity(self, entry_price: float) -> int:
        """Return share quantity that fits within max_trade_amount."""
        if entry_price <= 0:
            return 0
        return max(1, int(self.max_trade_amount // entry_price))
