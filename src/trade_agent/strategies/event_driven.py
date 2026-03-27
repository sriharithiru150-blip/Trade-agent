"""Event-driven positional strategy.

Entry thesis
------------
A trade is opened only when:
1. A material corporate event has been filed (earnings, buyback, M&A, etc.)
   within the last 90 minutes.
2. Institutional flow (bulk/block deals) corroborates the direction, OR is
   absent (neutral) and the event alone is high-urgency.
3. Options positioning does NOT contradict the thesis (PCR not screaming the
   opposite direction).
4. The stock passes the liquidity filter — illiquid names are skipped
   regardless of signal quality.
5. The combined conviction score from EventScorer is ≥ the configured threshold.

Holding period: swing (2–10 sessions), NOT intraday.
Stop-loss is set beyond the break-even cost threshold so we don't get stopped
out by noise before the thesis has time to play out.

What this strategy deliberately does NOT do
-------------------------------------------
- Trade on RSI / MACD in isolation — those are secondary confirmation only.
- Enter purely on news from RSS / NewsAPI (stale, already priced in).
- Enter on every board meeting announcement (most are no-ops).
- Size into illiquid stocks where slippage kills the edge.
"""

from __future__ import annotations

import pandas as pd

from trade_agent.analysis.event_scorer import EventEvidence
from trade_agent.analysis.liquidity import LiquidityProfile, SlippageModel, compute_slippage
from trade_agent.analysis.technical import TechnicalSnapshot
from trade_agent.data.events import EventType
from trade_agent.strategies.base import BaseStrategy, StrategyResult
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

# Minimum conviction from EventScorer to consider entry
_MIN_CONVICTION = 0.60

# Minimum urgency of at least one event in the bundle
_MIN_URGENCY = 0.55

# ATR-based stop placement parameters
_ATR_STOP_MULTIPLIER = 1.5      # stop placed 1.5×ATR below entry
_ATR_RR_RATIO = 2.5             # take-profit at 2.5× the risk distance (ATR)

# Percentage-based fallback if ATR cannot be computed (< 15 rows of history)
_SWING_STOP_LOSS_PCT = 0.03     # 3% stop fallback (swing)
_SWING_TAKE_PROFIT_PCT = 0.09   # 9% target fallback (swing)
_INTRADAY_STOP_LOSS_PCT = 0.005  # 0.5% stop fallback (intraday, covers break-even)
_INTRADAY_TAKE_PROFIT_PCT = 0.012  # 1.2% target fallback (intraday, 2.4:1 R:R)

# Events that on their own are sufficient to trigger analysis
_STANDALONE_TRIGGER_EVENTS = {
    EventType.RESULTS,
    EventType.MERGER_ACQUISITION,
    EventType.BUYBACK,
    EventType.DEMERGER,
    EventType.DEBT_DEFAULT,
    EventType.SEBI_ORDER,
    EventType.QIP,
}


class EventDrivenStrategy(BaseStrategy):
    """Enter swing positions when material exchange-filed events occur.

    Args:
        min_conviction: Minimum EventScorer conviction (0–1) to open.
        stop_loss_pct: Swing stop-loss as fraction of entry.
        take_profit_pct: Swing take-profit as fraction of entry.
        max_trade_amount: Max INR per position.
        require_technical_confirm: If True, technicals must not be bearish.
    """

    def __init__(
        self,
        min_conviction: float = _MIN_CONVICTION,
        stop_loss_pct: float = _SWING_STOP_LOSS_PCT,
        take_profit_pct: float = _SWING_TAKE_PROFIT_PCT,
        max_trade_amount: float = 10_000.0,
        require_technical_confirm: bool = True,
        intraday_mode: bool = False,
    ) -> None:
        if intraday_mode:
            stop_loss_pct = _INTRADAY_STOP_LOSS_PCT
            take_profit_pct = _INTRADAY_TAKE_PROFIT_PCT
        super().__init__(stop_loss_pct, take_profit_pct, max_trade_amount)
        self.min_conviction = min_conviction
        self.require_technical_confirm = require_technical_confirm
        self.intraday_mode = intraday_mode

    def evaluate(
        self,
        symbol: str,
        signal: object,                    # ignored — not used by this strategy
        history: pd.DataFrame,
        articles: list[object],            # ignored — events come through evidence
        evidence: EventEvidence | None = None,
        liquidity: LiquidityProfile | None = None,
        technical: TechnicalSnapshot | None = None,
    ) -> StrategyResult:
        """Evaluate event-driven entry conditions.

        Args:
            symbol: NSE symbol.
            signal: Unused (interface compatibility).
            history: Daily OHLCV DataFrame.
            articles: Unused (interface compatibility).
            evidence: Scored event bundle from EventScorer.
            liquidity: Liquidity profile from compute_liquidity_profile.
            technical: Optional technical snapshot for secondary confirmation.

        Returns:
            StrategyResult with LONG or SKIP.
        """
        entry_price = float(history["Close"].iloc[-1])
        sl_raw, tp, atr14 = self._calc_atr_levels(
            history, entry_price, _ATR_STOP_MULTIPLIER, _ATR_RR_RATIO
        )

        # ── Gate 1: liquidity ─────────────────────────────────────────────────
        if liquidity is not None and not liquidity.is_liquid:
            return self._skip(
                symbol, entry_price, sl_raw, tp,
                f"Illiquid: tier={liquidity.liquidity_tier}, "
                f"ADV=₹{liquidity.avg_daily_value_cr:.1f}Cr"
            )

        # ── Gate 2: must have event evidence ─────────────────────────────────
        if evidence is None or not evidence.events:
            return self._skip(symbol, entry_price, sl_raw, tp,
                              "No material corporate events in window")

        # ── Gate 3: at least one actionable event ────────────────────────────
        # Use intraday freshness window when in intraday mode (5/15/30 min vs 15/45/90 min)
        actionable = [
            e for e in evidence.events
            if e.urgency >= _MIN_URGENCY and (
                e.is_fresh_intraday if self.intraday_mode else e.is_fresh
            )
        ]
        window_label = "intraday" if self.intraday_mode else "swing"
        if not actionable:
            oldest_age = evidence.events[0].age_minutes if evidence.events else 0
            return self._skip(
                symbol, entry_price, sl_raw, tp,
                f"No high-urgency events within {window_label} freshness window "
                f"(oldest event: {oldest_age:.0f}m ago)"
            )

        # ── Gate 4: conviction threshold ─────────────────────────────────────
        if evidence.conviction < self.min_conviction:
            return self._skip(
                symbol, entry_price, sl_raw, tp,
                f"Conviction {evidence.conviction:.2f} < {self.min_conviction} threshold"
            )

        # ── Gate 5: bias must be bullish ──────────────────────────────────────
        if evidence.bias not in ("bullish",):
            return self._skip(
                symbol, entry_price, sl_raw, tp,
                f"Event bias is '{evidence.bias}' — only entering on bullish setups"
            )

        # ── Gate 6: technicals must not be firmly bearish (optional) ─────────
        if self.require_technical_confirm and technical is not None:
            if technical.trend_direction == "bearish" and technical.rsi_14 is not None:
                if technical.rsi_14 < 40:
                    return self._skip(
                        symbol, entry_price, sl_raw, tp,
                        f"Technicals firmly bearish (trend={technical.trend_direction}, "
                        f"RSI={technical.rsi_14:.0f}) — waiting for confirmation"
                    )

        # ── Gate 7: options must not violently disagree ───────────────────────
        if evidence.options is not None:
            if evidence.options.pcr_oi < 0.5:
                return self._skip(
                    symbol, entry_price, sl_raw, tp,
                    f"Options PCR={evidence.options.pcr_oi:.2f} very low — "
                    f"crowded calls, reversal risk"
                )

        # ── All gates passed — compute position ───────────────────────────────
        # Adjust stop-loss for slippage and transaction costs
        qty_raw = self._calc_quantity(entry_price)
        if liquidity is not None:
            # Cap quantity to max liquidity allows
            max_qty = int(liquidity.max_position_size // entry_price)
            qty_raw = min(qty_raw, max(1, max_qty))

        slippage: SlippageModel | None = None
        if liquidity is not None:
            slippage = compute_slippage(entry_price, qty_raw, liquidity, sl_raw)
            sl_adjusted = slippage.adjusted_stop_loss
            be_pct = slippage.break_even_pct
        else:
            sl_adjusted = sl_raw
            be_pct = 0.0

        # Ensure minimum R:R of 2:1 after cost adjustment
        risk = entry_price - sl_adjusted
        reward = tp - entry_price
        if risk > 0 and (reward / risk) < 2.0:
            # Widen take-profit to maintain ratio
            tp = round(entry_price + risk * 2.5, 2)

        # Build rationale
        top_event = actionable[0]
        stop_method = f"ATR14={atr14:.2f}" if atr14 > 0 else "pct-fallback"
        rationale_parts = [
            f"EventDriven: conviction={evidence.conviction:.2f} ({evidence.bias})",
            f"Trigger: [{top_event.event_type.value.upper()}] {top_event.headline[:80]}",
            f"Filed {top_event.age_minutes:.0f}m ago",
            f"Stop: {stop_method} ({_ATR_STOP_MULTIPLIER}×ATR)",
            evidence.flow_summary[:120] if evidence.flow_summary else "",
            evidence.options_summary[:100] if evidence.options_summary else "",
        ]
        if slippage:
            rationale_parts.append(
                f"Break-even: {slippage.break_even_pct*100:.2f}%, "
                f"Total cost: ₹{slippage.total_cost_inr:.0f}"
            )
        rationale = " | ".join(p for p in rationale_parts if p)

        if evidence.key_risks:
            rationale += f" | RISKS: {'; '.join(evidence.key_risks[:2])}"

        log.info(
            "event_strategy_entry",
            symbol=symbol,
            conviction=evidence.conviction,
            trigger_event=top_event.event_type.value,
            entry=entry_price,
            sl=sl_adjusted,
            tp=tp,
            qty=qty_raw,
            atr14=round(atr14, 2),
        )

        return StrategyResult(
            symbol=symbol,
            direction="LONG",
            entry_price=entry_price,
            stop_loss=sl_adjusted,
            take_profit=tp,
            quantity=qty_raw,
            confidence="high" if evidence.conviction >= 0.75 else "medium",
            rationale=rationale,
        )

    def _skip(
        self,
        symbol: str,
        entry: float,
        sl: float,
        tp: float,
        reason: str,
    ) -> StrategyResult:
        log.debug("event_strategy_skip", symbol=symbol, reason=reason)
        return StrategyResult(
            symbol=symbol,
            direction="SKIP",
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            quantity=0,
            confidence="low",
            rationale=reason,
        )
