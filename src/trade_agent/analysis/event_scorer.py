"""Event scorer — combines corporate events, institutional flows, and options
positioning into a single conviction score for Claude to reason over.

The score is deliberately NOT a trading signal on its own.  It is structured
evidence that Claude uses to form a thesis, check against technicals, and then
decide.  The goal is to surface the *right questions*, not automate the answer.

Scoring model
-------------
Each input dimension produces a sub-score in [-1, 1]:
  event_score     — what the corporate events imply
  flow_score      — what institutional bulk/block deals imply
  options_score   — what options positioning implies (0.5 neutral)
  age_penalty     — freshness weight (older events carry less conviction)

The final conviction score maps to [0, 1]:
  0.0–0.35  strong bearish evidence
  0.35–0.55 ambiguous / insufficient data
  0.55–0.75 moderate bullish evidence
  0.75–1.0  strong bullish evidence

Claude is told the sub-scores AND the evidence so it can override or discount
based on its broader understanding of the company / market context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trade_agent.data.events import CorporateEvent, BulkDeal, EventType
from trade_agent.data.options_chain import OptionsChainSnapshot
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

Bias = Literal["bullish", "bearish", "neutral", "mixed"]


@dataclass
class EventEvidence:
    """Structured evidence bundle passed to Claude."""

    symbol: str

    # Sub-scores in [-1, 1]
    event_score: float
    flow_score: float
    options_score: float   # already 0–1 from OptionsChainSnapshot.signal_score

    # Aggregated conviction in [0, 1]
    conviction: float
    bias: Bias

    # Human-readable evidence summary for Claude
    event_summary: str
    flow_summary: str
    options_summary: str
    key_risks: list[str]

    # Raw inputs for Claude to inspect
    events: list[CorporateEvent]
    deals: list[BulkDeal]
    options: OptionsChainSnapshot | None

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "conviction": round(self.conviction, 4),
            "bias": self.bias,
            "event_score": round(self.event_score, 4),
            "flow_score": round(self.flow_score, 4),
            "options_score": round(self.options_score, 4),
            "event_summary": self.event_summary,
            "flow_summary": self.flow_summary,
            "options_summary": self.options_summary,
            "key_risks": self.key_risks,
            "events": [e.to_dict() for e in self.events],
            "deals": [d.to_dict() for d in self.deals],
            "options": self.options.to_dict() if self.options else None,
        }


class EventScorer:
    """Combines event, flow, and options data into conviction scores."""

    def score(
        self,
        symbol: str,
        events: list[CorporateEvent],
        deals: list[BulkDeal],
        options: OptionsChainSnapshot | None,
    ) -> EventEvidence:
        """Compute structured evidence for ``symbol``.

        Args:
            symbol: NSE stock symbol.
            events: Recent corporate announcements from :class:`EventScraper`.
            deals: Today's bulk/block deals from :class:`EventScraper`.
            options: Options chain snapshot from :class:`OptionsChainClient`.

        Returns:
            :class:`EventEvidence` with all sub-scores and summaries.
        """
        event_score, event_summary, event_risks = self._score_events(events)
        flow_score, flow_summary, flow_risks = self._score_flows(deals)
        opt_score, opt_summary, opt_risks = self._score_options(options)

        # Weighted combination (events carry most weight since they're exchange-filed)
        # Weights: events 45%, flow 35%, options 20%
        raw = (0.45 * event_score + 0.35 * flow_score + 0.20 * (opt_score * 2 - 1))
        # raw is in [-1, 1]; map to [0, 1]
        conviction = round(min(1.0, max(0.0, (raw + 1.0) / 2.0)), 4)

        bias = _bias(conviction)
        all_risks = event_risks + flow_risks + opt_risks

        log.info(
            "event_scored",
            symbol=symbol,
            conviction=conviction,
            bias=bias,
            events=len(events),
            deals=len(deals),
        )

        return EventEvidence(
            symbol=symbol,
            event_score=round(event_score, 4),
            flow_score=round(flow_score, 4),
            options_score=round(opt_score, 4),
            conviction=conviction,
            bias=bias,
            event_summary=event_summary,
            flow_summary=flow_summary,
            options_summary=opt_summary,
            key_risks=all_risks,
            events=events,
            deals=deals,
            options=options,
        )

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    @staticmethod
    def _score_events(
        events: list[CorporateEvent],
    ) -> tuple[float, str, list[str]]:
        """Score corporate events. Returns (score, summary, risks)."""
        if not events:
            return 0.0, "No recent corporate announcements.", []

        total_weight = 0.0
        weighted_dir = 0.0
        highlights: list[str] = []
        risks: list[str] = []

        for evt in events:
            # Freshness weight: events older than 60 min carry half weight
            freshness = max(0.1, 1.0 - (evt.age_minutes / 120))
            weight = evt.urgency * freshness
            direction = float(evt.direction)

            weighted_dir += direction * weight
            total_weight += weight

            highlights.append(
                f"[{evt.event_type.value.upper()}] {evt.headline} "
                f"(filed {evt.age_minutes:.0f}m ago)"
            )

            # Risk flags
            if evt.event_type in (EventType.DEBT_DEFAULT, EventType.SEBI_ORDER):
                risks.append(f"HIGH RISK: {evt.event_type.value} — {evt.headline[:80]}")
            if evt.event_type == EventType.QIP:
                risks.append("Dilution risk from QIP/rights issue")
            if evt.event_type == EventType.PROMOTER_PLEDGE:
                risks.append("Promoter pledging — potential forced selling")

        score = (weighted_dir / total_weight) if total_weight > 0 else 0.0
        summary = "; ".join(highlights[:5])
        return float(min(1.0, max(-1.0, score))), summary, risks

    @staticmethod
    def _score_flows(
        deals: list[BulkDeal],
    ) -> tuple[float, str, list[str]]:
        """Score institutional bulk/block deal flows. Returns (score, summary, risks)."""
        if not deals:
            return 0.0, "No bulk/block deals today.", []

        buy_cr = 0.0
        sell_cr = 0.0
        inst_buy_cr = 0.0
        inst_sell_cr = 0.0
        highlights: list[str] = []
        risks: list[str] = []

        for deal in deals:
            val = deal.value_cr
            if deal.transaction == "BUY":
                buy_cr += val
                if deal.is_institutional:
                    inst_buy_cr += val
            else:
                sell_cr += val
                if deal.is_institutional:
                    inst_sell_cr += val

            highlights.append(
                f"{deal.client_name[:30]} {deal.transaction} "
                f"₹{val:.1f}Cr @ ₹{deal.price:.1f}"
                + (f" [{deal.institution_type}]" if deal.is_institutional else "")
            )

        # Institutional net flow as a fraction of total traded value
        total = buy_cr + sell_cr or 1.0
        net_inst = inst_buy_cr - inst_sell_cr
        score = net_inst / max(total, 1.0)

        if inst_sell_cr > inst_buy_cr * 2:
            risks.append(
                f"Heavy institutional selling: ₹{inst_sell_cr:.1f}Cr vs "
                f"₹{inst_buy_cr:.1f}Cr buying"
            )

        summary = (
            f"Total bulk/block flows — Buy: ₹{buy_cr:.1f}Cr, Sell: ₹{sell_cr:.1f}Cr. "
            f"Institutional net: ₹{net_inst:+.1f}Cr. "
            + "; ".join(highlights[:3])
        )
        return float(min(1.0, max(-1.0, score * 3))), summary, risks

    @staticmethod
    def _score_options(
        options: OptionsChainSnapshot | None,
    ) -> tuple[float, str, list[str]]:
        """Score options positioning. Returns (0–1 score, summary, risks)."""
        if options is None:
            return 0.5, "No options data available.", []

        risks: list[str] = []
        d = options.to_dict()

        if options.pcr_oi > 1.5:
            risks.append(f"Extreme PCR ({options.pcr_oi:.2f}) — potential mean reversion")
        if options.pcr_oi < 0.5:
            risks.append(f"Very low PCR ({options.pcr_oi:.2f}) — crowded longs, reversal risk")
        if options.iv_skew > 5:
            risks.append("Elevated put IV skew — market hedging downside")

        price = options.underlying_price
        pain_gap = ((options.max_pain - price) / price * 100) if price else 0
        summary = (
            f"PCR(OI)={options.pcr_oi:.2f} ({options.sentiment}), "
            f"MaxPain=₹{options.max_pain:.0f} ({pain_gap:+.1f}% from CMP), "
            f"Call wall=₹{options.call_oi_wall:.0f}, "
            f"Put wall=₹{options.put_oi_wall:.0f}, "
            f"IV skew={options.iv_skew:.1f}"
        )
        if options.unusual_call_strikes:
            summary += f". Unusual call OI at {options.unusual_call_strikes[:2]}"
        if options.unusual_put_strikes:
            summary += f". Unusual put OI at {options.unusual_put_strikes[:2]}"

        return options.signal_score, summary, risks


def _bias(conviction: float) -> Bias:
    if conviction >= 0.65:
        return "bullish"
    if conviction <= 0.35:
        return "bearish"
    return "neutral"
