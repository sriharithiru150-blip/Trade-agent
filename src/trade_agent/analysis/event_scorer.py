"""Event scorer — combines corporate events, institutional flows, and options
positioning into a single conviction score for Claude to reason over.

Changes vs previous version
----------------------------
- flow_score now normalises against the stock's ADV (passed in), not just
  total traded value of the deals. A ₹2 Cr deal on ₹8 Cr ADV stock is
  significant; the same ₹2 Cr on ₹400 Cr ADV is noise. An absolute INR
  floor (₹1 Cr) filters deals too small to be informative regardless of ADV.
- event_score uses direction_score (float, contextual) instead of flat int.
  QIPs, credit rating changes, and management changes are now scored based
  on headline context rather than a fixed ±1.
- options_score is downweighted when the snapshot is stale (>5 min from
  NSE's public refresh cycle). Claude is explicitly told this in the summary.
- Conviction formula weights adjusted for the above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trade_agent.data.events import CorporateEvent, BulkDeal, EventType
from trade_agent.data.options_chain import OptionsChainSnapshot
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

Bias = Literal["bullish", "bearish", "neutral", "mixed"]

# Minimum bulk/block deal size to be scored (absolute floor)
_MIN_DEAL_CR = 1.0          # ignore deals < ₹1 Cr — too small to be informative

# Minimum deal size as fraction of ADV to be considered meaningful
_MIN_DEAL_ADV_FRACTION = 0.005   # 0.5% of ADV

# Weights — must sum to 1.0
_W_EVENT   = 0.45
_W_FLOW    = 0.35
_W_OPTIONS = 0.20


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
        adv_cr: float | None = None,
        intraday_mode: bool = False,
        options_staleness_minutes: float = 5.0,
    ) -> EventEvidence:
        """Compute structured evidence for ``symbol``.

        Args:
            symbol: NSE stock symbol.
            events: Recent corporate announcements from :class:`EventScraper`.
            deals: Today's bulk/block deals from :class:`EventScraper`.
            options: Options chain snapshot (may be stale — staleness is flagged).
            adv_cr: 20-day average daily traded value in crores. Used to
                normalise deal sizes. If None, falls back to relative sizing.
            intraday_mode: When True, applies tighter intraday freshness windows
                to event scoring (5/15/30 min instead of 15/45/90 min).
            options_staleness_minutes: Age threshold above which options weight
                is reduced.  Pass 2.0 for intraday, keep default 5.0 for swing.

        Returns:
            :class:`EventEvidence` with all sub-scores and summaries.
        """
        event_score, event_summary, event_risks = self._score_events(
            events, intraday_mode=intraday_mode
        )
        flow_score, flow_summary, flow_risks = self._score_flows(deals, adv_cr)
        opt_score, opt_summary, opt_risks, opt_weight = self._score_options(
            options, staleness_minutes=options_staleness_minutes
        )

        # Adjust weights if options data is stale — redistribute to events/flow
        if opt_weight < _W_OPTIONS:
            deficit = _W_OPTIONS - opt_weight
            w_evt = _W_EVENT + deficit * 0.6
            w_flw = _W_FLOW + deficit * 0.4
        else:
            w_evt, w_flw = _W_EVENT, _W_FLOW

        raw = w_evt * event_score + w_flw * flow_score + opt_weight * (opt_score * 2 - 1)
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
            opt_weight=round(opt_weight, 2),
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
        intraday_mode: bool = False,
    ) -> tuple[float, str, list[str]]:
        """Score corporate events using contextual direction_score.

        Uses event.direction_score (float, headline-aware) instead of the old
        flat integer EVENT_DIRECTION map.  Freshness is tiered by market cap:
        swing mode 15/45/90 min; intraday mode 5/15/30 min.
        """
        if not events:
            return 0.0, "No recent corporate announcements.", []

        total_weight = 0.0
        weighted_dir = 0.0
        highlights: list[str] = []
        risks: list[str] = []

        for evt in events:
            window = evt.intraday_freshness_window if intraday_mode else evt.freshness_window
            # Freshness: linear decay within the tier's window, zero beyond it
            if evt.age_minutes > window:
                continue   # outside the edge window for this market cap tier
            freshness = 1.0 - (evt.age_minutes / window)
            weight = evt.urgency * max(freshness, 0.05)
            direction = evt.direction_score   # float, contextual

            weighted_dir += direction * weight
            total_weight += weight

            mode_tag = "intra" if intraday_mode else "swing"
            highlights.append(
                f"[{evt.event_type.value.upper()}] {evt.headline[:80]} "
                f"(filed {evt.age_minutes:.0f}m ago, "
                f"window={window}m [{mode_tag}], dir={direction:+.1f})"
            )

            if evt.event_type in (EventType.DEBT_DEFAULT, EventType.SEBI_ORDER):
                risks.append(f"HIGH RISK: {evt.event_type.value} — {evt.headline[:80]}")
            if evt.event_type == EventType.QIP and direction < 0:
                risks.append("Distress-signal QIP — possible dilution under duress")
            if evt.event_type == EventType.PROMOTER_PLEDGE:
                risks.append("Promoter pledging — potential forced selling pressure")

        if total_weight == 0:
            mode = "intraday" if intraday_mode else "swing"
            return 0.0, f"All events outside {mode} freshness window for this stock's market cap tier.", []

        score = float(min(1.0, max(-1.0, weighted_dir / total_weight)))
        summary = " | ".join(highlights[:5])
        return score, summary, risks

    @staticmethod
    def _score_flows(
        deals: list[BulkDeal],
        adv_cr: float | None,
    ) -> tuple[float, str, list[str]]:
        """Score institutional bulk/block deal flows.

        Normalises institutional net flow against ADV (if provided) so that
        the same absolute deal INR carries more weight on a low-ADV stock than
        on a high-ADV stock.  Deals below ₹1 Cr absolute or below 0.5% of ADV
        are filtered as noise.
        """
        if not deals:
            return 0.0, "No bulk/block deals today.", []

        inst_buy_cr = 0.0
        inst_sell_cr = 0.0
        retail_buy_cr = 0.0
        retail_sell_cr = 0.0
        highlights: list[str] = []
        risks: list[str] = []
        skipped = 0

        for deal in deals:
            val = deal.value_cr

            # Absolute floor
            if val < _MIN_DEAL_CR:
                skipped += 1
                continue

            # ADV-relative floor (if ADV known)
            if adv_cr is not None and adv_cr > 0:
                if val / adv_cr < _MIN_DEAL_ADV_FRACTION:
                    skipped += 1
                    continue

            if deal.transaction == "BUY":
                if deal.is_institutional:
                    inst_buy_cr += val
                else:
                    retail_buy_cr += val
            else:
                if deal.is_institutional:
                    inst_sell_cr += val
                else:
                    retail_sell_cr += val

            tag = f"[{deal.institution_type}]" if deal.is_institutional else "[RETAIL]"
            highlights.append(
                f"{deal.client_name[:28]} {deal.transaction} ₹{val:.1f}Cr {tag}"
            )

        net_inst = inst_buy_cr - inst_sell_cr
        total_inst = inst_buy_cr + inst_sell_cr or 1.0

        # Normalise: if ADV known, express as multiple of ADV; else use fraction of inst total
        if adv_cr is not None and adv_cr > 0:
            # Score based on net flow as % of ADV — 5% of ADV = full score
            score = float(min(1.0, max(-1.0, (net_inst / adv_cr) / 0.05)))
        else:
            score = float(min(1.0, max(-1.0, net_inst / total_inst)))

        if inst_sell_cr > inst_buy_cr * 2 and inst_sell_cr > 5.0:
            risks.append(
                f"Heavy institutional selling: ₹{inst_sell_cr:.1f}Cr vs "
                f"₹{inst_buy_cr:.1f}Cr buying"
            )

        adv_context = f" (ADV ₹{adv_cr:.0f}Cr)" if adv_cr else ""
        summary = (
            f"Inst net{adv_context}: ₹{net_inst:+.1f}Cr "
            f"(buy ₹{inst_buy_cr:.1f}Cr / sell ₹{inst_sell_cr:.1f}Cr). "
            + "; ".join(highlights[:4])
            + (f" [{skipped} deals below floor filtered]" if skipped else "")
        )
        return score, summary, risks

    @staticmethod
    def _score_options(
        options: OptionsChainSnapshot | None,
        staleness_minutes: float = 5.0,
    ) -> tuple[float, str, list[str], float]:
        """Score options positioning. Returns (0–1 score, summary, risks, weight).

        If the snapshot is stale (older than ``staleness_minutes``), the
        effective weight returned to the caller is reduced so Claude and the
        conviction formula are both informed of the reduced confidence.

        Args:
            options: Options chain snapshot, or None.
            staleness_minutes: Age threshold to call data stale.  Use 2.0 for
                intraday mode, 5.0 (default) for swing.
        """
        if options is None:
            return 0.5, "No options data available.", [], 0.0

        risks: list[str] = []
        age = options.data_age_minutes

        # Staleness: linearly reduce weight from full at 0 to 0 at 2×threshold
        decay_window = staleness_minutes * 2.0
        staleness_factor = max(0.0, 1.0 - (age / decay_window))
        effective_weight = round(_W_OPTIONS * staleness_factor, 4)
        is_stale = age > staleness_minutes

        if is_stale:
            risks.append(
                f"Options data is {age:.1f} min old (threshold: {staleness_minutes} min). "
                f"On event days this covers the full tradeable window — "
                f"options weight reduced to {effective_weight:.2f} (from {_W_OPTIONS})."
            )
        if options.pcr_oi > 1.5:
            risks.append(f"Extreme PCR ({options.pcr_oi:.2f}) — potential mean reversion")
        if options.pcr_oi < 0.5:
            risks.append(f"Very low PCR ({options.pcr_oi:.2f}) — crowded longs, reversal risk")
        if options.iv_skew > 5:
            risks.append("Elevated put IV skew — market paying for downside protection")

        price = options.underlying_price
        pain_gap = ((options.max_pain - price) / price * 100) if price else 0
        stale_tag = f" ⚠ DATA {age:.0f}m OLD" if is_stale else ""
        summary = (
            f"PCR(OI)={options.pcr_oi:.2f} ({options.sentiment}){stale_tag}, "
            f"MaxPain=₹{options.max_pain:.0f} ({pain_gap:+.1f}% from CMP), "
            f"Call wall=₹{options.call_oi_wall:.0f}, "
            f"Put wall=₹{options.put_oi_wall:.0f}, "
            f"IV skew={options.iv_skew:.1f}"
        )
        if options.unusual_call_strikes:
            summary += f". Unusual call OI at {options.unusual_call_strikes[:2]}"
        if options.unusual_put_strikes:
            summary += f". Unusual put OI at {options.unusual_put_strikes[:2]}"

        return options.signal_score, summary, risks, effective_weight


def _bias(conviction: float) -> Bias:
    if conviction >= 0.65:
        return "bullish"
    if conviction <= 0.35:
        return "bearish"
    return "neutral"
