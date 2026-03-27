"""Liquidity filter and slippage model.

Before sizing any position the agent must verify:
1. The stock is liquid enough that the full position can be entered/exited
   without moving the price against itself.
2. Slippage, impact cost, and transaction charges are factored into the
   risk/reward calculation — a position that looks profitable gross may be
   a loser after costs.

Transaction cost breakdown (NSE, intraday MIS orders)
------------------------------------------------------
STT (Securities Transaction Tax)   : 0.025% on sell side only
Exchange transaction charge        : 0.00345% each way
SEBI turnover fee                  : 0.0001% each way
Stamp duty                         : 0.003% on buy side
GST (on brokerage + charges)       : 18%
Brokerage (Zerodha flat)           : ₹20 per executed order (or 0.03%, whichever lower)
Impact cost / bid-ask spread       : ~0.05–0.3% depending on liquidity

Break-even calculation
-----------------------
For a round-trip intraday trade you must gain ≥ ~0.15–0.25% just to cover costs
before slippage.  On a ₹10,000 position that's ₹15–25 minimum.  Stop-losses set
tighter than this will fire in the noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

# ── Transaction cost constants ────────────────────────────────────────────────

_STT_SELL = 0.00025          # 0.025% on sell value
_EXCHANGE_CHARGE = 0.0000345 # each way
_SEBI_FEE = 0.000001         # each way
_STAMP_DUTY_BUY = 0.00003    # on buy value
_GST_RATE = 0.18
_BROKERAGE_PER_ORDER = 20.0  # ₹ (Zerodha flat rate)
_BROKERAGE_PCT = 0.0003      # 0.03% cap


@dataclass
class LiquidityProfile:
    """Liquidity and cost characteristics for a symbol."""

    symbol: str
    avg_daily_volume: int
    avg_daily_value_cr: float      # average daily traded value in crores
    adv_20: int                    # 20-day average daily volume
    atr_pct: float                 # ATR as % of price (proxy for bid-ask + noise)
    is_liquid: bool                # passes minimum liquidity threshold
    liquidity_tier: str            # "high", "medium", "low", "illiquid"
    estimated_spread_pct: float    # estimated bid-ask spread %
    round_trip_cost_pct: float     # all-in cost for entry + exit
    max_position_size: float       # max INR to deploy without significant impact

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "adv_20": self.adv_20,
            "avg_daily_value_cr": round(self.avg_daily_value_cr, 2),
            "atr_pct": round(self.atr_pct, 4),
            "is_liquid": self.is_liquid,
            "liquidity_tier": self.liquidity_tier,
            "estimated_spread_pct": round(self.estimated_spread_pct, 4),
            "round_trip_cost_pct": round(self.round_trip_cost_pct, 4),
            "max_position_size": round(self.max_position_size, 0),
        }


@dataclass
class SlippageModel:
    """Per-trade cost and slippage estimate."""

    entry_cost_inr: float     # brokerage + charges on entry
    exit_cost_inr: float      # brokerage + charges + STT on exit
    estimated_slippage_inr: float  # market impact on entry + exit
    total_cost_inr: float
    break_even_pct: float     # price must move this much just to cover costs
    adjusted_stop_loss: float  # stop-loss moved out to absorb costs

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_cost_inr": round(self.entry_cost_inr, 2),
            "exit_cost_inr": round(self.exit_cost_inr, 2),
            "estimated_slippage_inr": round(self.estimated_slippage_inr, 2),
            "total_cost_inr": round(self.total_cost_inr, 2),
            "break_even_pct": round(self.break_even_pct, 4),
            "adjusted_stop_loss": round(self.adjusted_stop_loss, 2),
        }


# ── Minimum liquidity thresholds ─────────────────────────────────────────────

# Minimum 20-day average daily turnover to be tradeable for our position sizes
_MIN_ADV_CR = 5.0            # ₹5 crore minimum for swing positions
_MIN_ADV_CR_INTRADAY = 50.0  # ₹50 crore minimum for intraday — need tight spreads
_IMPACT_COST_THRESHOLD = 0.10  # maximum acceptable impact cost %


def compute_liquidity_profile(
    symbol: str,
    history: pd.DataFrame,
    current_price: float,
    max_trade_amount: float = 10_000.0,
    intraday_mode: bool = False,
) -> LiquidityProfile:
    """Compute liquidity characteristics from OHLCV history.

    Args:
        symbol: NSE stock symbol.
        history: Daily OHLCV DataFrame (at least 20 rows).
        current_price: Latest price for impact cost estimation.
        max_trade_amount: Intended position size in INR.
        intraday_mode: When True, applies ₹50Cr ADV floor instead of ₹5Cr.
            Intraday positions need tighter spreads and faster fills.

    Returns:
        :class:`LiquidityProfile` with tier classification and cost estimates.
    """
    if len(history) < 5:
        return _illiquid_profile(symbol, current_price)

    close = history["Close"]
    volume = history["Volume"]

    adv_20 = int(volume.tail(20).mean())
    avg_daily_value_cr = float((close.tail(20) * volume.tail(20)).mean()) / 1e7

    # ATR % as a proxy for noise / spread
    high = history["High"]
    low = history["Low"]
    tr = (high - low).tail(20)
    atr_pct = float((tr / close).tail(20).mean()) * 100

    # Bid-ask spread estimate: roughly 0.1–0.5 × ATR for liquid stocks
    estimated_spread_pct = max(0.03, min(0.5, atr_pct * 0.15))

    # Impact cost: our position as a fraction of ADV
    position_as_adv_pct = (max_trade_amount / max(avg_daily_value_cr * 1e7, 1)) * 100
    impact_pct = max(0.0, (position_as_adv_pct - 0.1) * 0.5)  # starts above 0.1% of ADV

    # All-in round-trip cost
    round_trip_cost_pct = _calc_round_trip_cost(
        trade_value=max_trade_amount,
        spread_pct=estimated_spread_pct,
        impact_pct=impact_pct,
    )

    # Tier classification
    # Intraday requires tighter spreads: raise the floor to ₹50Cr
    min_adv = _MIN_ADV_CR_INTRADAY if intraday_mode else _MIN_ADV_CR
    if avg_daily_value_cr >= 100:
        tier = "high"
        is_liquid = True
        max_pos = min(max_trade_amount, avg_daily_value_cr * 1e7 * 0.005)  # 0.5% of ADV
    elif avg_daily_value_cr >= 50:
        tier = "medium"
        is_liquid = True
        max_pos = min(max_trade_amount, avg_daily_value_cr * 1e7 * 0.003)
    elif avg_daily_value_cr >= 20:
        tier = "low"
        is_liquid = not intraday_mode and impact_pct < _IMPACT_COST_THRESHOLD
        max_pos = min(max_trade_amount, avg_daily_value_cr * 1e7 * 0.001)
    elif avg_daily_value_cr >= min_adv:
        tier = "low"
        is_liquid = impact_pct < _IMPACT_COST_THRESHOLD
        max_pos = min(max_trade_amount, avg_daily_value_cr * 1e7 * 0.001)
    else:
        tier = "illiquid"
        is_liquid = False
        max_pos = 0.0

    log.debug(
        "liquidity_profile",
        symbol=symbol,
        tier=tier,
        adv_cr=round(avg_daily_value_cr, 1),
        spread_pct=round(estimated_spread_pct, 3),
        impact_pct=round(impact_pct, 3),
    )

    return LiquidityProfile(
        symbol=symbol,
        avg_daily_volume=adv_20,
        avg_daily_value_cr=avg_daily_value_cr,
        adv_20=adv_20,
        atr_pct=atr_pct,
        is_liquid=is_liquid,
        liquidity_tier=tier,
        estimated_spread_pct=estimated_spread_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        max_position_size=max_pos,
    )


def compute_slippage(
    entry_price: float,
    quantity: int,
    liquidity: LiquidityProfile,
    stop_loss_raw: float,
) -> SlippageModel:
    """Compute all-in trade costs and adjust stop-loss.

    Args:
        entry_price: Expected entry price.
        quantity: Number of shares.
        liquidity: Liquidity profile from :func:`compute_liquidity_profile`.
        stop_loss_raw: Initial stop-loss before cost adjustment.

    Returns:
        :class:`SlippageModel` with adjusted stop-loss and break-even point.
    """
    trade_value = entry_price * quantity

    # Entry charges (no STT on buy for equity intraday)
    brokerage_entry = min(_BROKERAGE_PER_ORDER, trade_value * _BROKERAGE_PCT)
    exchange_entry = trade_value * _EXCHANGE_CHARGE
    sebi_entry = trade_value * _SEBI_FEE
    stamp_buy = trade_value * _STAMP_DUTY_BUY
    gst_entry = (brokerage_entry + exchange_entry) * _GST_RATE
    entry_cost = brokerage_entry + exchange_entry + sebi_entry + stamp_buy + gst_entry

    # Exit charges (STT applies on sell)
    brokerage_exit = min(_BROKERAGE_PER_ORDER, trade_value * _BROKERAGE_PCT)
    exchange_exit = trade_value * _EXCHANGE_CHARGE
    sebi_exit = trade_value * _SEBI_FEE
    stt_sell = trade_value * _STT_SELL
    gst_exit = (brokerage_exit + exchange_exit) * _GST_RATE
    exit_cost = brokerage_exit + exchange_exit + sebi_exit + stt_sell + gst_exit

    # Market impact / slippage (2 × spread, rounded up by impact cost)
    slippage = trade_value * (liquidity.estimated_spread_pct / 100) * 2

    total_cost = entry_cost + exit_cost + slippage
    break_even_pct = (total_cost / trade_value) if trade_value > 0 else 0

    # Adjust stop-loss outward to absorb costs (avoid firing in noise)
    cost_per_share = total_cost / quantity if quantity > 0 else 0
    adjusted_sl = round(stop_loss_raw - cost_per_share * 0.5, 2)

    return SlippageModel(
        entry_cost_inr=entry_cost,
        exit_cost_inr=exit_cost,
        estimated_slippage_inr=slippage,
        total_cost_inr=total_cost,
        break_even_pct=break_even_pct,
        adjusted_stop_loss=adjusted_sl,
    )


def _calc_round_trip_cost(
    trade_value: float,
    spread_pct: float,
    impact_pct: float,
) -> float:
    """Return all-in round-trip cost as a percentage of trade value."""
    brokerage = 2 * min(_BROKERAGE_PER_ORDER, trade_value * _BROKERAGE_PCT)
    stt = trade_value * _STT_SELL
    exchange = 2 * trade_value * _EXCHANGE_CHARGE
    sebi = 2 * trade_value * _SEBI_FEE
    stamp = trade_value * _STAMP_DUTY_BUY
    gst = (brokerage + exchange) * _GST_RATE
    spread_cost = trade_value * spread_pct / 100 * 2
    impact_cost = trade_value * impact_pct / 100 * 2
    total = brokerage + stt + exchange + sebi + stamp + gst + spread_cost + impact_cost
    return (total / trade_value) * 100 if trade_value > 0 else 0.0


def _illiquid_profile(symbol: str, price: float) -> LiquidityProfile:
    return LiquidityProfile(
        symbol=symbol,
        avg_daily_volume=0,
        avg_daily_value_cr=0.0,
        adv_20=0,
        atr_pct=0.0,
        is_liquid=False,
        liquidity_tier="illiquid",
        estimated_spread_pct=1.0,
        round_trip_cost_pct=2.0,
        max_position_size=0.0,
    )
