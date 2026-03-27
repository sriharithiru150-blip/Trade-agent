"""NSE options chain analyser.

Options flow is one of the few publicly available signals that genuinely leads
price — institutions hedge and position through options before the underlying
moves.  Key metrics:

Put-Call Ratio (PCR)
    PCR by OI > 1.2  → market expecting support (contrarian bullish)
    PCR by OI < 0.7  → complacency / stretched longs (contrarian bearish)
    Extreme readings (>1.5 or <0.5) often precede sharp reversals.

Max Pain
    The strike price at which the most options expire worthless.  Expiry week
    price tends to gravitate toward max pain as market makers delta-hedge.

Unusual OI Build-up
    A sudden spike in OI at a specific strike is an informed bet.  If large OI
    appears at a far-OTM call, someone may know about an upcoming catalyst.

IV Skew
    High IV on puts vs calls → market paying for downside protection (bearish).
    High IV on calls vs puts → speculation / squeeze potential (bullish).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

_NSE_OPTION_CHAIN_URL = (
    "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
)
_NSE_INDEX_OPTION_CHAIN_URL = (
    "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/option-chain",
}


@dataclass
class StrikeData:
    """OI and IV data for a single strike price."""

    strike: float
    call_oi: int
    call_oi_change: int
    call_iv: float
    call_ltp: float
    put_oi: int
    put_oi_change: int
    put_iv: float
    put_ltp: float

    @property
    def pcr(self) -> float | None:
        """Per-strike Put-Call ratio by OI."""
        if self.call_oi > 0:
            return self.put_oi / self.call_oi
        return None


@dataclass
class OptionsChainSnapshot:
    """Processed options chain for a single symbol / expiry."""

    symbol: str
    underlying_price: float
    expiry: str
    strikes: list[StrikeData]

    # Computed metrics
    pcr_oi: float = 0.0           # total put OI / total call OI
    pcr_volume: float = 0.0
    max_pain: float = 0.0
    call_oi_wall: float = 0.0     # strike with highest call OI (resistance)
    put_oi_wall: float = 0.0      # strike with highest put OI (support)
    iv_skew: float = 0.0          # avg put IV − avg call IV (positive = put skew)
    unusual_call_strikes: list[float] = field(default_factory=list)
    unusual_put_strikes: list[float] = field(default_factory=list)

    @property
    def sentiment(self) -> str:
        """Options-derived sentiment label."""
        if self.pcr_oi > 1.3:
            return "bullish"     # hedging demand on puts = fear = contrarian up
        if self.pcr_oi < 0.7:
            return "bearish"     # call speculation = complacency = contrarian down
        return "neutral"

    @property
    def signal_score(self) -> float:
        """Normalised 0–1 options signal (0.5 = neutral)."""
        # PCR maps non-linearly: 0.7→0.3, 1.0→0.5, 1.3→0.7
        base = min(1.0, max(0.0, (self.pcr_oi - 0.4) / 1.2))
        # Slight skew adjustment: high put IV premium = bearish pressure
        skew_adj = -0.05 * min(1.0, max(-1.0, self.iv_skew / 10.0))
        return round(min(1.0, max(0.0, base + skew_adj)), 4)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "underlying_price": self.underlying_price,
            "expiry": self.expiry,
            "pcr_oi": round(self.pcr_oi, 4),
            "max_pain": self.max_pain,
            "call_oi_wall": self.call_oi_wall,
            "put_oi_wall": self.put_oi_wall,
            "iv_skew": round(self.iv_skew, 4),
            "sentiment": self.sentiment,
            "signal_score": self.signal_score,
            "unusual_call_strikes": self.unusual_call_strikes[:3],
            "unusual_put_strikes": self.unusual_put_strikes[:3],
        }


class OptionsChainClient:
    """Fetch and analyse NSE equity options chains.

    Args:
        timeout: HTTP timeout in seconds.
    """

    def __init__(self, timeout: int = 15) -> None:
        self._client = httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True)
        self._cookie_valid = False

    def _ensure_cookies(self) -> None:
        if not self._cookie_valid:
            try:
                self._client.get("https://www.nseindia.com")
                self._cookie_valid = True
            except Exception as exc:
                log.warning("options_cookie_failed", error=str(exc))

    def get_snapshot(
        self,
        symbol: str,
        expiry_index: int = 0,
        is_index: bool = False,
    ) -> OptionsChainSnapshot | None:
        """Fetch and compute the options chain snapshot for one symbol.

        Args:
            symbol: NSE equity symbol (e.g. ``RELIANCE``) or index
                    (e.g. ``NIFTY``, ``BANKNIFTY``).
            expiry_index: 0 = nearest expiry, 1 = next expiry, etc.
            is_index: True if ``symbol`` is an index.

        Returns:
            A :class:`OptionsChainSnapshot` or None on failure.
        """
        self._ensure_cookies()
        if is_index:
            url = _NSE_INDEX_OPTION_CHAIN_URL.format(symbol=symbol.upper())
        else:
            url = _NSE_OPTION_CHAIN_URL.format(symbol=symbol.upper())

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("options_chain_fetch_failed", symbol=symbol, error=str(exc))
            return None

        return self._parse(symbol, data, expiry_index)

    # ── Parser ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(
        symbol: str,
        data: dict[str, Any],
        expiry_index: int,
    ) -> OptionsChainSnapshot | None:
        """Parse the raw NSE option chain JSON."""
        try:
            records = data.get("records", {})
            underlying_price = float(records.get("underlyingValue", 0))
            expiry_dates: list[str] = records.get("expiryDates", [])

            if not expiry_dates:
                return None

            expiry = expiry_dates[min(expiry_index, len(expiry_dates) - 1)]
            raw_data: list[dict[str, Any]] = records.get("data", [])

            strikes: list[StrikeData] = []
            for row in raw_data:
                if row.get("expiryDate") != expiry:
                    continue
                strike = float(row.get("strikePrice", 0))
                ce = row.get("CE") or {}
                pe = row.get("PE") or {}
                strikes.append(StrikeData(
                    strike=strike,
                    call_oi=int(ce.get("openInterest", 0)),
                    call_oi_change=int(ce.get("changeinOpenInterest", 0)),
                    call_iv=float(ce.get("impliedVolatility", 0)),
                    call_ltp=float(ce.get("lastPrice", 0)),
                    put_oi=int(pe.get("openInterest", 0)),
                    put_oi_change=int(pe.get("changeinOpenInterest", 0)),
                    put_iv=float(pe.get("impliedVolatility", 0)),
                    put_ltp=float(pe.get("lastPrice", 0)),
                ))

            if not strikes:
                return None

            snap = OptionsChainSnapshot(
                symbol=symbol.upper(),
                underlying_price=underlying_price,
                expiry=expiry,
                strikes=strikes,
            )

            # Compute aggregate metrics
            total_call_oi = sum(s.call_oi for s in strikes)
            total_put_oi = sum(s.put_oi for s in strikes)
            snap.pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

            # Max pain
            snap.max_pain = _compute_max_pain(strikes)

            # OI walls (highest OI strike = support/resistance)
            max_call_s = max(strikes, key=lambda s: s.call_oi, default=None)
            max_put_s = max(strikes, key=lambda s: s.put_oi, default=None)
            snap.call_oi_wall = max_call_s.strike if max_call_s else 0.0
            snap.put_oi_wall = max_put_s.strike if max_put_s else 0.0

            # IV skew
            call_ivs = [s.call_iv for s in strikes if s.call_iv > 0]
            put_ivs = [s.put_iv for s in strikes if s.put_iv > 0]
            avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0
            avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else 0
            snap.iv_skew = avg_put_iv - avg_call_iv

            # Unusual OI: strikes with OI change > 2× average
            avg_call_change = (
                sum(abs(s.call_oi_change) for s in strikes) / len(strikes)
            ) if strikes else 1
            avg_put_change = (
                sum(abs(s.put_oi_change) for s in strikes) / len(strikes)
            ) if strikes else 1

            snap.unusual_call_strikes = [
                s.strike for s in strikes
                if s.call_oi_change > avg_call_change * 2
            ]
            snap.unusual_put_strikes = [
                s.strike for s in strikes
                if s.put_oi_change > avg_put_change * 2
            ]

            log.info(
                "options_chain_parsed",
                symbol=symbol,
                expiry=expiry,
                pcr=round(snap.pcr_oi, 3),
                max_pain=snap.max_pain,
                sentiment=snap.sentiment,
            )
            return snap

        except Exception as exc:
            log.error("options_chain_parse_error", symbol=symbol, error=str(exc))
            return None


def _compute_max_pain(strikes: list[StrikeData]) -> float:
    """Compute the max pain strike price.

    Max pain = the strike where total option value (calls + puts) at expiry
    would be minimised for option buyers (i.e. maximised for sellers).
    """
    if not strikes:
        return 0.0

    all_strikes = [s.strike for s in strikes]
    min_pain = float("inf")
    max_pain_strike = all_strikes[0]

    for candidate in all_strikes:
        # Value of all calls expiring in-the-money if price = candidate
        call_value = sum(
            s.call_oi * max(0.0, candidate - s.strike)
            for s in strikes
        )
        # Value of all puts expiring in-the-money if price = candidate
        put_value = sum(
            s.put_oi * max(0.0, s.strike - candidate)
            for s in strikes
        )
        total = call_value + put_value
        if total < min_pain:
            min_pain = total
            max_pain_strike = candidate

    return max_pain_strike
