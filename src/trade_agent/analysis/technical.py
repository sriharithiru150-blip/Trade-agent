"""Technical analysis — computes indicators from OHLCV DataFrames.

Uses the ``ta`` library (https://github.com/bukosabino/ta) which wraps
pandas operations and requires no native C extensions.

All public functions return a :class:`TechnicalSnapshot` dataclass so
callers never need to inspect raw DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, SMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    _TA_AVAILABLE = True
except ImportError:  # graceful degradation for environments without ta
    _TA_AVAILABLE = False

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TechnicalSnapshot:
    """Latest values for all computed technical indicators."""

    # Price
    last_close: float
    sma_20: float | None
    sma_50: float | None
    ema_20: float | None

    # Momentum
    rsi_14: float | None          # 0–100; >70 overbought, <30 oversold
    stoch_k: float | None         # Stochastic %K
    stoch_d: float | None         # Stochastic %D

    # Trend
    macd: float | None            # MACD line
    macd_signal: float | None     # Signal line
    macd_diff: float | None       # Histogram

    # Volatility
    bb_upper: float | None
    bb_lower: float | None
    bb_pct: float | None          # Price position within Bollinger Bands (0–1)
    atr_14: float | None

    # Volume
    obv: float | None
    volume_ratio: float | None    # Current volume / 20-day average volume

    # Derived signals
    trend_direction: str = "neutral"   # "bullish" | "bearish" | "neutral"
    momentum_signal: str = "neutral"
    volatility_signal: str = "normal"  # "normal" | "high" | "squeeze"

    def to_dict(self) -> dict[str, float | str | None]:
        """Return a flat dict suitable for JSON serialisation."""
        return {
            "last_close": self.last_close,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "ema_20": self.ema_20,
            "rsi_14": self.rsi_14,
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_diff": self.macd_diff,
            "bb_upper": self.bb_upper,
            "bb_lower": self.bb_lower,
            "bb_pct": self.bb_pct,
            "atr_14": self.atr_14,
            "obv": self.obv,
            "volume_ratio": self.volume_ratio,
            "trend_direction": self.trend_direction,
            "momentum_signal": self.momentum_signal,
            "volatility_signal": self.volatility_signal,
        }


def _safe_float(series: pd.Series, idx: int = -1) -> float | None:
    """Extract a float from the last (or specified) row of a Series safely."""
    try:
        val = series.iloc[idx]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except (IndexError, TypeError):
        return None


def compute_technical_snapshot(df: pd.DataFrame) -> TechnicalSnapshot:
    """Compute all technical indicators for the given OHLCV DataFrame.

    Args:
        df: DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume`` and at least 50 rows for reliable indicator values.

    Returns:
        A :class:`TechnicalSnapshot` with the latest indicator values.

    Raises:
        ValueError: If ``df`` is missing required columns or has fewer than
            20 rows.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 20:
        raise ValueError(f"DataFrame has only {len(df)} rows; need at least 20 for indicators.")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])

    if not _TA_AVAILABLE:
        log.warning("ta_library_not_available", msg="Install 'ta' for technical indicators")
        return TechnicalSnapshot(last_close=last_close, **{k: None for k in [
            "sma_20", "sma_50", "ema_20", "rsi_14", "stoch_k", "stoch_d",
            "macd", "macd_signal", "macd_diff", "bb_upper", "bb_lower", "bb_pct",
            "atr_14", "obv", "volume_ratio",
        ]})

    # ── Moving averages ────────────────────────────────────────────────────────
    sma_20 = _safe_float(SMAIndicator(close, window=20).sma_indicator())
    sma_50 = _safe_float(SMAIndicator(close, window=50).sma_indicator()) if len(df) >= 50 else None
    ema_20 = _safe_float(EMAIndicator(close, window=20).ema_indicator())

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi_14 = _safe_float(RSIIndicator(close, window=14).rsi())

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    stoch_k = _safe_float(stoch.stoch())
    stoch_d = _safe_float(stoch.stoch_signal())

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = _safe_float(macd_ind.macd())
    macd_signal = _safe_float(macd_ind.macd_signal())
    macd_diff = _safe_float(macd_ind.macd_diff())

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_upper = _safe_float(bb.bollinger_hband())
    bb_lower = _safe_float(bb.bollinger_lband())
    bb_pct = _safe_float(bb.bollinger_pband())

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr_14 = _safe_float(AverageTrueRange(high, low, close, window=14).average_true_range())

    # ── OBV ───────────────────────────────────────────────────────────────────
    obv = _safe_float(OnBalanceVolumeIndicator(close, volume).on_balance_volume())

    # ── Volume ratio ─────────────────────────────────────────────────────────
    vol_avg_20 = float(volume.rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
    current_vol = float(volume.iloc[-1])
    volume_ratio = (current_vol / vol_avg_20) if vol_avg_20 and vol_avg_20 > 0 else None

    # ── Derived signals ───────────────────────────────────────────────────────
    trend_direction = _derive_trend(last_close, sma_20, sma_50, macd_diff)
    momentum_signal = _derive_momentum(rsi_14, stoch_k, stoch_d)
    volatility_signal = _derive_volatility(bb_pct, atr_14, last_close)

    log.debug(
        "technical_snapshot",
        close=last_close,
        rsi=rsi_14,
        macd_diff=macd_diff,
        trend=trend_direction,
        momentum=momentum_signal,
    )

    return TechnicalSnapshot(
        last_close=last_close,
        sma_20=sma_20,
        sma_50=sma_50,
        ema_20=ema_20,
        rsi_14=rsi_14,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        macd=macd,
        macd_signal=macd_signal,
        macd_diff=macd_diff,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_pct=bb_pct,
        atr_14=atr_14,
        obv=obv,
        volume_ratio=volume_ratio,
        trend_direction=trend_direction,
        momentum_signal=momentum_signal,
        volatility_signal=volatility_signal,
    )


def _derive_trend(
    close: float,
    sma_20: float | None,
    sma_50: float | None,
    macd_diff: float | None,
) -> str:
    """Derive overall trend direction from price vs moving averages and MACD."""
    bullish_signals = 0
    bearish_signals = 0
    if sma_20 is not None:
        if close > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
    if sma_50 is not None:
        if close > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
    if macd_diff is not None:
        if macd_diff > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
    if bullish_signals > bearish_signals:
        return "bullish"
    if bearish_signals > bullish_signals:
        return "bearish"
    return "neutral"


def _derive_momentum(
    rsi: float | None,
    stoch_k: float | None,
    stoch_d: float | None,
) -> str:
    """Derive momentum signal from RSI and Stochastics."""
    if rsi is None:
        return "neutral"
    if rsi > 70:
        return "overbought"
    if rsi < 30:
        return "oversold"
    if stoch_k is not None and stoch_d is not None:
        if stoch_k > 80 and stoch_d > 80:
            return "overbought"
        if stoch_k < 20 and stoch_d < 20:
            return "oversold"
    return "neutral"


def _derive_volatility(
    bb_pct: float | None,
    atr_14: float | None,
    close: float,
) -> str:
    """Classify volatility level."""
    if bb_pct is None:
        return "normal"
    # Very narrow Bollinger Bands → squeeze
    if 0.4 < bb_pct < 0.6:
        return "squeeze"
    # ATR as % of price
    if atr_14 is not None and close > 0:
        atr_pct = atr_14 / close
        if atr_pct > 0.03:
            return "high"
    return "normal"
