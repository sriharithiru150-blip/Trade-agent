"""Market data client — fetches quotes and OHLCV history for NSE/BSE stocks.

Uses ``yfinance`` under the hood.  NSE symbols get the ``.NS`` suffix;
BSE symbols use ``.BO``.  By default all symbols are assumed to be NSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd
import yfinance as yf

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

Exchange = Literal["NSE", "BSE"]

_SUFFIX: dict[Exchange, str] = {"NSE": ".NS", "BSE": ".BO"}

# Nifty 50 / Sensex index tickers for market breadth
INDEX_TICKERS = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY_BANK": "^NSEBANK",
    "INDIA_VIX": "^INDIAVIX",
}


@dataclass
class Quote:
    """Current snapshot for a single stock."""

    symbol: str
    exchange: Exchange
    price: float
    open: float
    high: float
    low: float
    volume: int
    prev_close: float
    change_pct: float
    market_cap: float | None
    timestamp: datetime


def _ticker_symbol(symbol: str, exchange: Exchange = "NSE") -> str:
    """Return the yfinance ticker string, e.g. ``RELIANCE.NS``."""
    symbol = symbol.upper().strip()
    suffix = _SUFFIX[exchange]
    if not symbol.endswith(suffix):
        symbol = symbol + suffix
    return symbol


class MarketDataClient:
    """Thin wrapper around yfinance for NSE/BSE data.

    All public methods are synchronous; call them from a thread pool if
    you need async behaviour.
    """

    def get_quote(self, symbol: str, exchange: Exchange = "NSE") -> Quote:
        """Fetch the latest quote for a single symbol.

        Args:
            symbol: NSE/BSE stock symbol (e.g. ``RELIANCE``).
            exchange: ``"NSE"`` or ``"BSE"``.

        Returns:
            A :class:`Quote` dataclass with current market data.

        Raises:
            ValueError: If the ticker cannot be found or returns no data.
        """
        ticker_str = _ticker_symbol(symbol, exchange)
        ticker = yf.Ticker(ticker_str)
        info = ticker.fast_info

        try:
            price = float(info.last_price)
            prev_close = float(info.previous_close)
            open_ = float(info.open)
            day_high = float(info.day_high)
            day_low = float(info.day_low)
            volume = int(info.three_month_average_volume or 0)
            market_cap = float(info.market_cap) if info.market_cap else None
        except Exception as exc:
            raise ValueError(f"Could not fetch quote for {ticker_str}: {exc}") from exc

        change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0

        return Quote(
            symbol=symbol.upper(),
            exchange=exchange,
            price=price,
            open=open_,
            high=day_high,
            low=day_low,
            volume=volume,
            prev_close=prev_close,
            change_pct=round(change_pct, 4),
            market_cap=market_cap,
            timestamp=datetime.utcnow(),
        )

    def get_history(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
        exchange: Exchange = "NSE",
    ) -> pd.DataFrame:
        """Fetch OHLCV history for a symbol.

        Args:
            symbol: NSE/BSE stock symbol.
            period: yfinance period string (e.g. ``"3mo"``, ``"1y"``).
            interval: Bar interval (e.g. ``"1d"``, ``"1h"``, ``"5m"``).
            exchange: ``"NSE"`` or ``"BSE"``.

        Returns:
            DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume`` and a DatetimeIndex (UTC).

        Raises:
            ValueError: If no data is returned.
        """
        ticker_str = _ticker_symbol(symbol, exchange)
        df = yf.download(ticker_str, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No history data for {ticker_str} (period={period}, interval={interval})")
        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        log.debug("fetched_history", symbol=ticker_str, rows=len(df), interval=interval)
        return df

    def get_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        exchange: Exchange = "NSE",
    ) -> pd.DataFrame:
        """Convenience wrapper — fetch today's intraday bars.

        Args:
            symbol: NSE/BSE stock symbol.
            interval: Bar size (``"1m"``, ``"5m"``, ``"15m"``).
            exchange: Exchange.

        Returns:
            OHLCV DataFrame for the current trading day.
        """
        return self.get_history(symbol, period="1d", interval=interval, exchange=exchange)

    def get_index_data(self, index: str = "NIFTY50") -> pd.DataFrame:
        """Fetch daily OHLCV for a market index.

        Args:
            index: One of ``NIFTY50``, ``SENSEX``, ``NIFTY_BANK``, ``INDIA_VIX``.

        Returns:
            3-month daily OHLCV DataFrame.
        """
        ticker_str = INDEX_TICKERS.get(index.upper())
        if not ticker_str:
            raise ValueError(f"Unknown index '{index}'. Valid options: {list(INDEX_TICKERS)}")
        df = yf.download(ticker_str, period="3mo", progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for index {index}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def get_multiple_quotes(
        self, symbols: list[str], exchange: Exchange = "NSE"
    ) -> dict[str, Quote]:
        """Fetch quotes for multiple symbols.

        Args:
            symbols: List of NSE/BSE symbols.
            exchange: Exchange for all symbols.

        Returns:
            Dict mapping symbol → :class:`Quote`.  Symbols that fail are
            omitted and a warning is logged.
        """
        results: dict[str, Quote] = {}
        for sym in symbols:
            try:
                results[sym.upper()] = self.get_quote(sym, exchange)
            except ValueError as exc:
                log.warning("quote_fetch_failed", symbol=sym, error=str(exc))
        return results
