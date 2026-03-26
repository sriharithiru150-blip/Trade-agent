"""Portfolio manager — tracks positions, P&L, and enforces risk limits.

Positions are held in memory. In a production system you would persist
these to a database or Redis so they survive process restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from trade_agent.execution.order_executor import Order
from trade_agent.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Position:
    """An open long position."""

    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_order_id: str
    sl_order_id: str | None
    opened_at: datetime
    unrealised_pnl: float = 0.0
    current_price: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and recalculate unrealised P&L."""
        self.current_price = price
        self.unrealised_pnl = (price - self.entry_price) * self.quantity

    @property
    def entry_value(self) -> float:
        return self.entry_price * self.quantity

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "current_price": round(self.current_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "entry_value": round(self.entry_value, 2),
            "opened_at": self.opened_at.isoformat(),
        }


@dataclass
class ClosedTrade:
    """A completed round-trip trade."""

    symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    opened_at: datetime
    closed_at: datetime
    exit_reason: str

    @property
    def pnl_pct(self) -> float:
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "exit_reason": self.exit_reason,
        }


class PortfolioManager:
    """In-memory portfolio tracker with risk limit enforcement.

    Args:
        max_open_positions: Hard cap on simultaneous open positions.
        max_trade_amount: Max INR to deploy per trade (used for capacity checks).
    """

    def __init__(
        self,
        max_open_positions: int = 5,
        max_trade_amount: float = 10_000.0,
    ) -> None:
        self._max_positions = max_open_positions
        self._max_trade_amount = max_trade_amount
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._realised_pnl: float = 0.0

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> dict[str, Position]:
        """Return a copy of open positions keyed by symbol."""
        return dict(self._positions)

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def realised_pnl(self) -> float:
        return self._realised_pnl

    @property
    def unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self._positions.values())

    @property
    def total_pnl(self) -> float:
        return self._realised_pnl + self.unrealised_pnl

    def has_position(self, symbol: str) -> bool:
        return symbol.upper() in self._positions

    def can_open(self, symbol: str) -> bool:
        """Return True if a new position can be opened for ``symbol``."""
        if self.has_position(symbol):
            log.debug("cannot_open_duplicate", symbol=symbol)
            return False
        if self.position_count >= self._max_positions:
            log.warning("max_positions_reached", count=self.position_count, max=self._max_positions)
            return False
        return True

    # ── Mutations ─────────────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_order: Order,
        sl_order: Order | None,
    ) -> Position:
        """Record a new open position.

        Args:
            symbol: NSE/BSE symbol.
            quantity: Shares purchased.
            entry_price: Execution price.
            stop_loss: Stop-loss price.
            take_profit: Take-profit price.
            entry_order: The filled entry order.
            sl_order: The pending stop-loss order (or None in paper mode).

        Returns:
            The new :class:`Position`.
        """
        key = symbol.upper()
        pos = Position(
            symbol=key,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order_id=entry_order.order_id,
            sl_order_id=sl_order.order_id if sl_order else None,
            opened_at=datetime.now(timezone.utc),
            current_price=entry_price,
        )
        self._positions[key] = pos
        log.info(
            "position_opened",
            symbol=key,
            qty=quantity,
            entry=entry_price,
            sl=stop_loss,
            tp=take_profit,
        )
        return pos

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all open positions.

        Args:
            prices: Dict of symbol → current price.
        """
        for symbol, price in prices.items():
            key = symbol.upper()
            if key in self._positions:
                self._positions[key].update_price(price)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> ClosedTrade | None:
        """Close an open position and record the trade.

        Args:
            symbol: NSE/BSE symbol.
            exit_price: Execution price for the exit.
            reason: Why the position was closed (e.g. 'stop_loss', 'take_profit', 'eod').

        Returns:
            The :class:`ClosedTrade` if a position existed, else None.
        """
        key = symbol.upper()
        pos = self._positions.pop(key, None)
        if pos is None:
            log.warning("close_no_position", symbol=key)
            return None
        pnl = (exit_price - pos.entry_price) * pos.quantity
        self._realised_pnl += pnl
        trade = ClosedTrade(
            symbol=key,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=round(pnl, 2),
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
            exit_reason=reason,
        )
        self._closed_trades.append(trade)
        log.info(
            "position_closed",
            symbol=key,
            pnl=round(pnl, 2),
            pnl_pct=round(trade.pnl_pct, 4),
            reason=reason,
        )
        return trade

    def close_all_positions(
        self, current_prices: dict[str, float], reason: str = "eod"
    ) -> list[ClosedTrade]:
        """Close all open positions (end-of-day square-off).

        Args:
            current_prices: Map of symbol → current price.
            reason: Reason code (default ``"eod"``).

        Returns:
            List of :class:`ClosedTrade` objects.
        """
        closed = []
        for symbol in list(self._positions):
            price = current_prices.get(symbol, self._positions[symbol].entry_price)
            trade = self.close_position(symbol, price, reason)
            if trade:
                closed.append(trade)
        return closed

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, object]:
        """Return a portfolio summary dict."""
        return {
            "open_positions": self.position_count,
            "realised_pnl": round(self._realised_pnl, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "positions": [p.to_dict() for p in self._positions.values()],
            "total_trades": len(self._closed_trades),
            "winning_trades": sum(1 for t in self._closed_trades if t.pnl > 0),
            "losing_trades": sum(1 for t in self._closed_trades if t.pnl < 0),
        }

    def trade_history(self) -> list[dict[str, object]]:
        """Return all closed trades as a list of dicts."""
        return [t.to_dict() for t in self._closed_trades]
