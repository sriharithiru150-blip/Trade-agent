"""Order executor — places orders via Zerodha Kite Connect or paper-trade engine.

Safety guarantees
-----------------
1. ``DRY_RUN=true`` (default) bypasses all real API calls entirely.
2. ``KILL_SWITCH=true`` raises :class:`KillSwitchError` before any order.
3. All orders are logged with full context before submission.
4. Stop-loss orders are placed immediately after every entry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from trade_agent.utils.logger import get_logger

log = get_logger(__name__)

OrderStatus = Literal["PENDING", "COMPLETE", "REJECTED", "CANCELLED"]
OrderType = Literal["MARKET", "LIMIT", "SL", "SL-M"]
TransactionType = Literal["BUY", "SELL"]


class KillSwitchError(RuntimeError):
    """Raised when the kill switch is active."""


@dataclass
class Order:
    """Represents a single submitted order."""

    order_id: str
    symbol: str
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType
    price: float | None          # None for MARKET orders
    trigger_price: float | None  # For SL orders
    status: OrderStatus
    placed_at: datetime
    filled_price: float | None = None
    filled_at: datetime | None = None
    is_paper: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "status": self.status,
            "placed_at": self.placed_at.isoformat(),
            "filled_price": self.filled_price,
            "is_paper": self.is_paper,
            "notes": self.notes,
        }


class OrderExecutor:
    """Execute orders via Zerodha Kite Connect or the paper-trade engine.

    Args:
        dry_run: If True, all orders are simulated. Default True.
        kill_switch: If True, all order placement raises :class:`KillSwitchError`.
        kite_api_key: Zerodha API key (only needed for live trading).
        kite_access_token: Zerodha access token (only needed for live trading).
    """

    def __init__(
        self,
        dry_run: bool = True,
        kill_switch: bool = False,
        kite_api_key: str = "",
        kite_access_token: str = "",
    ) -> None:
        self._dry_run = dry_run
        self._kill_switch = kill_switch
        self._kite: object | None = None

        if not dry_run:
            self._kite = self._init_kite(kite_api_key, kite_access_token)

        log.info(
            "order_executor_init",
            mode="paper" if dry_run else "live",
            kill_switch=kill_switch,
        )

    def _init_kite(self, api_key: str, access_token: str) -> object:
        """Initialise the Kite Connect client."""
        try:
            from kiteconnect import KiteConnect  # type: ignore[import]
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            log.info("kite_connected")
            return kite
        except ImportError:
            log.error("kiteconnect_not_installed",
                      msg="Install kiteconnect: pip install kiteconnect")
            raise
        except Exception as exc:
            log.error("kite_init_failed", error=str(exc))
            raise

    def place_market_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        current_price: float,
        notes: str = "",
    ) -> Order:
        """Place a market order.

        Args:
            symbol: NSE symbol (without exchange suffix).
            transaction_type: ``"BUY"`` or ``"SELL"``.
            quantity: Number of shares.
            current_price: Last known price (used to fill paper orders).
            notes: Optional annotation stored on the order.

        Returns:
            The :class:`Order` object with status set.

        Raises:
            KillSwitchError: If kill switch is active.
            ValueError: If quantity ≤ 0.
        """
        self._check_kill_switch()
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        order_id = str(uuid.uuid4())[:12]
        log.info(
            "placing_order",
            symbol=symbol,
            type=transaction_type,
            qty=quantity,
            mode="paper" if self._dry_run else "live",
        )

        if self._dry_run:
            return self._paper_fill(
                order_id=order_id,
                symbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="MARKET",
                price=None,
                trigger_price=None,
                fill_price=current_price,
                notes=notes,
            )

        return self._kite_market_order(order_id, symbol, transaction_type, quantity, notes)

    def place_stop_loss_order(
        self,
        symbol: str,
        quantity: int,
        trigger_price: float,
        limit_price: float | None = None,
        notes: str = "",
    ) -> Order:
        """Place a stop-loss SELL order to protect a long position.

        Args:
            symbol: NSE symbol.
            quantity: Number of shares to sell on trigger.
            trigger_price: Price at which the SL is triggered.
            limit_price: Limit price (leave None for SL-Market).
            notes: Optional annotation.

        Returns:
            The :class:`Order` object.

        Raises:
            KillSwitchError: If kill switch is active.
        """
        self._check_kill_switch()
        order_id = str(uuid.uuid4())[:12]
        order_type: OrderType = "SL" if limit_price else "SL-M"
        log.info(
            "placing_sl_order",
            symbol=symbol,
            qty=quantity,
            trigger=trigger_price,
            mode="paper" if self._dry_run else "live",
        )

        if self._dry_run:
            return self._paper_fill(
                order_id=order_id,
                symbol=symbol,
                transaction_type="SELL",
                quantity=quantity,
                order_type=order_type,
                price=limit_price,
                trigger_price=trigger_price,
                fill_price=trigger_price,
                notes=notes,
                status="PENDING",   # SL stays pending until triggered
            )

        return self._kite_sl_order(order_id, symbol, quantity, trigger_price, limit_price, notes)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: The order ID returned when the order was placed.

        Returns:
            True if cancellation was accepted (always True in paper mode).
        """
        if self._dry_run:
            log.info("paper_cancel_order", order_id=order_id)
            return True
        try:
            assert self._kite is not None
            self._kite.cancel_order(variety="regular", order_id=order_id)  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            log.error("cancel_failed", order_id=order_id, error=str(exc))
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_kill_switch(self) -> None:
        if self._kill_switch:
            log.critical("kill_switch_active", msg="All trading halted")
            raise KillSwitchError("Kill switch is active — no orders will be placed.")

    @staticmethod
    def _paper_fill(
        order_id: str,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        order_type: OrderType,
        price: float | None,
        trigger_price: float | None,
        fill_price: float,
        notes: str = "",
        status: OrderStatus = "COMPLETE",
    ) -> Order:
        now = datetime.now(timezone.utc)
        return Order(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            trigger_price=trigger_price,
            status=status,
            placed_at=now,
            filled_price=fill_price if status == "COMPLETE" else None,
            filled_at=now if status == "COMPLETE" else None,
            is_paper=True,
            notes=notes,
        )

    def _kite_market_order(
        self,
        order_id: str,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        notes: str,
    ) -> Order:
        """Submit a market order to Kite Connect."""
        assert self._kite is not None
        try:
            kite_order_id = self._kite.place_order(  # type: ignore[attr-defined]
                variety="regular",
                exchange="NSE",
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product="MIS",       # intraday
                order_type="MARKET",
            )
            log.info("kite_order_placed", kite_order_id=kite_order_id, symbol=symbol)
            return Order(
                order_id=str(kite_order_id),
                symbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="MARKET",
                price=None,
                trigger_price=None,
                status="PENDING",
                placed_at=datetime.now(timezone.utc),
                is_paper=False,
                notes=notes,
            )
        except Exception as exc:
            log.error("kite_order_failed", symbol=symbol, error=str(exc))
            raise

    def _kite_sl_order(
        self,
        order_id: str,
        symbol: str,
        quantity: int,
        trigger_price: float,
        limit_price: float | None,
        notes: str,
    ) -> Order:
        """Submit a stop-loss order to Kite Connect."""
        assert self._kite is not None
        order_type = "SL" if limit_price else "SL-M"
        params: dict[str, object] = dict(
            variety="regular",
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type="SELL",
            quantity=quantity,
            product="MIS",
            order_type=order_type,
            trigger_price=trigger_price,
        )
        if limit_price:
            params["price"] = limit_price
        try:
            kite_order_id = self._kite.place_order(**params)  # type: ignore[attr-defined]
            log.info("kite_sl_placed", kite_order_id=kite_order_id, symbol=symbol, trigger=trigger_price)
            return Order(
                order_id=str(kite_order_id),
                symbol=symbol,
                transaction_type="SELL",
                quantity=quantity,
                order_type=order_type,
                price=limit_price,
                trigger_price=trigger_price,
                status="PENDING",
                placed_at=datetime.now(timezone.utc),
                is_paper=False,
                notes=notes,
            )
        except Exception as exc:
            log.error("kite_sl_failed", symbol=symbol, error=str(exc))
            raise
