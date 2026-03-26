"""Unit tests for the order executor."""

from __future__ import annotations

import pytest

from trade_agent.execution.order_executor import (
    KillSwitchError,
    Order,
    OrderExecutor,
)


class TestPaperTradingMode:
    """All tests run in DRY_RUN mode — no real orders placed."""

    def setup_method(self) -> None:
        self.executor = OrderExecutor(dry_run=True, kill_switch=False)

    def test_market_buy_returns_order(self) -> None:
        order = self.executor.place_market_order(
            symbol="RELIANCE", transaction_type="BUY", quantity=10, current_price=2500.0
        )
        assert isinstance(order, Order)
        assert order.status == "COMPLETE"
        assert order.symbol == "RELIANCE"
        assert order.quantity == 10
        assert order.is_paper is True

    def test_market_sell_returns_order(self) -> None:
        order = self.executor.place_market_order(
            symbol="TCS", transaction_type="SELL", quantity=5, current_price=3800.0
        )
        assert order.transaction_type == "SELL"
        assert order.filled_price == pytest.approx(3800.0)

    def test_stop_loss_order_is_pending(self) -> None:
        order = self.executor.place_stop_loss_order(
            symbol="INFY", quantity=20, trigger_price=1450.0
        )
        assert order.status == "PENDING"
        assert order.trigger_price == pytest.approx(1450.0)
        assert order.transaction_type == "SELL"

    def test_cancel_returns_true(self) -> None:
        order = self.executor.place_market_order("HDFC", "BUY", 5, 1700.0)
        result = self.executor.cancel_order(order.order_id)
        assert result is True

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="Quantity must be positive"):
            self.executor.place_market_order("WIPRO", "BUY", 0, 400.0)

    def test_negative_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="Quantity must be positive"):
            self.executor.place_market_order("WIPRO", "BUY", -5, 400.0)

    def test_order_id_is_unique(self) -> None:
        o1 = self.executor.place_market_order("TCS", "BUY", 1, 3800.0)
        o2 = self.executor.place_market_order("TCS", "BUY", 1, 3800.0)
        assert o1.order_id != o2.order_id


class TestKillSwitch:
    def test_kill_switch_blocks_buy(self) -> None:
        executor = OrderExecutor(dry_run=True, kill_switch=True)
        with pytest.raises(KillSwitchError):
            executor.place_market_order("RELIANCE", "BUY", 10, 2500.0)

    def test_kill_switch_blocks_sl(self) -> None:
        executor = OrderExecutor(dry_run=True, kill_switch=True)
        with pytest.raises(KillSwitchError):
            executor.place_stop_loss_order("RELIANCE", 10, 2450.0)
