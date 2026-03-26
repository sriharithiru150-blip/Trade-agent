"""Unit tests for the portfolio manager."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_agent.execution.order_executor import Order
from trade_agent.execution.portfolio_manager import PortfolioManager


def _make_order(symbol: str) -> Order:
    return Order(
        order_id="test-order-001",
        symbol=symbol,
        transaction_type="BUY",
        quantity=10,
        order_type="MARKET",
        price=None,
        trigger_price=None,
        status="COMPLETE",
        placed_at=datetime.now(timezone.utc),
        filled_price=1000.0,
        is_paper=True,
    )


class TestPortfolioManager:
    def setup_method(self) -> None:
        self.pm = PortfolioManager(max_open_positions=3, max_trade_amount=10_000.0)

    def test_initially_empty(self) -> None:
        assert self.pm.position_count == 0
        assert self.pm.realised_pnl == 0.0
        assert self.pm.unrealised_pnl == 0.0

    def test_open_position(self) -> None:
        entry_order = _make_order("RELIANCE")
        pos = self.pm.open_position(
            symbol="RELIANCE",
            quantity=10,
            entry_price=2500.0,
            stop_loss=2450.0,
            take_profit=2600.0,
            entry_order=entry_order,
            sl_order=None,
        )
        assert self.pm.has_position("RELIANCE")
        assert pos.quantity == 10
        assert pos.entry_price == pytest.approx(2500.0)

    def test_can_open_respects_max(self) -> None:
        for sym in ("SYM1", "SYM2", "SYM3"):
            order = _make_order(sym)
            self.pm.open_position(sym, 1, 100.0, 95.0, 110.0, order, None)
        assert self.pm.can_open("SYM4") is False

    def test_cannot_open_duplicate(self) -> None:
        order = _make_order("TCS")
        self.pm.open_position("TCS", 5, 3800.0, 3700.0, 4000.0, order, None)
        assert self.pm.can_open("TCS") is False

    def test_close_position_calculates_pnl(self) -> None:
        order = _make_order("INFY")
        self.pm.open_position("INFY", 10, 1500.0, 1450.0, 1600.0, order, None)
        trade = self.pm.close_position("INFY", 1550.0, "take_profit")
        assert trade is not None
        assert trade.pnl == pytest.approx(10 * (1550.0 - 1500.0))
        assert not self.pm.has_position("INFY")
        assert self.pm.realised_pnl == pytest.approx(500.0)

    def test_close_nonexistent_returns_none(self) -> None:
        result = self.pm.close_position("UNKNOWN", 100.0)
        assert result is None

    def test_update_prices(self) -> None:
        order = _make_order("WIPRO")
        self.pm.open_position("WIPRO", 20, 400.0, 390.0, 420.0, order, None)
        self.pm.update_prices({"WIPRO": 410.0})
        pos = self.pm.open_positions["WIPRO"]
        assert pos.current_price == pytest.approx(410.0)
        assert pos.unrealised_pnl == pytest.approx(20 * 10.0)

    def test_close_all_positions(self) -> None:
        for sym, price in [("A", 100.0), ("B", 200.0)]:
            self.pm.open_position(sym, 1, price, price * 0.98, price * 1.04, _make_order(sym), None)
        trades = self.pm.close_all_positions({"A": 105.0, "B": 195.0}, reason="eod")
        assert len(trades) == 2
        assert self.pm.position_count == 0

    def test_summary_structure(self) -> None:
        summary = self.pm.summary()
        assert "open_positions" in summary
        assert "realised_pnl" in summary
        assert "total_pnl" in summary
        assert "positions" in summary
