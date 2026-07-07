"""Order and fill objects for execution research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class OrderSide(StrEnum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Order:
    """Simple order representation for research execution simulation."""

    ticker: str
    quantity: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class Fill:
    """Executed fill representation."""

    ticker: str
    quantity: float
    side: OrderSide
    fill_price: float
    commission: float
    slippage: float
    timestamp: datetime | None = None

    @property
    def notional(self) -> float:
        """Absolute traded notional."""
        return abs(self.quantity * self.fill_price)

    @property
    def signed_quantity(self) -> float:
        """Quantity signed positive for buy and negative for sell."""
        return self.quantity if self.side == OrderSide.BUY else -self.quantity
