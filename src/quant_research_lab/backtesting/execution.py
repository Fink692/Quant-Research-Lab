"""Execution simulator helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research_lab.backtesting.costs import TransactionCostModel
from quant_research_lab.backtesting.orders import Fill, Order, OrderSide


def target_position_values(equity: float, target_weights: pd.Series) -> pd.Series:
    """Convert target portfolio weights into dollar position values."""
    return target_weights.fillna(0.0).astype(float) * equity


def calculate_turnover(current_values: pd.Series, target_values: pd.Series) -> float:
    """Calculate one-way absolute notional turnover for a rebalance."""
    aligned_current, aligned_target = current_values.align(target_values, fill_value=0.0)
    return float((aligned_target - aligned_current).abs().sum())


@dataclass(frozen=True)
class FillSimulator:
    """Simple close-price execution simulator with bps slippage."""

    cost_model: TransactionCostModel

    def fill_order(self, order: Order, market_price: float) -> Fill:
        """Convert an order into a fill using deterministic bps slippage."""
        slippage_rate = self.cost_model.slippage_bps / 10000.0
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1.0 + slippage_rate)
        else:
            fill_price = market_price * (1.0 - slippage_rate)
        notional = abs(order.quantity * fill_price)
        commission = max(
            notional * self.cost_model.commission_bps / 10000.0, self.cost_model.minimum_fee
        )
        slippage = abs(order.quantity * (fill_price - market_price))
        return Fill(
            ticker=order.ticker,
            quantity=abs(order.quantity),
            side=order.side,
            fill_price=float(fill_price),
            commission=float(commission),
            slippage=float(slippage),
            timestamp=order.timestamp,
        )
