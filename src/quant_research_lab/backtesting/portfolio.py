"""Portfolio state object used by the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_research_lab.backtesting.costs import TransactionCostModel
from quant_research_lab.backtesting.execution import calculate_turnover, target_position_values


@dataclass
class Portfolio:
    """Cash and holdings state for long-only or long-short backtests."""

    cash: float
    holdings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def position_values(self, prices: pd.Series) -> pd.Series:
        """Return current dollar value for each instrument."""
        holdings = self.holdings.reindex(prices.index).fillna(0.0)
        return holdings * prices

    def total_value(self, prices: pd.Series) -> float:
        """Return mark-to-market portfolio equity."""
        return float(self.cash + self.position_values(prices).sum())

    def weights(self, prices: pd.Series) -> pd.Series:
        """Return current portfolio weights by instrument."""
        total = self.total_value(prices)
        if total == 0:
            return pd.Series(0.0, index=prices.index)
        return self.position_values(prices) / total

    def rebalance(
        self,
        prices: pd.Series,
        target_weights: pd.Series,
        cost_model: TransactionCostModel,
    ) -> tuple[float, float]:
        """Rebalance to target weights and return turnover and transaction cost."""
        clean_prices = prices.dropna()
        current_values = self.position_values(clean_prices)
        equity_before_cost = self.cash + current_values.sum()
        target_weights = target_weights.reindex(clean_prices.index).fillna(0.0)
        target_values = target_position_values(equity_before_cost, target_weights)
        turnover = calculate_turnover(current_values, target_values)
        cost = cost_model.calculate(turnover)
        self.holdings = target_values / clean_prices
        self.cash = equity_before_cost - target_values.sum() - cost
        return turnover, cost
