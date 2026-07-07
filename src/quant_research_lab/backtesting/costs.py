"""Transaction cost and slippage models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCostModel:
    """Linear bps cost model with optional minimum ticket fee."""

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    minimum_fee: float = 0.0

    def calculate(self, traded_notional: float) -> float:
        """Calculate total trading cost for absolute traded notional."""
        if traded_notional <= 0:
            return 0.0
        variable_cost = traded_notional * (self.commission_bps + self.slippage_bps) / 10000.0
        return float(max(variable_cost, self.minimum_fee))
