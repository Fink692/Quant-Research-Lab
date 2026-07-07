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


@dataclass(frozen=True)
class FinancingModel:
    """Daily cash interest and short-borrow cost model."""

    cash_rate: float = 0.0
    debit_rate: float = 0.0
    short_borrow_rate: float = 0.0
    periods_per_year: int = 252

    def cash_interest(self, cash_balance: float) -> float:
        """Return one-period interest earned or paid on cash."""
        if cash_balance >= 0:
            return float(cash_balance * self.cash_rate / self.periods_per_year)
        return float(cash_balance * self.debit_rate / self.periods_per_year)

    def borrow_cost(self, short_notional: float) -> float:
        """Return one-period stock borrow cost for short exposure."""
        if short_notional <= 0:
            return 0.0
        return float(short_notional * self.short_borrow_rate / self.periods_per_year)
