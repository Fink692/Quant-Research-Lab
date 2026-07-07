from __future__ import annotations

import pandas as pd

from quant_research_lab.backtesting.calendar import rebalance_dates
from quant_research_lab.backtesting.constraints import (
    PositionConstraints,
    apply_position_constraints,
)
from quant_research_lab.backtesting.costs import FinancingModel, TransactionCostModel
from quant_research_lab.backtesting.execution import FillSimulator
from quant_research_lab.backtesting.orders import Order, OrderSide
from quant_research_lab.backtesting.research import parameter_grid, walk_forward_splits


def test_rebalance_dates_returns_period_ends() -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    dates = rebalance_dates(index, "ME")
    assert len(dates) >= 3
    assert dates.isin(index).all()


def test_position_constraints_clip_gross_and_single_name() -> None:
    weights = pd.Series({"A": 0.8, "B": -0.7, "C": 0.4})
    constrained = apply_position_constraints(
        weights,
        PositionConstraints(max_gross_exposure=1.0, max_single_position=0.5),
    )
    assert constrained.abs().sum() <= 1.0 + 1e-12
    assert constrained.abs().max() <= 0.5 + 1e-12


def test_fill_simulator_and_financing_model() -> None:
    simulator = FillSimulator(TransactionCostModel(commission_bps=1.0, slippage_bps=5.0))
    fill = simulator.fill_order(Order("A", 100, OrderSide.BUY), 10.0)
    assert fill.fill_price > 10.0
    assert fill.commission > 0
    financing = FinancingModel(cash_rate=0.02, debit_rate=0.05, short_borrow_rate=0.03)
    assert financing.cash_interest(1000.0) > 0
    assert financing.cash_interest(-1000.0) < 0
    assert financing.borrow_cost(1000.0) > 0


def test_walk_forward_and_parameter_grid() -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    splits = walk_forward_splits(index, train_size=40, test_size=20)
    assert len(splits) == 3
    grid = parameter_grid({"lookback": [20, 60], "entry_z": [1.5, 2.0]})
    assert len(grid) == 4
    assert {"lookback": 20, "entry_z": 1.5} in grid
