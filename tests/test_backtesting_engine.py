from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.backtesting.costs import TransactionCostModel
from quant_research_lab.backtesting.engine import BacktestEngine
from quant_research_lab.backtesting.portfolio import Portfolio


def test_transaction_cost_calculation() -> None:
    model = TransactionCostModel(commission_bps=1.0, slippage_bps=2.0, minimum_fee=0.0)
    assert np.isclose(model.calculate(1_000_000), 300.0)
    assert model.calculate(0) == 0


def test_portfolio_value_updates_after_rebalance() -> None:
    portfolio = Portfolio(cash=1000.0)
    prices = pd.Series({"A": 100.0, "B": 50.0})
    turnover, cost = portfolio.rebalance(
        prices,
        pd.Series({"A": 0.5, "B": 0.5}),
        TransactionCostModel(0.0, 0.0),
    )
    assert np.isclose(turnover, 1000.0)
    assert cost == 0
    assert np.isclose(portfolio.total_value(prices), 1000.0)
    assert np.isclose(portfolio.weights(prices).sum(), 1.0)


def test_backtest_engine_runs_buy_and_hold() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=index)
    weights = pd.DataFrame(index=index, columns=["A"], dtype=float)
    weights.iloc[0, 0] = 1.0
    engine = BacktestEngine(
        prices,
        initial_cash=1000.0,
        cost_model=TransactionCostModel(0.0, 0.0),
        execution_lag=0,
    )
    result = engine.run("buy_hold", weights)
    assert result.equity.iloc[-1] > result.equity.iloc[0]
    assert result.transaction_costs.sum() == 0
    assert result.metrics["cagr"] > 0
