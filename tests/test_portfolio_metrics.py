from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.data.market_data import calculate_returns
from quant_research_lab.risk.drawdowns import max_drawdown
from quant_research_lab.risk.portfolio_metrics import (
    annualized_return,
    portfolio_returns,
    sharpe_ratio,
)


def test_return_calculation_and_portfolio_returns() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 95.0, 95.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    returns = calculate_returns(prices)
    assert np.isclose(returns.loc["2024-01-02", "A"], 0.10)
    portfolio = portfolio_returns(returns, {"A": 0.5, "B": 0.5})
    assert np.isclose(portfolio.iloc[0], 0.025)


def test_sharpe_ratio_and_annualized_return_are_finite() -> None:
    returns = pd.Series([0.01, -0.002, 0.004, 0.003, -0.001] * 60)
    assert np.isfinite(sharpe_ratio(returns, risk_free_rate=0.0))
    assert annualized_return(returns) > 0


def test_max_drawdown() -> None:
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])
    assert np.isclose(max_drawdown(returns), -0.244, atol=0.002)
