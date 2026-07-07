from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.risk.ewma import ewma_correlation, ewma_volatility
from quant_research_lab.risk.factor_model import estimate_factor_model, factor_portfolio_variance
from quant_research_lab.risk.stress_testing import stress_test_table, tail_risk_report


def test_ewma_outputs_are_aligned() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.005], index=pd.bdate_range("2024-01-01", periods=4))
    vol = ewma_volatility(returns, span=2)
    corr = ewma_correlation(returns, returns, span=2)
    assert vol.index.equals(returns.index)
    assert corr.index.equals(returns.index)
    assert vol.iloc[-1] > 0


def test_factor_model_estimates_beta() -> None:
    index = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(99)
    factor = pd.Series(rng.normal(0, 0.01, len(index)), index=index, name="MKT")
    asset = 0.0001 + 1.5 * factor + rng.normal(0, 0.002, len(index))
    result = estimate_factor_model(asset.to_frame("Asset"), factor.to_frame())
    assert np.isclose(result.betas.loc["Asset", "MKT"], 1.5, atol=0.1)
    variance = factor_portfolio_variance(
        pd.Series({"Asset": 1.0}),
        result.betas,
        factor.to_frame().cov() * 252,
        result.residual_volatility,
    )
    assert variance > 0


def test_stress_and_tail_reports() -> None:
    weights = pd.Series({"SPY": 0.6, "IEF": 0.4})
    scenarios = pd.DataFrame(
        {"SPY": [-0.2, 0.1], "IEF": [0.03, -0.02]},
        index=["equity_selloff", "rates_up"],
    )
    report = stress_test_table(weights, scenarios)
    assert report.loc["equity_selloff", "portfolio_return"] < 0
    tail = tail_risk_report(pd.Series([-0.05, -0.02, 0.01, 0.02, 0.03]), (0.8,))
    assert tail.loc[0.8, "historical_var"] > 0
