from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.portfolio.covariance import (
    exponentially_weighted_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)
from quant_research_lab.portfolio.optimization import (
    mean_variance_weights,
    min_variance_weights,
    portfolio_risk_contributions,
    risk_parity_weights,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(300, 4)),
        columns=["A", "B", "C", "D"],
        index=pd.bdate_range("2023-01-01", periods=300),
    )


def test_covariance_estimators_return_square_matrices() -> None:
    returns = _returns()
    for estimator in [sample_covariance, ledoit_wolf_covariance, exponentially_weighted_covariance]:
        covariance = estimator(returns)
        assert covariance.shape == (4, 4)
        assert np.allclose(covariance, covariance.T)


def test_optimizers_return_valid_weights() -> None:
    returns = _returns()
    covariance = ledoit_wolf_covariance(returns)
    expected_returns = returns.mean() * 252
    for weights in [
        min_variance_weights(covariance),
        mean_variance_weights(expected_returns, covariance),
        risk_parity_weights(covariance),
    ]:
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= -1e-8).all()
        assert (weights <= 1.0 + 1e-8).all()


def test_risk_contributions_sum_to_one() -> None:
    returns = _returns()
    covariance = sample_covariance(returns)
    weights = pd.Series(0.25, index=returns.columns)
    contributions = portfolio_risk_contributions(weights, covariance)
    assert np.isclose(contributions.sum(), 1.0)
