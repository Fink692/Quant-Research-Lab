"""Portfolio optimization routines used in institutional research workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _normalise(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total == 0:
        return np.repeat(1.0 / len(weights), len(weights))
    return weights / total


def min_variance_weights(
    covariance: pd.DataFrame,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """Solve a minimum-variance portfolio."""
    n_assets = len(covariance)
    initial = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, max_weight)] * n_assets if long_only else [(-max_weight, max_weight)] * n_assets
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]

    def objective(weights: np.ndarray) -> float:
        return float(weights.T @ covariance.to_numpy() @ weights)

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Minimum-variance optimization failed: {result.message}")
    return pd.Series(_normalise(result.x), index=covariance.index, name="min_variance")


def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 5.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """Solve a constrained mean-variance utility portfolio."""
    assets = covariance.index.intersection(expected_returns.index)
    mu = expected_returns.loc[assets].to_numpy()
    cov = covariance.loc[assets, assets].to_numpy()
    n_assets = len(assets)
    initial = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, max_weight)] * n_assets if long_only else [(-max_weight, max_weight)] * n_assets
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]

    def objective(weights: np.ndarray) -> float:
        portfolio_return = weights @ mu
        portfolio_variance = weights.T @ cov @ weights
        return float(-(portfolio_return - 0.5 * risk_aversion * portfolio_variance))

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Mean-variance optimization failed: {result.message}")
    return pd.Series(_normalise(result.x), index=assets, name="mean_variance")


def risk_parity_weights(
    covariance: pd.DataFrame,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """Solve equal-risk-contribution risk parity weights."""
    n_assets = len(covariance)
    cov = covariance.to_numpy()
    initial = np.repeat(1.0 / n_assets, n_assets)
    bounds = (
        [(1e-6, max_weight)] * n_assets if long_only else [(-max_weight, max_weight)] * n_assets
    )
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = weights.T @ cov @ weights
        marginal_contribution = cov @ weights
        risk_contribution = weights * marginal_contribution / portfolio_variance
        target = np.repeat(1.0 / n_assets, n_assets)
        return float(((risk_contribution - target) ** 2).sum())

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Risk parity optimization failed: {result.message}")
    return pd.Series(_normalise(result.x), index=covariance.index, name="risk_parity")


def portfolio_risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Return percentage contribution to portfolio variance by asset."""
    assets = covariance.index.intersection(weights.index)
    weight_vector = weights.loc[assets].to_numpy()
    cov = covariance.loc[assets, assets].to_numpy()
    variance = weight_vector.T @ cov @ weight_vector
    contribution = weight_vector * (cov @ weight_vector) / variance
    return pd.Series(contribution, index=assets, name="risk_contribution")
