"""Covariance estimators for portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Return the annualized sample covariance matrix."""
    clean = returns.dropna(how="all").fillna(0.0)
    return clean.cov() * periods_per_year


def ledoit_wolf_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Return the annualized Ledoit-Wolf shrinkage covariance matrix."""
    clean = returns.dropna(how="all").fillna(0.0)
    estimator = LedoitWolf().fit(clean.to_numpy())
    covariance = estimator.covariance_ * periods_per_year
    return pd.DataFrame(covariance, index=clean.columns, columns=clean.columns)


def exponentially_weighted_covariance(
    returns: pd.DataFrame,
    span: int = 60,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Return an annualized exponentially weighted covariance estimate."""
    clean = returns.dropna(how="all").fillna(0.0)
    demeaned = clean - clean.ewm(span=span, adjust=False).mean()
    weights = np.exp(np.linspace(-1.0, 0.0, len(clean)))
    weights = weights / weights.sum()
    covariance = np.cov(demeaned.to_numpy().T, aweights=weights) * periods_per_year
    return pd.DataFrame(covariance, index=clean.columns, columns=clean.columns)


def correlation_from_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    volatility = np.sqrt(np.diag(covariance))
    denominator = np.outer(volatility, volatility)
    correlation = covariance.to_numpy() / denominator
    return pd.DataFrame(correlation, index=covariance.index, columns=covariance.columns)
