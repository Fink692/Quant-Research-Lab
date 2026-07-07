"""Exponentially weighted volatility and covariance models."""

from __future__ import annotations

import pandas as pd


def ewma_volatility(
    returns: pd.Series,
    span: int = 60,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate annualized EWMA volatility."""
    variance = returns.pow(2).ewm(span=span, adjust=False).mean()
    return (variance.pow(0.5) * periods_per_year**0.5).rename("ewma_volatility")


def ewma_correlation(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    span: int = 60,
) -> pd.Series:
    """Calculate EWMA correlation between an asset and benchmark."""
    covariance = asset_returns.ewm(span=span, adjust=False).cov(benchmark_returns)
    asset_var = asset_returns.pow(2).ewm(span=span, adjust=False).mean()
    bench_var = benchmark_returns.pow(2).ewm(span=span, adjust=False).mean()
    return (covariance / (asset_var.pow(0.5) * bench_var.pow(0.5))).rename("ewma_correlation")
