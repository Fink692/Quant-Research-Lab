"""Linear factor risk model utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FactorModelResult:
    """Estimated factor exposures and residual risk."""

    betas: pd.DataFrame
    alphas: pd.Series
    residual_volatility: pd.Series
    r_squared: pd.Series


def estimate_factor_model(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> FactorModelResult:
    """Estimate asset exposures to a set of return factors using OLS."""
    joined = asset_returns.join(factor_returns, how="inner", lsuffix="_asset")
    factors = factor_returns.columns
    betas: dict[str, pd.Series] = {}
    alphas: dict[str, float] = {}
    residual_vols: dict[str, float] = {}
    r_squared: dict[str, float] = {}

    x = joined[factors].to_numpy()
    x_design = np.column_stack([np.ones(len(x)), x])
    for asset in asset_returns.columns:
        y = joined[asset].to_numpy()
        coefficients, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
        fitted = x_design @ coefficients
        residuals = y - fitted
        total_ss = ((y - y.mean()) ** 2).sum()
        residual_ss = (residuals**2).sum()
        betas[asset] = pd.Series(coefficients[1:], index=factors)
        alphas[asset] = coefficients[0] * periods_per_year
        residual_vols[asset] = residuals.std(ddof=1) * np.sqrt(periods_per_year)
        r_squared[asset] = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan

    return FactorModelResult(
        betas=pd.DataFrame(betas).T,
        alphas=pd.Series(alphas, name="annualized_alpha"),
        residual_volatility=pd.Series(residual_vols, name="residual_volatility"),
        r_squared=pd.Series(r_squared, name="r_squared"),
    )


def factor_portfolio_variance(
    weights: pd.Series,
    factor_betas: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    residual_volatility: pd.Series,
) -> float:
    """Estimate portfolio variance from factor and idiosyncratic components."""
    assets = weights.index.intersection(factor_betas.index).intersection(residual_volatility.index)
    w = weights.loc[assets].to_numpy()
    beta = factor_betas.loc[assets].to_numpy()
    factor_cov = factor_covariance.loc[factor_betas.columns, factor_betas.columns].to_numpy()
    specific_var = np.diag(residual_volatility.loc[assets].pow(2).to_numpy())
    total_cov = beta @ factor_cov @ beta.T + specific_var
    return float(w.T @ total_cov @ w)


def single_factor_beta(asset_returns: pd.Series, market_returns: pd.Series) -> tuple[float, float]:
    """Estimate single-factor beta and annualized alpha versus a market proxy."""
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    slope, intercept, _, _, _ = stats.linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
    return float(slope), float(intercept * 252)
