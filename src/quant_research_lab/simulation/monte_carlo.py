"""Monte Carlo portfolio simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_portfolio_paths(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    n_days: int = 252,
    n_paths: int = 1000,
    initial_value: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate portfolio value paths using multivariate normal daily returns."""
    assets = covariance.index.intersection(expected_returns.index).intersection(weights.index)
    mu_daily = expected_returns.loc[assets].to_numpy() / 252.0
    cov_daily = covariance.loc[assets, assets].to_numpy() / 252.0
    weight_vector = weights.loc[assets].to_numpy()
    rng = np.random.default_rng(seed)
    simulated_asset_returns = rng.multivariate_normal(mu_daily, cov_daily, size=(n_days, n_paths))
    portfolio_returns = simulated_asset_returns @ weight_vector
    values = initial_value * np.cumprod(1.0 + portfolio_returns, axis=0)
    index = pd.RangeIndex(start=1, stop=n_days + 1, name="day")
    columns = [f"path_{idx + 1}" for idx in range(n_paths)]
    return pd.DataFrame(values, index=index, columns=columns)


def monte_carlo_summary(paths: pd.DataFrame) -> pd.Series:
    """Summarize simulated terminal wealth distribution."""
    terminal = paths.iloc[-1]
    return pd.Series(
        {
            "mean_terminal_value": terminal.mean(),
            "median_terminal_value": terminal.median(),
            "p05_terminal_value": terminal.quantile(0.05),
            "p95_terminal_value": terminal.quantile(0.95),
            "probability_loss": float((terminal < paths.iloc[0].mean()).mean()),
        }
    )
