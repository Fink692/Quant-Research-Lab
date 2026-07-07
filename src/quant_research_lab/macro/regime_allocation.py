"""Regime-aware allocation utilities."""

from __future__ import annotations

import pandas as pd

from quant_research_lab.portfolio.covariance import ledoit_wolf_covariance
from quant_research_lab.portfolio.optimization import mean_variance_weights, min_variance_weights


def regime_conditioned_returns(
    returns: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Split returns into samples keyed by macro regime label."""
    aligned = returns.join(regimes.rename("regime"), how="inner").dropna(subset=["regime"])
    return {
        str(regime): group.drop(columns=["regime"])
        for regime, group in aligned.groupby("regime", observed=True)
    }


def regime_allocation_table(
    returns: pd.DataFrame,
    regimes: pd.Series,
    optimizer: str = "min_variance",
) -> pd.DataFrame:
    """Estimate one allocation vector per regime."""
    allocations: dict[str, pd.Series] = {}
    for regime, sample in regime_conditioned_returns(returns, regimes).items():
        if len(sample) < 20:
            continue
        covariance = ledoit_wolf_covariance(sample)
        if optimizer == "mean_variance":
            expected_returns = sample.mean() * 252
            allocations[regime] = mean_variance_weights(
                expected_returns, covariance, long_only=True
            )
        else:
            allocations[regime] = min_variance_weights(covariance, long_only=True)
    return pd.DataFrame(allocations).T.fillna(0.0)


def current_regime_weights(
    allocation_table: pd.DataFrame,
    current_regime: str,
    fallback_equal_weight: bool = True,
) -> pd.Series:
    """Return weights for the current regime with optional equal-weight fallback."""
    if current_regime in allocation_table.index:
        return allocation_table.loc[current_regime].rename("current_regime_weights")
    if not fallback_equal_weight:
        raise KeyError(f"No allocation available for regime {current_regime}.")
    return pd.Series(
        1.0 / len(allocation_table.columns),
        index=allocation_table.columns,
        name="current_regime_weights",
    )
