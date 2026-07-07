"""Stress testing and tail-risk reporting."""

from __future__ import annotations

import pandas as pd

from quant_research_lab.risk.var_models import historical_cvar, historical_var


def stress_portfolio_return(weights: pd.Series, scenario_returns: pd.Series) -> float:
    """Calculate portfolio return under a single deterministic shock scenario."""
    aligned_weights, aligned_shocks = weights.align(scenario_returns, fill_value=0.0)
    return float((aligned_weights * aligned_shocks).sum())


def stress_test_table(weights: pd.Series, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Calculate portfolio shock return for each named scenario."""
    rows = []
    for scenario_name, shocks in scenarios.iterrows():
        rows.append(
            {
                "scenario": scenario_name,
                "portfolio_return": stress_portfolio_return(weights, shocks),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def tail_risk_report(
    returns: pd.Series,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """Return historical VaR and CVaR for multiple confidence levels."""
    rows = []
    for confidence in confidence_levels:
        rows.append(
            {
                "confidence": confidence,
                "historical_var": historical_var(returns, confidence),
                "historical_cvar": historical_cvar(returns, confidence),
            }
        )
    return pd.DataFrame(rows).set_index("confidence")
