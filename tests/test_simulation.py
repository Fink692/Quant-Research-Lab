from __future__ import annotations

import pandas as pd

from quant_research_lab.simulation.monte_carlo import monte_carlo_summary, simulate_portfolio_paths


def test_monte_carlo_paths_shape_and_summary() -> None:
    expected_returns = pd.Series({"A": 0.08, "B": 0.04})
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=["A", "B"], columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "B": 0.4})
    paths = simulate_portfolio_paths(expected_returns, covariance, weights, n_days=20, n_paths=50)
    assert paths.shape == (20, 50)
    summary = monte_carlo_summary(paths)
    assert "mean_terminal_value" in summary
    assert summary["p95_terminal_value"] >= summary["p05_terminal_value"]
