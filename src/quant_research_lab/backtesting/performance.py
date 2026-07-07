"""Backtest performance analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.risk.drawdowns import max_drawdown
from quant_research_lab.risk.portfolio_metrics import (
    alpha_to_benchmark,
    annualized_return,
    annualized_volatility,
    beta_to_benchmark,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)


def calculate_performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> pd.Series:
    """Calculate institutional backtest performance metrics."""
    clean = returns.dropna()
    nonzero = clean[clean != 0]
    metrics = {
        "cagr": annualized_return(clean),
        "annualized_volatility": annualized_volatility(clean),
        "sharpe_ratio": sharpe_ratio(clean, risk_free_rate),
        "sortino_ratio": sortino_ratio(clean, risk_free_rate),
        "max_drawdown": max_drawdown(clean, input_is_returns=True),
        "calmar_ratio": calmar_ratio(clean),
        "win_rate": float((nonzero > 0).mean()) if len(nonzero) else np.nan,
        "average_turnover": float(turnover.mean()) if turnover is not None else np.nan,
        "total_turnover": float(turnover.sum()) if turnover is not None else np.nan,
        "average_exposure": float(exposure.mean()) if exposure is not None else np.nan,
        "total_transaction_costs": float(costs.sum()) if costs is not None else np.nan,
    }
    if benchmark_returns is not None:
        aligned = pd.concat([clean, benchmark_returns], axis=1).dropna()
        if not aligned.empty:
            metrics["beta"] = beta_to_benchmark(aligned.iloc[:, 0], aligned.iloc[:, 1])
            metrics["alpha"] = alpha_to_benchmark(aligned.iloc[:, 0], aligned.iloc[:, 1], risk_free_rate)
    return pd.Series(metrics, name=returns.name or "strategy")


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Return a year-by-month table of compounded monthly returns."""
    monthly = (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.strftime("%b")
    pivot = table.pivot(index="year", columns="month", values="return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot.reindex(columns=month_order)


def strategy_comparison(metrics: dict[str, pd.Series]) -> pd.DataFrame:
    """Combine strategy metric Series into a comparison table."""
    return pd.DataFrame(metrics).T.sort_index()
