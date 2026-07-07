"""Cointegration-based pairs trading research tools."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint

from quant_research_lab.risk.drawdowns import drawdown_series, equity_curve
from quant_research_lab.risk.portfolio_metrics import performance_summary
from quant_research_lab.strategies.signals import rolling_zscore, threshold_positions


@dataclass(frozen=True)
class PairModel:
    """Estimated statistical relationship for a pair."""

    y: str
    x: str
    hedge_ratio: float
    intercept: float
    p_value: float


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Estimate OLS hedge ratio for ``y = intercept + beta * x + error``."""
    aligned = pd.concat([y, x], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("Need at least three observations to estimate hedge ratio.")
    slope, intercept, _, _, _ = stats.linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
    return float(slope), float(intercept)


def find_cointegrated_pairs(prices: pd.DataFrame, min_observations: int = 252) -> pd.DataFrame:
    """Test all possible pairs and rank them by Engle-Granger p-value."""
    results: list[dict[str, float | str]] = []
    clean_prices = prices.dropna(axis=1, thresh=min_observations).ffill().dropna(how="all")
    for y_name, x_name in combinations(clean_prices.columns, 2):
        pair = clean_prices[[y_name, x_name]].dropna()
        if len(pair) < min_observations:
            continue
        try:
            score, p_value, _ = coint(pair[y_name], pair[x_name])
            hedge_ratio, intercept = estimate_hedge_ratio(pair[y_name], pair[x_name])
            results.append(
                {
                    "y": y_name,
                    "x": x_name,
                    "p_value": float(p_value),
                    "test_statistic": float(score),
                    "hedge_ratio": hedge_ratio,
                    "intercept": intercept,
                    "correlation": float(pair[y_name].pct_change().corr(pair[x_name].pct_change())),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)


def build_spread(y: pd.Series, x: pd.Series, hedge_ratio: float, intercept: float = 0.0) -> pd.Series:
    """Construct the stationary spread implied by the hedge ratio."""
    spread = y - (intercept + hedge_ratio * x)
    spread.name = "spread"
    return spread


def generate_pair_weights(
    prices: pd.DataFrame,
    y: str,
    x: str,
    hedge_ratio: float,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.25,
) -> pd.DataFrame:
    """Generate dollar-normalized pair target weights from rolling spread z-scores."""
    spread = build_spread(prices[y], prices[x], hedge_ratio)
    zscores = rolling_zscore(spread, lookback)
    signal = threshold_positions(zscores, entry_z=entry_z, exit_z=exit_z)
    gross = 1.0 + abs(hedge_ratio)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[y] = signal * (1.0 / gross)
    weights[x] = signal * (-hedge_ratio / gross)
    return weights


def backtest_pair_strategy(
    prices: pd.DataFrame,
    y: str,
    x: str,
    hedge_ratio: float,
    intercept: float = 0.0,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.25,
    transaction_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Backtest a close-to-close pair strategy with one-period signal lag.

    Signal decisions use the spread observed at time ``t`` and are applied to
    returns at ``t+1`` through ``signal.shift(1)`` to avoid lookahead bias.
    """
    pair_prices = prices[[y, x]].dropna()
    spread = build_spread(pair_prices[y], pair_prices[x], hedge_ratio, intercept)
    zscores = rolling_zscore(spread, lookback)
    signal = threshold_positions(zscores, entry_z=entry_z, exit_z=exit_z)
    returns = pair_prices.pct_change().fillna(0.0)
    gross = 1.0 + abs(hedge_ratio)
    pair_return = (returns[y] - hedge_ratio * returns[x]) / gross
    strategy_returns = signal.shift(1).fillna(0.0) * pair_return
    turnover = signal.diff().abs().fillna(0.0)
    costs = turnover * transaction_cost_bps / 10000.0
    net_returns = (strategy_returns - costs).rename("strategy_returns")
    equity = equity_curve(net_returns, initial_value=1.0)
    return pd.DataFrame(
        {
            "y_price": pair_prices[y],
            "x_price": pair_prices[x],
            "spread": spread,
            "zscore": zscores,
            "signal": signal,
            "gross_pair_return": strategy_returns,
            "transaction_cost": costs,
            "strategy_returns": net_returns,
            "equity": equity,
            "drawdown": drawdown_series(equity),
        }
    )


def pair_performance_report(strategy_frame: pd.DataFrame) -> pd.Series:
    """Return performance metrics for a pairs strategy output frame."""
    return performance_summary(strategy_frame["strategy_returns"].dropna())
