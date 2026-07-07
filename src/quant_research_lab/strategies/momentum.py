"""Cross-sectional momentum strategy helpers."""

from __future__ import annotations

import pandas as pd


def momentum_scores(prices: pd.DataFrame, lookback_days: int = 126) -> pd.DataFrame:
    """Calculate trailing total-return momentum scores."""
    return prices / prices.shift(lookback_days) - 1.0


def monthly_momentum_weights(
    prices: pd.DataFrame,
    lookback_days: int = 126,
    top_n: int = 3,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Create monthly rebalance weights for a top-N momentum rotation strategy."""
    scores = momentum_scores(prices, lookback_days=lookback_days)
    rebal_dates = prices.resample("ME").last().index
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for date in rebal_dates:
        if date not in scores.index:
            continue
        row = scores.loc[date].dropna().sort_values(ascending=False)
        winners = row.head(top_n).index
        if len(winners) == 0:
            continue
        weights.loc[date, winners] = max_gross_exposure / len(winners)
        weights.loc[date, weights.columns.difference(winners)] = 0.0
    return weights


def buy_and_hold_weights(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create one initial long-only target weight for a buy-and-hold strategy."""
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    if ticker not in prices.columns:
        raise ValueError(f"{ticker} is not present in price data.")
    weights.iloc[0, weights.columns.get_loc(ticker)] = 1.0
    weights.iloc[0] = weights.iloc[0].fillna(0.0)
    return weights
