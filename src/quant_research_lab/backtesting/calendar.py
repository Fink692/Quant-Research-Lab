"""Rebalance calendar helpers."""

from __future__ import annotations

import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.DatetimeIndex:
    """Return dates closest to a requested pandas rebalance frequency."""
    if len(index) == 0:
        return index
    series = pd.Series(index=index, data=index)
    grouped = series.resample(frequency).last().dropna()
    return pd.DatetimeIndex(grouped.values)


def is_rebalance_date(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.Series:
    """Return a boolean Series marking rebalance dates."""
    dates = rebalance_dates(index, frequency)
    return pd.Series(index=index, data=index.isin(dates), name="is_rebalance_date")
