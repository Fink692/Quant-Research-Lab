"""Reusable signal generation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    """Calculate full-sample z-score for research diagnostics."""
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index, name="zscore")
    return ((series - series.mean()) / std).rename("zscore")


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score without using future observations."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (
        ((series - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).rename("zscore")
    )


def threshold_positions(zscores: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Convert z-scores into long/short/flat mean-reversion states."""
    state = 0
    positions: list[int] = []
    for value in zscores:
        if np.isnan(value):
            positions.append(state)
            continue
        if state == 0:
            if value <= -abs(entry_z):
                state = 1
            elif value >= abs(entry_z):
                state = -1
        elif state == 1 and value >= -abs(exit_z) or state == -1 and value <= abs(exit_z):
            state = 0
        positions.append(state)
    return pd.Series(positions, index=zscores.index, name="signal")
