"""Execution simulator helpers."""

from __future__ import annotations

import pandas as pd


def target_position_values(equity: float, target_weights: pd.Series) -> pd.Series:
    """Convert target portfolio weights into dollar position values."""
    return target_weights.fillna(0.0).astype(float) * equity


def calculate_turnover(current_values: pd.Series, target_values: pd.Series) -> float:
    """Calculate one-way absolute notional turnover for a rebalance."""
    aligned_current, aligned_target = current_values.align(target_values, fill_value=0.0)
    return float((aligned_target - aligned_current).abs().sum())
