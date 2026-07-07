"""Drawdown analytics."""

from __future__ import annotations

import pandas as pd


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Compound periodic returns into an equity curve."""
    curve = initial_value * (1.0 + returns.fillna(0.0)).cumprod()
    curve.name = returns.name or "equity"
    return curve


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate percentage drawdown from the running high-water mark."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    drawdown.name = "drawdown"
    return drawdown


def max_drawdown(returns_or_equity: pd.Series, input_is_returns: bool = True) -> float:
    """Return the maximum drawdown as a negative decimal."""
    equity = equity_curve(returns_or_equity) if input_is_returns else returns_or_equity
    return float(drawdown_series(equity).min())


def drawdown_table(equity: pd.Series) -> pd.DataFrame:
    """Return a simple drawdown table with equity, high-water mark, and drawdown."""
    high_water = equity.cummax()
    return pd.DataFrame(
        {
            "equity": equity,
            "high_water_mark": high_water,
            "drawdown": equity / high_water - 1.0,
        }
    )
