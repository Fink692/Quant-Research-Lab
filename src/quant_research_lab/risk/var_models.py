"""Value-at-risk and expected shortfall models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Estimate historical value-at-risk as a positive loss number."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(-np.quantile(clean, 1.0 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Estimate conditional value-at-risk as average tail loss."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    threshold = np.quantile(clean, 1.0 - confidence)
    return float(-clean[clean <= threshold].mean())


def gaussian_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Estimate parametric Gaussian value-at-risk as a positive loss number."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    return float(-(mu + sigma * norm.ppf(1.0 - confidence)))
