"""Feature engineering for macro regime models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def to_monthly(data: pd.DataFrame, method: str = "last") -> pd.DataFrame:
    """Convert a time series DataFrame to monthly start frequency."""
    if method == "mean":
        monthly = data.resample("MS").mean()
    elif method == "sum":
        monthly = data.resample("MS").sum()
    else:
        monthly = data.resample("MS").last()
    return monthly.ffill()


def year_over_year(series: pd.Series, periods: int = 12) -> pd.Series:
    """Calculate year-over-year percentage change."""
    return series.pct_change(periods=periods) * 100.0


def build_macro_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build institutional-style macro features from raw FRED levels."""
    monthly = to_monthly(raw)
    features = pd.DataFrame(index=monthly.index)
    if "cpi" in monthly:
        features["inflation_yoy"] = year_over_year(monthly["cpi"])
        features["inflation_3m_change"] = monthly["cpi"].pct_change(3) * 400.0
    if "unemployment" in monthly:
        features["unemployment"] = monthly["unemployment"]
        features["unemployment_change_6m"] = monthly["unemployment"].diff(6)
    if "fed_funds" in monthly:
        features["fed_funds"] = monthly["fed_funds"]
        features["fed_funds_change_6m"] = monthly["fed_funds"].diff(6)
    if {"treasury_10y", "treasury_2y"}.issubset(monthly.columns):
        features["yield_curve_10y_2y"] = monthly["treasury_10y"] - monthly["treasury_2y"]
    if "industrial_production" in monthly:
        features["industrial_production_yoy"] = year_over_year(monthly["industrial_production"])
    if {"baa_yield", "aaa_yield"}.issubset(monthly.columns):
        features["credit_spread_proxy"] = monthly["baa_yield"] - monthly["aaa_yield"]
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("Macro feature set is empty after transformations.")
    return features


def standardize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Standardize macro features to zero mean and unit variance."""
    scaler = StandardScaler()
    values = scaler.fit_transform(features)
    return pd.DataFrame(values, index=features.index, columns=features.columns)
