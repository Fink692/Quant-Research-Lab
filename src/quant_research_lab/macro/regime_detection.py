"""Macro regime detection using PCA and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from quant_research_lab.data.market_data import calculate_returns


def fit_pca(features: pd.DataFrame, n_components: int = 2) -> tuple[PCA, pd.DataFrame]:
    """Fit PCA and return principal-component scores."""
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(features)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pca, pd.DataFrame(scores, index=features.index, columns=columns)


def detect_regimes(
    standardized_features: pd.DataFrame,
    n_regimes: int = 4,
    pca_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA, KMeans]:
    """Detect macro regimes from standardized features using PCA and KMeans."""
    pca, scores = fit_pca(standardized_features, n_components=pca_components)
    model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    labels = model.fit_predict(scores)
    regimes = scores.copy()
    regimes["regime_id"] = labels
    return regimes, pca, model


def label_regimes(regimes: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Assign descriptive labels based on average feature characteristics."""
    labels: dict[int, str] = {}
    combined = features.join(regimes["regime_id"], how="inner")
    grouped = combined.groupby("regime_id").mean(numeric_only=True)
    for regime_id, row in grouped.iterrows():
        inflation = row.get("inflation_yoy", 0.0)
        unemployment_change = row.get("unemployment_change_6m", 0.0)
        curve = row.get("yield_curve_10y_2y", 0.0)
        fed_change = row.get("fed_funds_change_6m", 0.0)
        if inflation > grouped.get("inflation_yoy", pd.Series([0])).median() and fed_change > 0:
            labels[int(regime_id)] = "Inflation shock"
        elif unemployment_change > 0 and curve < 0:
            labels[int(regime_id)] = "Recession risk"
        elif fed_change < 0 and curve > 0:
            labels[int(regime_id)] = "Liquidity rally"
        else:
            labels[int(regime_id)] = "Expansion"
    return regimes["regime_id"].map(labels).rename("regime")


def transition_matrix(regime_labels: pd.Series) -> pd.DataFrame:
    """Calculate one-period regime transition probabilities."""
    current = regime_labels.dropna().astype(str)
    transitions = pd.crosstab(current.shift(1), current, normalize="index")
    transitions.index.name = "from_regime"
    transitions.columns.name = "to_regime"
    return transitions.fillna(0.0)


def asset_returns_by_regime(prices: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
    """Compare monthly asset returns by detected macro regime."""
    monthly_prices = prices.resample("MS").last()
    monthly_returns = calculate_returns(monthly_prices).dropna(how="all")
    labels = regime_labels.reindex(monthly_returns.index).ffill()
    joined = monthly_returns.join(labels.rename("regime"), how="inner")
    return joined.groupby("regime").mean(numeric_only=True).T
