"""Run macro regime detection research."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from quant_research_lab.data.fred_data import FredClient
from quant_research_lab.data.market_data import download_prices
from quant_research_lab.macro.macro_features import build_macro_features, standardize_features
from quant_research_lab.macro.regime_detection import (
    asset_returns_by_regime,
    detect_regimes,
    label_regimes,
    transition_matrix,
)
from quant_research_lab.utils.config import ensure_directories, load_yaml
from quant_research_lab.utils.logging import log_path, setup_logging
from quant_research_lab.visualization.charts import save_heatmap, save_pca_scatter, save_regime_timeline
from quant_research_lab.visualization.reports import save_dataframe, write_markdown_report


def _save_regime_colored_price(prices: pd.Series, regimes: pd.Series, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    monthly = prices.resample("MS").last()
    frame = pd.concat([monthly.rename("price"), regimes.rename("regime")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(monthly.index, monthly.values, color="#222222", linewidth=1.0, alpha=0.5)
    sns.scatterplot(data=frame, x=frame.index, y="price", hue="regime", ax=ax, s=60)
    ax.set_title("SPY Price Colored by Macro Regime")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    """Build macro features, detect regimes, and compare asset returns by regime."""
    logger = setup_logging("macro_regime")
    ensure_directories("outputs/figures", "outputs/reports")
    cfg = load_yaml("configs/macro.yaml")["macro"]
    fred = FredClient()
    raw_macro = fred.fetch_dataset(cfg["fred_series"], start=cfg["start_date"], end=cfg.get("end_date"))
    features = build_macro_features(raw_macro)
    standardized = standardize_features(features)
    regimes, pca, _ = detect_regimes(
        standardized,
        n_regimes=int(cfg["n_regimes"]),
        pca_components=int(cfg["pca_components"]),
        random_state=int(cfg["random_state"]),
    )
    labels = label_regimes(regimes, features)
    regime_history = features.join(regimes).join(labels)
    prices = download_prices(cfg["asset_universe"], start=str(features.index.min().date()), end=cfg.get("end_date"))
    returns_by_regime = asset_returns_by_regime(prices, labels)
    transitions = transition_matrix(labels)

    figures = ROOT / "outputs" / "figures"
    reports = ROOT / "outputs" / "reports"
    save_dataframe(regime_history, reports / "macro_regime_history.csv")
    save_dataframe(returns_by_regime, reports / "macro_asset_returns_by_regime.csv")
    save_dataframe(transitions, reports / "macro_regime_transition_matrix.csv")
    save_regime_timeline(labels, figures / "macro_regime_timeline.png")
    if "SPY" in prices:
        _save_regime_colored_price(prices["SPY"], labels, figures / "macro_spy_regime_colored.png")
    save_pca_scatter(regimes[["PC1", "PC2"]], labels, figures / "macro_pca_scatter.png")
    save_heatmap(
        features.join(labels).groupby("regime").mean(numeric_only=True),
        figures / "macro_feature_heatmap.png",
        "Average Macro Features by Regime",
    )
    save_heatmap(transitions, figures / "macro_transition_matrix.png", "Regime Transition Matrix")
    save_heatmap(returns_by_regime, figures / "macro_asset_returns_by_regime.png", "Average Monthly Asset Returns by Regime", fmt=".2%")
    report_path = write_markdown_report(
        "Macro Regime Detection Model",
        {
            "Methodology": (
                "FRED macro series are converted to monthly features, standardized, reduced with PCA, "
                "and clustered into regimes. Asset managers use this type of model to condition "
                "allocation, scenario analysis, drawdown controls, and tactical risk budgets."
            ),
            "PCA Explained Variance": ", ".join(f"{v:.1%}" for v in pca.explained_variance_ratio_),
        },
        reports / "macro_regime_report.md",
    )
    log_path(logger, "Macro report", report_path)
    logger.info("Macro regime model complete.")


if __name__ == "__main__":
    main()
