"""Matplotlib and seaborn chart helpers for saved research artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from quant_research_lab.risk.drawdowns import drawdown_series

sns.set_theme(style="whitegrid", context="talk")


def _prepare_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_line_chart(
    data: pd.Series | pd.DataFrame, path: str | Path, title: str, ylabel: str = ""
) -> Path:
    """Save a line chart."""
    output = _prepare_path(path)
    ax = data.plot(figsize=(12, 6), linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=160)
    plt.close(ax.figure)
    return output


def save_drawdown_chart(equity: pd.Series, path: str | Path, title: str = "Drawdown") -> Path:
    """Save a drawdown chart from an equity curve."""
    output = _prepare_path(path)
    drawdown = drawdown_series(equity)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#b22222", alpha=0.35)
    ax.plot(drawdown.index, drawdown.values, color="#8b0000", linewidth=1.3)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_heatmap(data: pd.DataFrame, path: str | Path, title: str, fmt: str = ".2f") -> Path:
    """Save a heatmap."""
    output = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(data, annot=True, cmap="RdBu_r", center=0, fmt=fmt, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_allocation_pie(
    weights: pd.Series, path: str | Path, title: str = "Asset Allocation"
) -> Path:
    """Save an allocation pie chart."""
    output = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(8, 8))
    weights[weights.abs() > 0].plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_risk_return_scatter(
    summary: pd.DataFrame, path: str | Path, title: str = "Risk/Return"
) -> Path:
    """Save annualized return versus volatility scatter plot."""
    output = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(summary["annualized_volatility"], summary["annualized_return"], s=90)
    for label, row in summary.iterrows():
        ax.annotate(
            label,
            (row["annualized_volatility"], row["annualized_return"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_zscore_chart(
    zscore: pd.Series,
    path: str | Path,
    entry_z: float,
    exit_z: float,
    title: str = "Spread Z-Score",
) -> Path:
    """Save z-score chart with entry and exit bands."""
    output = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(zscore.index, zscore.values, color="#1f4e79", linewidth=1.3)
    for level, color, style in [
        (entry_z, "#b22222", "--"),
        (-entry_z, "#228b22", "--"),
        (exit_z, "#666666", ":"),
        (-exit_z, "#666666", ":"),
        (0, "#111111", "-"),
    ]:
        ax.axhline(level, color=color, linestyle=style, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Z-score")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_regime_timeline(
    regimes: pd.Series, path: str | Path, title: str = "Macro Regime Timeline"
) -> Path:
    """Save categorical regime history."""
    output = _prepare_path(path)
    codes = regimes.astype("category").cat.codes
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(regimes.index, codes, where="post", linewidth=2)
    categories = list(regimes.astype("category").cat.categories)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_pca_scatter(scores: pd.DataFrame, regimes: pd.Series, path: str | Path) -> Path:
    """Save PCA scatter colored by regime label."""
    output = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = scores.join(regimes.rename("regime")).dropna()
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue="regime", ax=ax, s=70)
    ax.set_title("Macro PCA Regime Map")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_options_smile(options: pd.DataFrame, path: str | Path, option_type: str = "c") -> Path:
    """Save implied volatility smiles by expiration."""
    output = _prepare_path(path)
    frame = options[options["option_type"].str.lower().str[0] == option_type.lower()[0]]
    fig, ax = plt.subplots(figsize=(12, 7))
    for expiry, group in frame.groupby("expiration"):
        ax.plot(
            group["moneyness"],
            group["implied_volatility"],
            marker="o",
            linewidth=1.4,
            label=str(pd.Timestamp(expiry).date()),
        )
    ax.set_title("Implied Volatility Smile")
    ax.set_xlabel("Strike / Spot")
    ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_3d_surface(
    xx, yy, zz, path: str | Path, title: str = "Implied Volatility Surface"
) -> Path:
    """Save a 3D implied volatility surface."""
    output = _prepare_path(path)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Time to Expiry")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_monthly_returns_heatmap(monthly_table: pd.DataFrame, path: str | Path) -> Path:
    """Save monthly returns heatmap."""
    output = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(monthly_table, cmap="RdYlGn", center=0, annot=True, fmt=".1%", ax=ax)
    ax.set_title("Monthly Returns")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
