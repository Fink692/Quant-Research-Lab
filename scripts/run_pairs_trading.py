"""Run cointegration-based pairs trading research."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from quant_research_lab.data.market_data import download_prices
from quant_research_lab.strategies.pairs_trading import (
    backtest_pair_strategy,
    estimate_hedge_ratio,
    find_cointegrated_pairs,
    pair_performance_report,
)
from quant_research_lab.utils.config import ensure_directories, load_yaml
from quant_research_lab.utils.logging import log_path, setup_logging
from quant_research_lab.visualization.charts import (
    save_drawdown_chart,
    save_line_chart,
    save_zscore_chart,
)
from quant_research_lab.visualization.reports import save_dataframe, write_markdown_report


def _save_trade_marker_chart(strategy, path: Path, entry_z: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = strategy["signal"].diff().fillna(0) != 0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy.index, strategy["spread"], color="#1f4e79", linewidth=1.3, label="Spread")
    long_entries = entries & strategy["signal"].eq(1)
    short_entries = entries & strategy["signal"].eq(-1)
    exits = entries & strategy["signal"].eq(0)
    ax.scatter(
        strategy.index[long_entries],
        strategy.loc[long_entries, "spread"],
        color="#228b22",
        marker="^",
        s=70,
        label="Long spread",
    )
    ax.scatter(
        strategy.index[short_entries],
        strategy.loc[short_entries, "spread"],
        color="#b22222",
        marker="v",
        s=70,
        label="Short spread",
    )
    ax.scatter(
        strategy.index[exits],
        strategy.loc[exits, "spread"],
        color="#333333",
        marker="x",
        s=55,
        label="Exit",
    )
    ax.set_title(f"Spread Trade Markers (entry z={entry_z})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    """Find cointegrated pairs, backtest the top candidate, and save outputs."""
    logger = setup_logging("pairs_trading")
    ensure_directories("outputs/figures", "outputs/reports", "outputs/backtests")
    cfg = load_yaml("configs/assets.yaml")["pairs_trading"]
    prices = download_prices(cfg["universe"], start=cfg["start_date"], end=cfg.get("end_date"))
    ranked_pairs = find_cointegrated_pairs(prices)
    if ranked_pairs.empty:
        logger.warning(
            "No cointegrated pair found; falling back to first two tickers for demonstration."
        )
        y, x = cfg["universe"][:2]
        hedge_ratio, intercept = estimate_hedge_ratio(prices[y], prices[x])
        ranked_pairs = ranked_pairs._append(
            {
                "y": y,
                "x": x,
                "p_value": float("nan"),
                "test_statistic": float("nan"),
                "hedge_ratio": hedge_ratio,
                "intercept": intercept,
                "correlation": prices[y].pct_change().corr(prices[x].pct_change()),
            },
            ignore_index=True,
        )

    top = ranked_pairs.iloc[0]
    strategy = backtest_pair_strategy(
        prices,
        y=str(top["y"]),
        x=str(top["x"]),
        hedge_ratio=float(top["hedge_ratio"]),
        intercept=float(top["intercept"]),
        lookback=int(cfg["lookback"]),
        entry_z=float(cfg["entry_z"]),
        exit_z=float(cfg["exit_z"]),
        transaction_cost_bps=float(cfg["transaction_cost_bps"]),
    )
    perf = pair_performance_report(strategy)

    figures = ROOT / "outputs" / "figures"
    reports = ROOT / "outputs" / "reports"
    backtests = ROOT / "outputs" / "backtests"
    save_dataframe(
        ranked_pairs.head(int(cfg["top_n_pairs"])), reports / "pairs_best_cointegrated_pairs.csv"
    )
    save_dataframe(strategy, backtests / "pairs_strategy_timeseries.csv")
    perf_path = save_dataframe(perf, reports / "pairs_strategy_performance.csv")
    save_line_chart(
        strategy[["y_price", "x_price"]],
        figures / "pairs_price_comparison.png",
        "Pair Price Comparison",
        "Price",
    )
    save_line_chart(
        strategy["spread"], figures / "pairs_spread.png", "Cointegration Spread", "Spread"
    )
    save_zscore_chart(
        strategy["zscore"],
        figures / "pairs_zscore_bands.png",
        float(cfg["entry_z"]),
        float(cfg["exit_z"]),
    )
    _save_trade_marker_chart(strategy, figures / "pairs_trade_markers.png", float(cfg["entry_z"]))
    save_line_chart(
        strategy["equity"],
        figures / "pairs_equity_curve.png",
        "Pairs Strategy Equity Curve",
        "Growth of $1",
    )
    save_drawdown_chart(
        strategy["equity"], figures / "pairs_drawdown.png", "Pairs Strategy Drawdown"
    )
    report_path = write_markdown_report(
        "Pairs Trading Statistical Arbitrage Model",
        {
            "Methodology": (
                "The model ranks stock pairs by Engle-Granger cointegration p-value, estimates an OLS "
                "hedge ratio, trades spread z-score extremes, exits near mean reversion, and deducts "
                "transaction costs on signal changes."
            ),
            "Selected Pair": f"{top['y']} vs {top['x']} with hedge ratio {float(top['hedge_ratio']):.4f}.",
        },
        reports / "pairs_trading_report.md",
    )
    log_path(logger, "Pairs performance", perf_path)
    log_path(logger, "Pairs report", report_path)
    logger.info("Pairs trading research complete.")


if __name__ == "__main__":
    main()
