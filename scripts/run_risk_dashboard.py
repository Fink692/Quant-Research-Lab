"""Run the multi-asset portfolio risk dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from quant_research_lab.data.market_data import calculate_returns, download_prices
from quant_research_lab.risk.drawdowns import equity_curve
from quant_research_lab.risk.portfolio_metrics import (
    performance_summary,
    portfolio_returns,
    rolling_beta,
    rolling_correlation,
    rolling_sharpe,
    rolling_volatility,
)
from quant_research_lab.utils.config import ensure_directories, load_yaml
from quant_research_lab.utils.logging import log_path, setup_logging
from quant_research_lab.visualization.charts import (
    save_allocation_pie,
    save_drawdown_chart,
    save_heatmap,
    save_line_chart,
    save_risk_return_scatter,
)
from quant_research_lab.visualization.reports import save_dataframe, write_markdown_report


def main() -> None:
    """Download prices, calculate portfolio risk metrics, and save dashboard outputs."""
    logger = setup_logging("risk_dashboard")
    ensure_directories("outputs/figures", "outputs/reports")
    cfg = load_yaml("configs/assets.yaml")["portfolio"]
    weights = pd.Series(cfg["assets"], dtype=float)
    benchmark = cfg["benchmark"]
    tickers = list(weights.index) + [benchmark]
    prices = download_prices(tickers, start=cfg["start_date"], end=cfg.get("end_date"))
    returns = calculate_returns(prices)
    port_returns = portfolio_returns(returns, weights)
    benchmark_returns = returns[benchmark].dropna()
    port_equity = equity_curve(port_returns, initial_value=1.0)
    benchmark_equity = equity_curve(benchmark_returns, initial_value=1.0)

    risk_free_rate = float(cfg.get("risk_free_rate", 0.0))
    summary = performance_summary(port_returns, benchmark_returns, risk_free_rate=risk_free_rate)
    asset_summaries = {
        col: performance_summary(returns[col].dropna(), risk_free_rate=risk_free_rate)
        for col in returns.columns
        if col != benchmark
    }
    asset_summary = pd.DataFrame(asset_summaries).T

    figures = ROOT / "outputs" / "figures"
    reports = ROOT / "outputs" / "reports"
    save_line_chart(
        pd.concat([port_equity.rename("Portfolio"), benchmark_equity.rename(benchmark)], axis=1),
        figures / "risk_portfolio_equity_curve.png",
        "Portfolio Equity Curve vs Benchmark",
        "Growth of $1",
    )
    save_drawdown_chart(port_equity, figures / "risk_portfolio_drawdown.png", "Portfolio Drawdown")
    save_heatmap(
        returns[list(weights.index)].corr(),
        figures / "risk_correlation_heatmap.png",
        "Asset Return Correlation",
    )
    save_line_chart(
        rolling_volatility(port_returns).rename("Rolling Volatility"),
        figures / "risk_rolling_volatility.png",
        "Rolling Annualized Volatility",
        "Volatility",
    )
    save_line_chart(
        rolling_beta(port_returns, benchmark_returns).rename("Rolling Beta"),
        figures / "risk_rolling_beta.png",
        "Rolling Beta vs Benchmark",
        "Beta",
    )
    save_line_chart(
        rolling_correlation(port_returns, benchmark_returns).rename("Rolling Correlation"),
        figures / "risk_rolling_correlation.png",
        "Rolling Correlation vs Benchmark",
        "Correlation",
    )
    save_line_chart(
        rolling_sharpe(port_returns, risk_free_rate=risk_free_rate).rename("Rolling Sharpe"),
        figures / "risk_rolling_sharpe.png",
        "Rolling Sharpe Ratio",
        "Sharpe",
    )
    save_allocation_pie(
        weights / weights.sum(), figures / "risk_asset_allocation.png", "Strategic Asset Allocation"
    )
    save_risk_return_scatter(
        asset_summary, figures / "risk_return_scatter.png", "Asset Risk/Return Profile"
    )

    summary_path = save_dataframe(summary, reports / "risk_performance_summary.csv")
    asset_path = save_dataframe(asset_summary, reports / "risk_asset_metrics.csv")
    report_path = write_markdown_report(
        "Multi-Asset Portfolio Risk Dashboard",
        {
            "Purpose": (
                "This report summarizes return, volatility, downside risk, benchmark beta, and drawdown. "
                "These metrics matter because institutional portfolios are evaluated on both absolute "
                "compounding and the amount of risk required to earn that return."
            ),
            "Artifacts": (
                "Figures include equity, drawdown, allocation, correlation, rolling volatility, rolling "
                "beta, rolling correlation, rolling Sharpe, and asset risk/return scatter plots."
            ),
        },
        reports / "risk_dashboard_report.md",
    )
    for label, path in [
        ("Performance summary", summary_path),
        ("Asset metrics", asset_path),
        ("Markdown report", report_path),
    ]:
        log_path(logger, label, path)
    logger.info("Risk dashboard complete.")


if __name__ == "__main__":
    main()
