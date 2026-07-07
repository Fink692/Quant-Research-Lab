"""Run institutional backtesting engine examples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from quant_research_lab.backtesting.costs import TransactionCostModel
from quant_research_lab.backtesting.engine import BacktestEngine
from quant_research_lab.backtesting.performance import monthly_returns_table, strategy_comparison
from quant_research_lab.data.market_data import calculate_returns, download_prices
from quant_research_lab.strategies.momentum import buy_and_hold_weights, monthly_momentum_weights
from quant_research_lab.strategies.pairs_trading import estimate_hedge_ratio, generate_pair_weights
from quant_research_lab.utils.config import ensure_directories, load_yaml
from quant_research_lab.utils.logging import log_path, setup_logging
from quant_research_lab.visualization.charts import (
    save_drawdown_chart,
    save_line_chart,
    save_monthly_returns_heatmap,
)
from quant_research_lab.visualization.reports import save_dataframe, write_markdown_report


def main() -> None:
    """Run buy-and-hold, momentum rotation, and pairs strategies with transaction costs."""
    logger = setup_logging("backtest")
    ensure_directories("outputs/figures", "outputs/reports", "outputs/backtests")
    config = load_yaml("configs/backtest.yaml")
    cfg = config["backtest"]
    strategies = config["strategies"]
    benchmark = cfg["benchmark"]
    momentum_universe = strategies["momentum_rotation"]["universe"]
    pair = strategies["pairs_trading"]["pair"]
    tickers = sorted(set(momentum_universe + pair + [benchmark]))
    prices = download_prices(tickers, start=cfg["start_date"], end=cfg.get("end_date"))
    returns = calculate_returns(prices)
    benchmark_returns = returns[benchmark].dropna()
    cost_model = TransactionCostModel(
        commission_bps=float(cfg["commission_bps"]),
        slippage_bps=float(cfg["slippage_bps"]),
        minimum_fee=float(cfg["minimum_fee"]),
    )
    engine = BacktestEngine(
        prices=prices,
        initial_cash=float(cfg["initial_cash"]),
        cost_model=cost_model,
        max_gross_exposure=float(cfg["max_gross_exposure"]),
        max_single_position=float(cfg.get("max_single_position", 1.0)),
        risk_free_rate=float(cfg["risk_free_rate"]),
        cash_rate=float(cfg.get("cash_rate", 0.0)),
        debit_rate=float(cfg.get("debit_rate", 0.0)),
        short_borrow_rate=float(cfg.get("short_borrow_rate", 0.0)),
    )

    buy_hold = buy_and_hold_weights(prices, strategies["buy_and_hold"]["ticker"])
    momentum = monthly_momentum_weights(
        prices[momentum_universe],
        lookback_days=int(strategies["momentum_rotation"]["lookback_days"]),
        top_n=int(strategies["momentum_rotation"]["top_n"]),
        max_gross_exposure=float(cfg["max_gross_exposure"]),
    ).reindex(columns=prices.columns)
    hedge_ratio, intercept = estimate_hedge_ratio(prices[pair[0]], prices[pair[1]])
    pair_weights = generate_pair_weights(
        prices,
        y=pair[0],
        x=pair[1],
        hedge_ratio=hedge_ratio,
        lookback=int(strategies["pairs_trading"]["lookback"]),
        entry_z=float(strategies["pairs_trading"]["entry_z"]),
        exit_z=float(strategies["pairs_trading"]["exit_z"]),
    )

    results = {
        "buy_and_hold": engine.run("buy_and_hold", buy_hold, benchmark_returns=benchmark_returns),
        "momentum_rotation": engine.run(
            "momentum_rotation", momentum, benchmark_returns=benchmark_returns
        ),
        "pairs_trading": engine.run(
            "pairs_trading", pair_weights, benchmark_returns=benchmark_returns
        ),
    }
    comparison = strategy_comparison({name: result.metrics for name, result in results.items()})

    figures = ROOT / "outputs" / "figures"
    reports = ROOT / "outputs" / "reports"
    backtests = ROOT / "outputs" / "backtests"
    save_dataframe(comparison, reports / "backtest_strategy_comparison.csv")
    equity_frame = pd.concat({name: result.equity for name, result in results.items()}, axis=1)
    returns_frame = pd.concat({name: result.returns for name, result in results.items()}, axis=1)
    turnover_frame = pd.concat({name: result.turnover for name, result in results.items()}, axis=1)
    exposure_frame = pd.concat({name: result.exposure for name, result in results.items()}, axis=1)
    save_dataframe(equity_frame, backtests / "backtest_equity_curves.csv")
    save_dataframe(returns_frame, backtests / "backtest_daily_returns.csv")
    save_dataframe(turnover_frame, backtests / "backtest_turnover.csv")
    save_dataframe(exposure_frame, backtests / "backtest_exposure.csv")
    save_dataframe(
        pd.concat({name: result.borrow_costs for name, result in results.items()}, axis=1),
        backtests / "backtest_borrow_costs.csv",
    )
    save_line_chart(
        equity_frame,
        figures / "backtest_equity_curves.png",
        "Strategy Equity Curves",
        "Portfolio Value",
    )
    save_drawdown_chart(
        results["momentum_rotation"].equity,
        figures / "backtest_momentum_drawdown.png",
        "Momentum Strategy Drawdown",
    )
    save_line_chart(
        returns_frame.rolling(63).mean() / returns_frame.rolling(63).std() * (252**0.5),
        figures / "backtest_rolling_sharpe.png",
        "Rolling 63-Day Sharpe Ratio",
        "Sharpe",
    )
    save_line_chart(
        turnover_frame, figures / "backtest_turnover.png", "Daily Turnover", "Turnover / Equity"
    )
    save_line_chart(
        exposure_frame, figures / "backtest_exposure.png", "Gross Exposure", "Gross Exposure"
    )
    save_monthly_returns_heatmap(
        monthly_returns_table(results["momentum_rotation"].returns),
        figures / "backtest_monthly_returns_heatmap.png",
    )
    report_path = write_markdown_report(
        "Institutional Backtesting Engine",
        {
            "Methodology": (
                "The engine tracks cash, holdings, mark-to-market equity, rebalancing, transaction costs, "
                "slippage, turnover, exposure, and benchmark-relative alpha/beta. Signals are shifted by "
                "one period before execution to avoid same-close lookahead bias."
            ),
            "Strategies": "Examples include buy-and-hold, monthly momentum rotation, and a pairs trading allocation.",
        },
        reports / "backtesting_engine_report.md",
    )
    log_path(logger, "Strategy comparison", reports / "backtest_strategy_comparison.csv")
    log_path(logger, "Backtest report", report_path)
    logger.info("Backtesting engine run complete.")


if __name__ == "__main__":
    main()
