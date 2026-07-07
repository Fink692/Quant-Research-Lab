"""Vector-friendly institutional backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_lab.backtesting.constraints import (
    PositionConstraints,
    apply_position_constraints,
)
from quant_research_lab.backtesting.costs import FinancingModel, TransactionCostModel
from quant_research_lab.backtesting.performance import calculate_performance_metrics
from quant_research_lab.backtesting.portfolio import Portfolio
from quant_research_lab.risk.drawdowns import drawdown_series


@dataclass
class BacktestResult:
    """Container for a completed backtest."""

    name: str
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    exposure: pd.Series
    transaction_costs: pd.Series
    cash_interest: pd.Series
    borrow_costs: pd.Series
    drawdown: pd.Series
    metrics: pd.Series


class BacktestEngine:
    """Run daily mark-to-market backtests with explicit execution costs."""

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_cash: float = 1_000_000.0,
        cost_model: TransactionCostModel | None = None,
        max_gross_exposure: float = 1.0,
        max_single_position: float = 1.0,
        long_only: bool = False,
        execution_lag: int = 1,
        risk_free_rate: float = 0.0,
        cash_rate: float = 0.0,
        debit_rate: float = 0.0,
        short_borrow_rate: float = 0.0,
    ) -> None:
        self.prices = prices.sort_index().ffill().dropna(how="all")
        self.initial_cash = initial_cash
        self.cost_model = cost_model or TransactionCostModel()
        self.max_gross_exposure = max_gross_exposure
        self.constraints = PositionConstraints(
            max_gross_exposure=max_gross_exposure,
            max_single_position=max_single_position,
            long_only=long_only,
        )
        self.execution_lag = execution_lag
        self.risk_free_rate = risk_free_rate
        self.financing_model = FinancingModel(
            cash_rate=cash_rate,
            debit_rate=debit_rate,
            short_borrow_rate=short_borrow_rate,
        )

    def _clip_weights(self, weights: pd.Series) -> pd.Series:
        return apply_position_constraints(weights, self.constraints)

    def run(
        self,
        name: str,
        target_weights: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
    ) -> BacktestResult:
        """Run a backtest from target weights.

        Non-null target-weight rows are rebalance decisions. Decisions are shifted
        by ``execution_lag`` periods before execution, preventing same-close
        lookahead bias in strategy examples.
        """
        target_weights = target_weights.reindex(self.prices.index)
        executable_targets = target_weights.shift(self.execution_lag)

        portfolio = Portfolio(cash=self.initial_cash)
        equity_values: list[float] = []
        returns: list[float] = []
        weights_rows: list[pd.Series] = []
        turnover_values: list[float] = []
        cost_values: list[float] = []
        cash_interest_values: list[float] = []
        borrow_cost_values: list[float] = []
        exposure_values: list[float] = []
        previous_equity = self.initial_cash

        for date, prices_row in self.prices.iterrows():
            prices_row = prices_row.dropna()
            target_row = executable_targets.loc[date].dropna()
            turnover = 0.0
            cost = 0.0
            if not target_row.empty:
                aligned_target = self._clip_weights(
                    target_row.reindex(prices_row.index).fillna(0.0)
                )
                turnover, cost = portfolio.rebalance(prices_row, aligned_target, self.cost_model)

            current_values = portfolio.position_values(prices_row)
            short_notional = abs(float(current_values[current_values < 0.0].sum()))
            cash_interest = self.financing_model.cash_interest(portfolio.cash)
            borrow_cost = self.financing_model.borrow_cost(short_notional)
            portfolio.cash += cash_interest - borrow_cost

            equity = portfolio.total_value(prices_row)
            period_return = equity / previous_equity - 1.0 if previous_equity else 0.0
            current_weights = portfolio.weights(prices_row).reindex(self.prices.columns).fillna(0.0)
            equity_values.append(equity)
            returns.append(period_return)
            weights_rows.append(current_weights)
            turnover_values.append(turnover / max(equity, 1e-12))
            cost_values.append(cost)
            cash_interest_values.append(cash_interest)
            borrow_cost_values.append(borrow_cost)
            exposure_values.append(float(current_weights.abs().sum()))
            previous_equity = equity

        index = self.prices.index
        equity_series = pd.Series(equity_values, index=index, name=name)
        returns_series = (
            pd.Series(returns, index=index, name=name)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        weights_frame = pd.DataFrame(weights_rows, index=index, columns=self.prices.columns)
        turnover_series = pd.Series(turnover_values, index=index, name="turnover")
        costs_series = pd.Series(cost_values, index=index, name="transaction_costs")
        cash_interest_series = pd.Series(cash_interest_values, index=index, name="cash_interest")
        borrow_costs_series = pd.Series(borrow_cost_values, index=index, name="borrow_costs")
        exposure_series = pd.Series(exposure_values, index=index, name="exposure")
        drawdown = drawdown_series(equity_series)
        metrics = calculate_performance_metrics(
            returns_series,
            benchmark_returns=benchmark_returns,
            risk_free_rate=self.risk_free_rate,
            turnover=turnover_series,
            exposure=exposure_series,
            costs=costs_series + borrow_costs_series - cash_interest_series.clip(upper=0.0),
        )
        metrics["total_cash_interest"] = float(cash_interest_series.sum())
        metrics["total_borrow_costs"] = float(borrow_costs_series.sum())
        metrics.name = name
        return BacktestResult(
            name=name,
            equity=equity_series,
            returns=returns_series,
            weights=weights_frame,
            turnover=turnover_series,
            exposure=exposure_series,
            transaction_costs=costs_series,
            cash_interest=cash_interest_series,
            borrow_costs=borrow_costs_series,
            drawdown=drawdown,
            metrics=metrics,
        )
