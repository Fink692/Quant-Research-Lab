"""Portfolio return, risk, and benchmark-relative metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.risk.drawdowns import equity_curve, max_drawdown

TRADING_DAYS = 252


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float] | pd.Series) -> pd.Series:
    """Calculate weighted portfolio returns from asset returns and target weights."""
    weight_series = pd.Series(weights, dtype=float)
    weight_series = weight_series / weight_series.sum()
    aligned = returns.loc[:, returns.columns.intersection(weight_series.index)]
    weighted = aligned.mul(weight_series.loc[aligned.columns], axis=1).sum(axis=1)
    weighted.name = "portfolio"
    return weighted


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Calculate geometric annualized return."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    compounded = (1.0 + clean).prod()
    years = len(clean) / periods_per_year
    return float(compounded ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Calculate annualized standard deviation of periodic returns."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Calculate annualized downside deviation below a periodic target return."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    downside = np.minimum(clean - target_return, 0.0)
    return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Calculate annualized Sharpe ratio using an annual risk-free rate."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    excess = clean - risk_free_rate / periods_per_year
    vol = clean.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Calculate annualized Sortino ratio using downside volatility."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    excess_ann = annualized_return(clean, periods_per_year) - risk_free_rate
    downside = downside_deviation(clean, risk_free_rate / periods_per_year, periods_per_year)
    if downside == 0 or np.isnan(downside):
        return float("nan")
    return float(excess_ann / downside)


def beta_to_benchmark(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Estimate beta versus a benchmark with covariance/variance."""
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or aligned.iloc[:, 1].var(ddof=1) == 0:
        return float("nan")
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    return float(covariance / aligned.iloc[:, 1].var(ddof=1))


def alpha_to_benchmark(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Estimate annualized CAPM alpha versus a benchmark."""
    beta = beta_to_benchmark(asset_returns, benchmark_returns)
    asset_ann = annualized_return(asset_returns, periods_per_year)
    bench_ann = annualized_return(benchmark_returns, periods_per_year)
    if np.isnan(beta) or np.isnan(asset_ann) or np.isnan(bench_ann):
        return float("nan")
    return float(asset_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate)))


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Calculate Calmar ratio as annualized return divided by absolute max drawdown."""
    ann_return = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns, input_is_returns=True))
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ann_return / mdd)


def rolling_volatility(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Calculate rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Calculate rolling annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods_per_year
    return excess.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
    """Calculate rolling beta versus a benchmark."""
    covariance = asset_returns.rolling(window).cov(benchmark_returns)
    variance = benchmark_returns.rolling(window).var()
    return covariance / variance


def rolling_correlation(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
    """Calculate rolling correlation versus a benchmark."""
    return asset_returns.rolling(window).corr(benchmark_returns)


def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Return a standard institutional performance metric summary."""
    summary = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns, input_is_returns=True),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "ending_value_1usd": equity_curve(returns, 1.0).iloc[-1] if not returns.dropna().empty else np.nan,
    }
    if benchmark_returns is not None:
        summary["beta"] = beta_to_benchmark(returns, benchmark_returns)
        summary["alpha"] = alpha_to_benchmark(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
    return pd.Series(summary, name=returns.name or "strategy")
