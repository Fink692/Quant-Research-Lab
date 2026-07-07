# Methodology

## Portfolio Risk Metrics

Returns are calculated from adjusted prices. Simple returns are used for realized performance and compounding. Log returns are available for research workflows where additive time aggregation is useful.

Annualized return uses geometric compounding:

```text
CAGR = (product(1 + r_t)) ** (periods_per_year / N) - 1
```

Annualized volatility scales daily standard deviation by the square root of 252. The Sharpe ratio measures excess return per unit of total volatility. The Sortino ratio replaces total volatility with downside deviation, which is more relevant when upside volatility is not considered harmful.

Drawdown is measured from the running high-water mark. Maximum drawdown captures peak-to-trough capital loss, which is central to institutional risk control because investors experience path dependency and redemption pressure.

Beta is covariance with the benchmark divided by benchmark variance. Alpha is the annualized return unexplained by CAPM-style benchmark exposure.

## Cointegration And Pairs Trading

Pairs trading looks for two nonstationary price series whose linear combination is stationary. The repository uses the Engle-Granger test to rank candidate pairs by cointegration p-value.

The hedge ratio is estimated with OLS:

```text
Y_t = alpha + beta X_t + epsilon_t
spread_t = Y_t - alpha - beta X_t
```

The strategy calculates a rolling z-score for the spread. It goes long spread when the z-score is below a negative threshold, short spread when it is above a positive threshold, and exits when the spread mean-reverts near zero.

Signals are shifted before returns are applied. This avoids using the closing spread from the same timestamp to trade the same close.

## Macro Regime Detection

FRED macro series are converted to monthly frequency, cleaned, and transformed into economically meaningful features such as inflation year-over-year, unemployment change, policy-rate change, yield-curve slope, industrial production growth, and credit spread proxy.

Features are standardized before PCA so variables with large numeric scales do not dominate. PCA reduces the macro state space. KMeans clustering assigns regime IDs. Labels such as Expansion, Inflation shock, Recession risk, and Liquidity rally are assigned from average feature characteristics.

The result is not a causal macro forecasting model. It is a descriptive state-classification tool that can support allocation review, stress testing, and risk-budget discussion.

## Black-Scholes And Implied Volatility

The options module implements European Black-Scholes pricing with continuous dividend yield. Implied volatility is solved by root-finding the volatility that makes theoretical price equal market mid price.

The Greeks measure sensitivities:

- Delta: sensitivity to underlying price
- Gamma: sensitivity of delta to underlying price
- Theta: sensitivity to time decay
- Vega: sensitivity to volatility
- Rho: sensitivity to rates

The volatility smile is IV across strikes for a fixed expiration. The term structure is IV across expirations around at-the-money strikes. The surface interpolates IV across moneyness and time to expiry.

## Backtesting

The backtesting engine marks positions to market daily. It accepts target weights, shifts signals by one period by default, rebalances on non-null target rows, deducts transaction costs and slippage, and records turnover and gross exposure.

Performance reporting includes CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, win rate, turnover, exposure, total transaction costs, alpha, and beta.

## Transaction Costs

The default cost model is linear:

```text
cost = max(abs(traded_notional) * (commission_bps + slippage_bps) / 10000, minimum_fee)
```

This is intentionally transparent. It can be replaced with broker-specific, asset-specific, spread-aware, or market-impact-aware models.

## Portfolio Construction

The portfolio construction module includes sample covariance, Ledoit-Wolf shrinkage covariance, and exponentially weighted covariance estimators. Shrinkage is useful when the asset universe is large relative to the observation window because sample covariance matrices can be noisy and unstable.

The optimization module includes minimum variance, mean-variance utility, and equal-risk-contribution risk parity. These optimizers are baseline institutional allocation tools rather than full production optimizers. They make constraints explicit and return transparent weights that can be inspected, stress-tested, and backtested.

## Factor Risk

The factor model estimates linear exposures of assets to return factors using OLS. Estimated betas, residual volatility, and R-squared can be used to decompose portfolio variance into systematic and idiosyncratic components.

## Stress Testing And Simulation

Stress tests apply deterministic asset shocks to portfolio weights to estimate scenario loss. Monte Carlo simulation uses expected returns, covariance, and weights to generate a distribution of terminal wealth paths. Both tools are useful complements to historical backtests because they ask how a portfolio behaves outside a single realized path.

## Common Backtesting Biases

The examples explicitly avoid same-close lookahead by shifting signals before execution. Other important risks include survivorship bias, stale constituents, data snooping, overfitting, corporate-action errors, omitted borrow fees, short-sale constraints, market impact, and unrealistic liquidity assumptions.
