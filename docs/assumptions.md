# Assumptions

## Market Data

- Yahoo Finance adjusted close prices are accepted as the default free source for daily research.
- Missing market data is forward-filled only after download alignment.
- If external providers fail, deterministic sample data is used so scripts remain runnable.
- Crypto tickers are treated like daily assets for portfolio analytics, even though crypto trades continuously.

## Portfolio Analytics

- Daily trading periods are annualized with 252 periods per year.
- Portfolio weights in `configs/assets.yaml` are normalized before return aggregation.
- The risk-free rate is an annual input and converted to daily where needed.

## Pairs Trading

- The hedge ratio is estimated with static OLS over the available sample.
- Transaction costs are charged when the spread signal changes.
- Borrow costs, locate availability, and hard-to-borrow constraints are not modeled.
- Signals are lagged before returns are realized.

## Macro Regimes

- FRED series are resampled to monthly frequency using last observation.
- PCA and KMeans are used as interpretable baseline methods.
- Regime labels are heuristic descriptions based on average feature values.

## Options

- Option prices use bid/ask mid.
- Contracts with bad quotes, zero bid/ask, low volume, low open interest, and expired dates are removed.
- Black-Scholes assumes European exercise, continuous rates, continuous dividend yield, and lognormal underlying returns.

## Backtesting

- Rebalances occur at the close of the execution date after signal lag.
- Slippage and commissions are linear bps assumptions.
- Position values can be negative for short exposure.
- Maximum gross exposure clips target weights when needed.
