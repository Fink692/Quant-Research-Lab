# API Reference

## Data

- `quant_research_lab.data.market_data.download_prices`: download adjusted prices with local cache and deterministic fallback.
- `quant_research_lab.data.market_data.calculate_returns`: calculate simple or log returns.
- `quant_research_lab.data.fred_data.FredClient`: fetch and align FRED data.
- `quant_research_lab.data.options_data.load_options_chain`: load options from CSV, yfinance, or synthetic fallback.
- `quant_research_lab.data.cache.DataCache`: file-backed cache for DataFrames and Python objects.

## Risk

- `portfolio_returns`: weighted portfolio return aggregation.
- `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`: core performance metrics.
- `beta_to_benchmark`, `alpha_to_benchmark`: benchmark-relative metrics.
- `rolling_volatility`, `rolling_beta`, `rolling_correlation`, `rolling_sharpe`: rolling risk diagnostics.
- `drawdown_series`, `max_drawdown`: high-water-mark drawdown tools.
- `historical_var`, `historical_cvar`, `gaussian_var`: value-at-risk models.

## Strategies

- `find_cointegrated_pairs`: Engle-Granger pair scan.
- `estimate_hedge_ratio`: OLS hedge ratio.
- `backtest_pair_strategy`: cost-adjusted pairs backtest.
- `monthly_momentum_weights`: monthly top-N momentum rotation weights.
- `buy_and_hold_weights`: initial buy-and-hold target weights.

## Macro

- `build_macro_features`: transform raw macro series into regime features.
- `standardize_features`: z-score macro feature matrix.
- `detect_regimes`: PCA and KMeans regime detection.
- `label_regimes`: descriptive regime labels.
- `transition_matrix`: one-step transition probabilities.
- `asset_returns_by_regime`: monthly asset return comparison by regime.

## Derivatives

- `black_scholes_price`: European call/put price.
- `implied_volatility`: numerical implied-volatility solver.
- `greeks`: delta, gamma, theta, vega, and rho bundle.
- `clean_options_chain`: quote and liquidity filters.
- `enrich_options_chain`: IV, theoretical value, moneyness, and Greeks.
- `build_surface_grid`: interpolated IV surface grid.

## Backtesting

- `TransactionCostModel`: commission and slippage bps model.
- `Portfolio`: cash and holdings state.
- `BacktestEngine`: daily mark-to-market engine with signal lag and costs.
- `calculate_performance_metrics`: institutional metric summary.
- `monthly_returns_table`: monthly return heatmap input.
- `strategy_comparison`: combined strategy metric table.
