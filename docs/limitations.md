# Limitations

## Free Market Data

Free sources can be incomplete, delayed, revised, or adjusted differently than institutional feeds. Yahoo Finance data may not include delisted securities, point-in-time constituents, borrow costs, accurate corporate actions for all instruments, or exchange-grade quote history.

## FRED Data

Macroeconomic series are revised after initial publication. A rigorous point-in-time macro backtest should use vintage data. This repository uses current historical series, which can introduce revision bias.

## Options Data

Yahoo options chains are snapshots, not historical full-depth quote data. The implied volatility surface builder is useful for analytics and research structure, but production derivatives systems require exchange-quality quotes, no-arbitrage validation, dividend curves, rate curves, exercise style handling, and market microstructure checks.

## Backtesting

The engine models daily close-to-close fills with linear costs. It does not model order-book depth, partial fills, borrow constraints, financing, margin calls, taxes, exchange fees, latency, intraday volatility, or market impact. Strategy results should be interpreted as research diagnostics, not live trading expectations.

## Statistical Risk

Cointegration can be unstable. Macro regimes can be sensitive to feature selection and clustering assumptions. Momentum can be crowded and regime-dependent. Backtests can overfit through parameter selection. Every model should be stress-tested before being used for capital allocation.

## Disclaimer

This project is research software, not investment advice.
