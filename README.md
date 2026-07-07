# Quant Research Lab

[![CI](https://github.com/Fink692/Quant-Research-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Fink692/Quant-Research-Lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-000000)
![Lint](https://img.shields.io/badge/lint-ruff-red)
[![Docs](https://github.com/Fink692/Quant-Research-Lab/actions/workflows/docs.yml/badge.svg)](https://github.com/Fink692/Quant-Research-Lab/actions/workflows/docs.yml)

Quant Research Lab is a local Python research framework for institutional-style portfolio analytics, statistical arbitrage, macro regime research, options volatility analysis, and transaction-cost-aware backtesting.

This is structured as a package-style repository rather than a notebook dump. The modules are reusable from scripts, tests, notebooks, VS Code, PyCharm, or an installed Python environment.

## Why This Project Matters

Professional quant research is judged on reproducibility, risk controls, clean data handling, and the ability to convert research into repeatable workflows. This repository combines five systems that are common in asset management, banking, prop trading, and portfolio analytics:

1. Multi-asset portfolio risk dashboard
2. Pairs trading statistical arbitrage model
3. Macro regime detection model
4. Implied volatility surface builder
5. Institutional backtesting engine with transaction costs

The implementation emphasizes modular architecture, local execution, graceful data-provider failure handling, saved artifacts, and testable analytics code.

## Repository Structure

```text
Quant-Research-Lab/
  configs/                  YAML parameters for assets, macro data, options, and backtests
  scripts/                  Command-line entry points for each research system
  notebooks/                Optional research demonstration notebooks
  src/quant_research_lab/   Reusable Python package
  tests/                    Pytest regression tests
  data/                     Raw, processed, and cached local data
  outputs/                  Saved figures, reports, and backtest results
  docs/                     Methodology, assumptions, limitations, and API reference
```

## Installation

```bash
cd Quant-Research-Lab
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For editable package development:

```bash
pip install -e .
```

Development tooling:

```bash
pip install -r requirements-dev.txt
pre-commit install
ruff check src tests scripts
black --check src tests scripts
mypy src --ignore-missing-imports
```

## Local Setup

Copy `.env.example` to `.env` if you have optional API keys:

```bash
copy .env.example .env
```

FRED can be accessed through a public CSV endpoint without a key, but `FRED_API_KEY` is supported. Yahoo Finance access is handled through `yfinance`. If a provider fails or no key is present, the scripts use deterministic sample data and log the fallback.

## Command-Line Usage

```bash
python scripts/run_risk_dashboard.py
python scripts/run_pairs_trading.py
python scripts/run_macro_regime_model.py
python scripts/run_vol_surface.py
python scripts/run_backtest.py
pytest
```

After `pip install -e .`, the same workflows are available as console commands:

```bash
qrl-risk
qrl-pairs
qrl-macro
qrl-vol
qrl-backtest
```

Outputs are written to:

```text
outputs/figures/
outputs/reports/
outputs/backtests/
```

## Research Modules

### 1. Multi-Asset Portfolio Risk Dashboard

Downloads prices, calculates simple and log returns, creates a weighted multi-asset portfolio, compares it against a benchmark, and saves figures and CSV summaries. Metrics include annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar, beta, alpha, rolling volatility, rolling beta, rolling correlation, and rolling Sharpe.

### 2. Pairs Trading Statistical Arbitrage

Tests all possible stock pairs for cointegration, ranks pairs by p-value, estimates hedge ratios with OLS, builds spreads, generates z-score entry and exit signals, deducts transaction costs, and saves strategy diagnostics.

### 3. Macro Regime Detection

Pulls FRED macro data, builds monthly macro features, standardizes them, applies PCA, clusters regimes, labels regimes with economic descriptions, and compares asset returns by regime.

### 4. Implied Volatility Surface Builder

Loads options chains from yfinance or manual CSV, cleans quotes, computes Black-Scholes implied volatility, calculates Greeks, creates smiles, term structure, and a 3D interpolated volatility surface.

### 5. Institutional Backtesting Engine

Tracks cash, holdings, target weights, transaction costs, slippage, turnover, exposure, drawdowns, and benchmark-relative performance. Includes buy-and-hold, momentum rotation, and pairs trading examples. Signals are shifted before execution to avoid same-close lookahead bias.

## Professional Tooling

The repository includes GitHub Actions CI, `ruff`, `black`, `mypy`, `pre-commit`, a Dockerfile, a VS Code dev container, package console commands, MkDocs documentation, and tracked example images for README/docs presentation.

## Additional Institutional Research Modules

The professionalization pass adds:

- Ledoit-Wolf covariance shrinkage and exponentially weighted covariance
- Minimum-variance, mean-variance, and risk-parity portfolio optimization
- Linear factor risk model estimation
- EWMA volatility and correlation
- Historical VaR/CVaR tail-risk reports
- Deterministic stress testing
- Monte Carlo portfolio path simulation
- Regime-aware allocation helpers
- Order/fill objects, rebalance calendars, position constraints, walk-forward splits, parameter sweeps, borrow costs, and cash financing hooks for backtesting

## Data Sources

- Yahoo Finance through `yfinance`
- FRED through `fredapi` or direct public CSV requests
- Manual options-chain CSV uploads
- Optional future support for Polygon and Alpha Vantage keys
- Deterministic local fallback data for reproducible offline runs

## Methodology Summary

Portfolio analytics use geometric compounding, annualized volatility, drawdown from high-water marks, downside volatility, and benchmark covariance estimates. Pairs trading uses Engle-Granger cointegration, OLS hedge ratios, rolling z-scores, and cost-adjusted close-to-close backtests. Macro regimes use engineered FRED features, PCA, and KMeans clustering. Options analytics use Black-Scholes pricing, numerical implied-volatility solving, Greeks, and surface interpolation. Backtests explicitly track costs, slippage, turnover, exposure, and benchmark alpha/beta.

## Example Outputs

- Portfolio equity curve and drawdown chart
- Correlation heatmap and allocation chart
- Cointegrated pair ranking table
- Pair spread, z-score, trade marker, and equity charts
- Macro regime timeline and PCA scatter
- Regime transition matrix and returns by regime
- IV smile, term structure, 3D surface, Greeks chart
- Strategy comparison table and monthly returns heatmap

Selected tracked examples are available in `docs/assets/` and on the MkDocs site:

![Backtest equity curves](docs/assets/backtest_equity_curves.png)

![Implied volatility surface](docs/assets/options_3d_iv_surface.png)

## Limitations And Assumptions

Free market data can contain survivorship bias, stale quotes, adjusted-price quirks, missing delistings, and delayed data. Transaction costs are simplified bps assumptions, not broker-specific fills. Options surfaces depend heavily on quote quality. Macro regimes are descriptive research tools, not causal forecasts. See `docs/limitations.md` and `docs/assumptions.md`.

## Future Improvements

- Add broker-grade corporate action handling
- Add event-driven intraday backtesting
- Add richer options no-arbitrage checks
- Add Polygon/Alpha Vantage adapters
- Add live data adapters behind a strict research/production boundary

## Disclaimer

This repository is research software for education and portfolio analytics demonstration. It is not investment advice, a trading recommendation, or a production trading system.
