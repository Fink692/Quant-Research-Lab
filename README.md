# Quant Research Lab

A local Python package for portfolio risk, pairs research, macro regimes, options volatility, and transaction-cost-aware backtesting.

This is a separate project from [Quant Systems Lab](https://github.com/Fink692/quant-systems-lab), which focuses on order-book replay and broader model experiments.

## Run locally

Requires Python 3.11 or newer. Create and activate a virtual environment, then:

```sh
python -m pip install -e ".[dev]"
pytest
qrl-risk
```

| Command | Workflow |
| --- | --- |
| `qrl-risk` | Portfolio returns, drawdown, risk, and benchmark comparison |
| `qrl-pairs` | Cointegration screening, spread signals, and costs |
| `qrl-macro` | Macro features, PCA, regime clustering, and allocation diagnostics |
| `qrl-vol` | Options-chain cleaning, implied volatility, Greeks, and surfaces |
| `qrl-backtest` | Cash, holdings, fills, slippage, turnover, constraints, and performance |

Equivalent scripts are in `scripts/`. Outputs are saved under `outputs/figures/`, `outputs/reports/`, and `outputs/backtests/`.

## Data

Market data uses Yahoo Finance through `yfinance`; macro data uses FRED through a public CSV endpoint or an optional API key. Manual options-chain CSV inputs are supported. Copy `.env.example` to `.env` for optional configuration.

Provider failures can use deterministic sample data. Check the run's logs and source before interpreting an output as empirical market evidence. Adjusted history and incomplete quote coverage impose limits on backtests.

## Package structure

| Path | Purpose |
| --- | --- |
| `src/quant_research_lab/` | Data adapters, portfolio models, risk, macro, derivatives, and backtesting |
| `configs/` | Asset lists and workflow parameters |
| `scripts/` | Command-line wrappers |
| `notebooks/` | Research walkthroughs |
| `tests/` | Regression checks |
| `docs/` | Methodology, assumptions, examples, and API reference |

Risk modules include covariance shrinkage, EWMA, factor models, tail-risk estimates, and stress tests. Backtests support delayed signals, transaction costs, rebalance calendars, position constraints, borrow costs, and cash financing.

## Development

```sh
ruff check src tests scripts
black --check src tests scripts
mypy src --ignore-missing-imports
pytest
```

[Getting started](docs/getting_started.md) · [Architecture](docs/architecture.md) · [Methodology](docs/methodology.md) · [Assumptions](docs/assumptions.md) · [Limitations](docs/limitations.md) · [Examples](docs/examples.md)

MIT licensed. Research outputs are not live performance or investment recommendations.
