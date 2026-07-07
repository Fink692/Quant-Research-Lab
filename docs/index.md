# Quant Research Lab

Quant Research Lab is a local Python framework for institutional-style portfolio analytics, statistical arbitrage, macro regime research, options volatility analysis, and transaction-cost-aware backtesting.

The repository is designed for serious research workflows:

- Reusable Python package under `src/quant_research_lab`
- YAML configuration for reproducible runs
- Command-line scripts and installable console commands
- Local data caching and provider fallbacks
- Unit tests, linting, formatting, and type-checking hooks
- Saved reports, charts, and backtest artifacts

## Core Research Systems

| System | What It Does |
| --- | --- |
| Multi-Asset Risk Dashboard | Portfolio metrics, benchmark beta, rolling risk, drawdowns, and allocation charts |
| Pairs Trading | Cointegration scan, hedge ratio estimation, z-score signals, and cost-adjusted backtest |
| Macro Regimes | FRED feature engineering, PCA, clustering, labels, and returns by regime |
| Volatility Surface | Options cleaning, Black-Scholes IV, Greeks, smiles, term structure, and 3D surface |
| Backtesting Engine | Cash, holdings, costs, slippage, turnover, exposure, benchmark analytics, and strategy comparison |

## Repository

```bash
git clone https://github.com/Fink692/Quant-Research-Lab.git
cd Quant-Research-Lab
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest
```
