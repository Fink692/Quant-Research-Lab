# Architecture

Quant Research Lab uses a package-first layout so research logic is reusable outside notebooks.

```text
src/quant_research_lab/
  data/           market, FRED, options, and cache adapters
  risk/           portfolio metrics, drawdowns, VaR, factor risk, stress testing
  strategies/     pairs trading, momentum, and signal utilities
  macro/          macro features, regime detection, regime-aware allocation
  derivatives/    Black-Scholes, Greeks, implied volatility surfaces
  portfolio/      optimization and covariance estimators
  simulation/     Monte Carlo portfolio simulation
  backtesting/    engine, portfolio state, costs, execution, constraints, workflows
  visualization/  saved figures and report helpers
  utils/          config and logging
```

## Design Principles

- **Configuration over hardcoding:** research parameters live in YAML files.
- **Local reproducibility:** data is cached and provider failures are handled gracefully.
- **No same-close lookahead:** strategy signals are shifted before execution in backtests.
- **Explicit costs:** transaction costs and slippage are included in strategy evaluation.
- **Institutional extensions:** covariance shrinkage, factor risk, risk parity, stress testing, Monte Carlo simulation, financing costs, borrow costs, and walk-forward research utilities are available as composable modules.
- **Composable modules:** risk, strategy, derivatives, and backtesting functions can be imported independently.

## Data Flow

1. Config files define assets, dates, thresholds, and cost assumptions.
2. Data modules fetch or generate reproducible input data.
3. Research modules transform data into signals, features, surfaces, or metrics.
4. Backtesting and reporting modules save results under `outputs/`.
5. Tests validate core calculations.
