from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.strategies.pairs_trading import (
    backtest_pair_strategy,
    build_spread,
    estimate_hedge_ratio,
    generate_pair_weights,
)


def _synthetic_pair() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2023-01-01", periods=350)
    x = 100 + np.cumsum(rng.normal(0, 1, len(index)))
    noise = rng.normal(0, 0.5, len(index))
    y = 5 + 1.4 * x + noise
    return pd.DataFrame({"Y": y, "X": x}, index=index)


def test_hedge_ratio_estimation() -> None:
    prices = _synthetic_pair()
    hedge_ratio, intercept = estimate_hedge_ratio(prices["Y"], prices["X"])
    assert np.isclose(hedge_ratio, 1.4, atol=0.05)
    spread = build_spread(prices["Y"], prices["X"], hedge_ratio, intercept)
    assert abs(spread.mean()) < 1e-8


def test_pair_weights_are_dollar_normalized() -> None:
    prices = _synthetic_pair()
    hedge_ratio, _ = estimate_hedge_ratio(prices["Y"], prices["X"])
    weights = generate_pair_weights(prices, "Y", "X", hedge_ratio, lookback=30, entry_z=1.0, exit_z=0.1)
    assert set(weights.columns) == {"Y", "X"}
    assert weights.abs().sum(axis=1).max() <= 1.0000001


def test_pair_backtest_outputs_expected_columns() -> None:
    prices = _synthetic_pair()
    hedge_ratio, intercept = estimate_hedge_ratio(prices["Y"], prices["X"])
    result = backtest_pair_strategy(prices, "Y", "X", hedge_ratio, intercept, lookback=30, entry_z=1.0, exit_z=0.1)
    assert {"zscore", "signal", "strategy_returns", "equity", "drawdown"}.issubset(result.columns)
    assert result["equity"].iloc[-1] > 0
