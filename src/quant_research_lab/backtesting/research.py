"""Walk-forward and parameter-sweep utilities for strategy research."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import pandas as pd


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Create rolling walk-forward train/test date splits."""
    step = step_size or test_size
    splits = []
    start = 0
    while start + train_size + test_size <= len(index):
        train = index[start : start + train_size]
        test = index[start + train_size : start + train_size + test_size]
        splits.append((pd.DatetimeIndex(train), pd.DatetimeIndex(test)))
        start += step
    return splits


def parameter_grid(parameters: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of dictionaries."""
    keys = list(parameters)
    values = [list(parameters[key]) for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def run_parameter_sweep(
    runner: Callable[..., pd.Series],
    parameters: dict[str, Iterable[Any]],
) -> pd.DataFrame:
    """Run a metric-producing function across a parameter grid."""
    rows = []
    for params in parameter_grid(parameters):
        metrics = runner(**params)
        row = dict(params)
        row.update(metrics.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_tearsheet_frame(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Create a compact return comparison frame for tear sheets."""
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    aligned["active_return"] = aligned["strategy"] - aligned["benchmark"]
    aligned["strategy_cumulative"] = (1.0 + aligned["strategy"]).cumprod()
    aligned["benchmark_cumulative"] = (1.0 + aligned["benchmark"]).cumprod()
    aligned["active_cumulative"] = (
        aligned["strategy_cumulative"] / aligned["benchmark_cumulative"] - 1.0
    )
    return aligned
