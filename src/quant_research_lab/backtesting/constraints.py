"""Portfolio and exposure constraints for target weights."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PositionConstraints:
    """Common portfolio target-weight constraints."""

    max_gross_exposure: float = 1.0
    max_single_position: float = 1.0
    long_only: bool = False
    sector_limits: dict[str, float] | None = None
    sector_map: dict[str, str] | None = None


def apply_position_constraints(
    weights: pd.Series,
    constraints: PositionConstraints,
) -> pd.Series:
    """Apply simple institutional target-weight constraints."""
    constrained = weights.fillna(0.0).astype(float).copy()
    if constraints.long_only:
        constrained = constrained.clip(lower=0.0)
    constrained = constrained.clip(
        lower=-constraints.max_single_position,
        upper=constraints.max_single_position,
    )

    if constraints.sector_limits and constraints.sector_map:
        for sector, limit in constraints.sector_limits.items():
            members = [
                ticker for ticker, mapped in constraints.sector_map.items() if mapped == sector
            ]
            sector_weight = constrained.reindex(members).abs().sum()
            if sector_weight > limit and sector_weight > 0:
                constrained.loc[members] = constrained.reindex(members).fillna(0.0) * (
                    limit / sector_weight
                )

    gross = constrained.abs().sum()
    if gross > constraints.max_gross_exposure and gross > 0:
        constrained *= constraints.max_gross_exposure / gross
    return constrained
