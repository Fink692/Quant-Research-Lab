from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_lab.macro.macro_features import (
    build_macro_features,
    standardize_features,
    year_over_year,
)


def _macro_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.DataFrame(
        {
            "cpi": np.linspace(250, 280, len(index)),
            "unemployment": np.linspace(6, 4, len(index)),
            "fed_funds": np.linspace(0.25, 4.5, len(index)),
            "treasury_10y": np.linspace(1.0, 4.0, len(index)),
            "treasury_2y": np.linspace(0.5, 4.2, len(index)),
            "industrial_production": np.linspace(95, 105, len(index)),
            "baa_yield": np.linspace(4, 6, len(index)),
            "aaa_yield": np.linspace(3, 4.5, len(index)),
        },
        index=index,
    )


def test_year_over_year_change() -> None:
    series = pd.Series(
        [100.0] * 12 + [110.0], index=pd.date_range("2020-01-01", periods=13, freq="MS")
    )
    assert np.isclose(year_over_year(series).iloc[-1], 10.0)


def test_macro_feature_builder_and_standardization() -> None:
    features = build_macro_features(_macro_frame())
    assert "inflation_yoy" in features.columns
    assert "yield_curve_10y_2y" in features.columns
    standardized = standardize_features(features)
    assert np.allclose(standardized.mean().values, 0.0, atol=1e-12)
    std = standardized.std(ddof=0).values
    assert np.all((np.isclose(std, 1.0, atol=1e-12)) | (np.isclose(std, 0.0, atol=1e-12)))
