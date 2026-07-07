"""FRED data access with direct public CSV fallback."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd
import requests

from quant_research_lab.data.cache import DataCache

LOGGER = logging.getLogger(__name__)


def generate_synthetic_macro(start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """Create deterministic macro data when FRED is unavailable."""
    index = pd.date_range(start=start, end=end or date.today().isoformat(), freq="MS")
    rng = np.random.default_rng(7)
    cycle = np.sin(np.linspace(0, 6 * np.pi, len(index)))
    inflation = 2.2 + 1.1 * cycle + rng.normal(0, 0.2, len(index))
    unemployment = 5.2 - 0.9 * cycle + rng.normal(0, 0.25, len(index))
    ten_year = 3.0 + 0.8 * cycle + rng.normal(0, 0.2, len(index))
    two_year = 2.4 + 1.0 * cycle + rng.normal(0, 0.25, len(index))
    industrial = 100 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.003, len(index))))
    return pd.DataFrame(
        {
            "cpi": 260 * np.exp(np.cumsum(inflation / 1200)),
            "unemployment": unemployment,
            "fed_funds": np.maximum(0.05, two_year - 0.4),
            "treasury_10y": ten_year,
            "treasury_2y": two_year,
            "industrial_production": industrial,
            "baa_yield": ten_year + 2.0 + rng.normal(0, 0.08, len(index)),
            "aaa_yield": ten_year + 0.7 + rng.normal(0, 0.05, len(index)),
        },
        index=index,
    ).round(4)


class FredClient:
    """Fetch FRED series by API key, public CSV endpoint, or synthetic fallback."""

    def __init__(self, cache: DataCache | None = None, api_key: str | None = None) -> None:
        self.cache = cache or DataCache()
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def fetch_series(
        self,
        series_id: str,
        start: str,
        end: str | None = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """Fetch one FRED series as a numeric time series."""
        key = DataCache.make_key("fred", series_id, start, end or "latest")
        if use_cache:
            cached = self.cache.get_frame(key)
            if cached is not None and not cached.empty:
                return cached.iloc[:, 0]

        try:
            if self.api_key:
                from fredapi import Fred

                fred = Fred(api_key=self.api_key)
                series = fred.get_series(series_id, observation_start=start, observation_end=end)
                series.name = series_id
            else:
                url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
                response = requests.get(url, params={"id": series_id}, timeout=20)
                response.raise_for_status()
                from io import StringIO

                frame = pd.read_csv(StringIO(response.text))
                frame["observation_date"] = pd.to_datetime(frame["observation_date"])
                series = frame.set_index("observation_date")[series_id]
                series = series.loc[pd.Timestamp(start) :]
                if end:
                    series = series.loc[: pd.Timestamp(end)]
            series = pd.to_numeric(series.replace(".", np.nan), errors="coerce").dropna()
            if series.empty:
                raise ValueError(f"FRED series {series_id} returned no usable observations.")
            self.cache.set_frame(key, series.to_frame(series_id))
            return series
        except Exception as exc:
            LOGGER.warning("FRED series %s unavailable: %s", series_id, exc)
            fallback = generate_synthetic_macro(start=start, end=end)
            column = fallback.columns[0]
            return fallback[column].rename(series_id)

    def fetch_dataset(
        self,
        series_map: Mapping[str, str],
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch and align a mapping of named FRED series."""
        frames: list[pd.Series] = []
        for name, series_id in series_map.items():
            series = self.fetch_series(series_id=series_id, start=start, end=end)
            frames.append(series.rename(name))
        data = pd.concat(frames, axis=1).sort_index()
        data = data.resample("MS").last().ffill(limit=3)
        if data.dropna(how="all").empty:
            LOGGER.warning("FRED dataset empty; using deterministic sample macro data.")
            return generate_synthetic_macro(start=start, end=end)
        return data.dropna(how="all")
