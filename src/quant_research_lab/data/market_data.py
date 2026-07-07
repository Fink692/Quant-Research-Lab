"""Market data download and return-calculation utilities."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_lab.data.cache import DataCache

LOGGER = logging.getLogger(__name__)


def _normalise_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        return [tickers]
    return list(dict.fromkeys(tickers))


def generate_synthetic_prices(
    tickers: str | Iterable[str],
    start: str = "2020-01-01",
    end: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic sample prices for offline demos and tests.

    The fallback is deliberately deterministic so reports are reproducible when
    external providers are unavailable.
    """
    tickers_list = _normalise_tickers(tickers)
    end_date = pd.Timestamp(end or date.today().isoformat())
    index = pd.bdate_range(start=start, end=end_date)
    if len(index) == 0:
        raise ValueError("Synthetic price date range is empty.")

    rng = np.random.default_rng(seed)
    market_shock = rng.normal(0.00025, 0.009, size=len(index))
    prices: dict[str, pd.Series] = {}
    for idx, ticker in enumerate(tickers_list):
        drift = 0.00015 + idx * 0.000015
        vol = 0.010 + (idx % 5) * 0.0025
        idio = rng.normal(drift, vol, size=len(index))
        returns = 0.55 * market_shock + 0.45 * idio
        start_price = 75 + 15 * idx
        prices[ticker] = pd.Series(start_price * np.exp(np.cumsum(returns)), index=index)
    return pd.DataFrame(prices).round(4)


def download_prices(
    tickers: str | Iterable[str],
    start: str,
    end: str | None = None,
    cache: DataCache | None = None,
    use_cache: bool = True,
    fallback: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance with local caching.

    If yfinance is not installed, the network is unavailable, or Yahoo returns no
    usable data, the function returns deterministic sample data when
    ``fallback=True``.
    """
    tickers_list = _normalise_tickers(tickers)
    cache = cache or DataCache()
    key = DataCache.make_key("yfinance", ",".join(tickers_list), start, end or "latest")
    if use_cache:
        cached = cache.get_frame(key)
        if cached is not None and not cached.empty:
            return cached.sort_index()

    try:
        import yfinance as yf

        raw = yf.download(
            tickers=tickers_list,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            raise ValueError("Yahoo Finance returned an empty DataFrame.")
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            elif "Adj Close" in raw.columns.get_level_values(0):
                prices = raw["Adj Close"]
            else:
                prices = raw.xs(raw.columns.levels[0][0], axis=1, level=0)
        else:
            close_col = "Close" if "Close" in raw else "Adj Close"
            prices = raw[[close_col]].rename(columns={close_col: tickers_list[0]})
        prices = prices.loc[:, [col for col in tickers_list if col in prices.columns]]
        prices = prices.dropna(how="all").ffill()
        if prices.empty:
            raise ValueError("Downloaded prices were empty after cleaning.")
        cache.set_frame(key, prices)
        return prices
    except Exception as exc:
        if not fallback:
            raise
        LOGGER.warning("Using deterministic sample prices because market data failed: %s", exc)
        prices = generate_synthetic_prices(tickers_list, start=start, end=end)
        cache.set_frame(key, prices)
        return prices


def calculate_returns(
    prices: pd.DataFrame | pd.Series, log: bool = False
) -> pd.DataFrame | pd.Series:
    """Calculate simple or log returns from a price series/DataFrame."""
    returns = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def load_csv_prices(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Load prices from a CSV file with a date column or date index."""
    frame = pd.read_csv(path)
    if date_column in frame.columns:
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame = frame.set_index(date_column)
    else:
        frame.index = pd.to_datetime(frame.iloc[:, 0])
        frame = frame.iloc[:, 1:]
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")
