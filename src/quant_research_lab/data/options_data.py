"""Options-chain data loading and fallback generation."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_lab.derivatives.black_scholes import black_scholes_price

LOGGER = logging.getLogger(__name__)


STANDARD_COLUMNS = {
    "contractSymbol": "contract_symbol",
    "lastTradeDate": "last_trade_date",
    "lastPrice": "last_price",
    "openInterest": "open_interest",
    "impliedVolatility": "provider_iv",
}


def normalise_options_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an options chain with consistent snake-case column names."""
    renamed = frame.rename(columns=STANDARD_COLUMNS).copy()
    renamed.columns = [str(col).strip().lower().replace(" ", "_") for col in renamed.columns]
    for column in ["expiration", "last_trade_date"]:
        if column in renamed.columns:
            renamed[column] = pd.to_datetime(renamed[column], errors="coerce")
    for column in ["strike", "bid", "ask", "last_price", "volume", "open_interest"]:
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    if "option_type" in renamed.columns:
        renamed["option_type"] = renamed["option_type"].str.lower().str[0]
    return renamed


def load_manual_options_csv(path: str | Path) -> pd.DataFrame:
    """Load a manually exported options chain CSV."""
    frame = pd.read_csv(path)
    return normalise_options_columns(frame)


def generate_synthetic_options_chain(
    spot: float = 500.0,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.015,
    valuation_date: str | None = None,
) -> pd.DataFrame:
    """Generate a realistic sample option chain for offline research runs."""
    val_date = pd.Timestamp(valuation_date or date.today().isoformat()).normalize()
    expiries = [30, 60, 90, 180, 365]
    moneyness_grid = np.linspace(0.75, 1.25, 21)
    rows: list[dict[str, object]] = []
    for days in expiries:
        expiry = val_date + pd.Timedelta(days=days)
        ttm = days / 365.0
        for option_type in ["c", "p"]:
            for mny in moneyness_grid:
                strike = round(spot * mny, 2)
                smile = 0.16 + 0.45 * (mny - 1.0) ** 2 + 0.025 * np.sqrt(ttm)
                price = black_scholes_price(
                    spot=spot,
                    strike=strike,
                    time_to_expiry=ttm,
                    risk_free_rate=risk_free_rate,
                    volatility=smile,
                    option_type=option_type,
                    dividend_yield=dividend_yield,
                )
                spread = max(0.03, price * 0.015)
                rows.append(
                    {
                        "contract_symbol": f"SYN{expiry:%Y%m%d}{option_type.upper()}{int(strike*1000):08d}",
                        "expiration": expiry,
                        "option_type": option_type,
                        "strike": strike,
                        "bid": max(0.01, price - spread / 2),
                        "ask": price + spread / 2,
                        "last_price": price,
                        "volume": int(100 + 800 * np.exp(-abs(mny - 1) * 8)),
                        "open_interest": int(250 + 2000 * np.exp(-abs(mny - 1) * 7)),
                    }
                )
    return pd.DataFrame(rows)


def load_options_chain(
    ticker: str,
    manual_csv_path: str | Path | None = None,
    max_expirations: int = 6,
    fallback_spot: float = 500.0,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.015,
    valuation_date: str | None = None,
) -> pd.DataFrame:
    """Load an options chain from CSV, yfinance, or deterministic sample data."""
    if manual_csv_path:
        path = Path(manual_csv_path)
        if path.exists():
            return load_manual_options_csv(path)
        LOGGER.warning("Manual options CSV path does not exist: %s", path)

    try:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        expirations = list(ticker_obj.options)[:max_expirations]
        rows: list[pd.DataFrame] = []
        for expiry in expirations:
            chain = ticker_obj.option_chain(expiry)
            calls = chain.calls.assign(expiration=pd.Timestamp(expiry), option_type="c")
            puts = chain.puts.assign(expiration=pd.Timestamp(expiry), option_type="p")
            rows.extend([calls, puts])
        if not rows:
            raise ValueError(f"No option expirations returned for {ticker}.")
        return normalise_options_columns(pd.concat(rows, ignore_index=True))
    except Exception as exc:
        LOGGER.warning("Using synthetic options chain because provider data failed: %s", exc)
        return generate_synthetic_options_chain(
            spot=fallback_spot,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            valuation_date=valuation_date,
        )
