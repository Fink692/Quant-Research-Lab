"""Options-chain cleaning and implied volatility surface construction."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from quant_research_lab.derivatives.black_scholes import black_scholes_price, implied_volatility
from quant_research_lab.derivatives.greeks import greeks


def clean_options_chain(
    options: pd.DataFrame,
    valuation_date: str | None = None,
    min_days_to_expiry: int = 7,
    max_days_to_expiry: int = 730,
    min_open_interest: int = 10,
    min_volume: int = 1,
) -> pd.DataFrame:
    """Remove stale, expired, illiquid, and bad-quote option contracts."""
    frame = options.copy()
    valuation = pd.Timestamp(valuation_date or date.today().isoformat()).normalize()
    frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce")
    frame["days_to_expiry"] = (frame["expiration"] - valuation).dt.days
    frame["mid"] = (pd.to_numeric(frame["bid"], errors="coerce") + pd.to_numeric(frame["ask"], errors="coerce")) / 2
    frame["option_type"] = frame["option_type"].str.lower().str[0]
    filters = (
        frame["expiration"].notna()
        & frame["strike"].gt(0)
        & frame["bid"].gt(0)
        & frame["ask"].gt(frame["bid"])
        & frame["mid"].gt(0)
        & frame["days_to_expiry"].between(min_days_to_expiry, max_days_to_expiry)
        & frame.get("open_interest", pd.Series(0, index=frame.index)).fillna(0).ge(min_open_interest)
        & frame.get("volume", pd.Series(0, index=frame.index)).fillna(0).ge(min_volume)
        & frame["option_type"].isin(["c", "p"])
    )
    return frame.loc[filters].sort_values(["expiration", "option_type", "strike"]).reset_index(drop=True)


def enrich_options_chain(
    options: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    valuation_date: str | None = None,
) -> pd.DataFrame:
    """Add moneyness, implied volatility, theoretical price, and Greeks."""
    valuation = pd.Timestamp(valuation_date or date.today().isoformat()).normalize()
    frame = options.copy()
    frame["time_to_expiry"] = (pd.to_datetime(frame["expiration"]) - valuation).dt.days / 365.0
    frame["moneyness"] = frame["strike"] / spot

    ivs: list[float] = []
    theoretical: list[float] = []
    greek_rows: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        iv = implied_volatility(
            market_price=float(row.mid),
            spot=spot,
            strike=float(row.strike),
            time_to_expiry=float(row.time_to_expiry),
            risk_free_rate=risk_free_rate,
            option_type=str(row.option_type),
            dividend_yield=dividend_yield,
        )
        ivs.append(iv)
        vol_for_model = iv if np.isfinite(iv) and iv > 0 else 0.2
        theoretical.append(
            black_scholes_price(
                spot=spot,
                strike=float(row.strike),
                time_to_expiry=float(row.time_to_expiry),
                risk_free_rate=risk_free_rate,
                volatility=vol_for_model,
                option_type=str(row.option_type),
                dividend_yield=dividend_yield,
            )
        )
        greek_rows.append(
            greeks(
                spot=spot,
                strike=float(row.strike),
                time_to_expiry=float(row.time_to_expiry),
                risk_free_rate=risk_free_rate,
                volatility=vol_for_model,
                option_type=str(row.option_type),
                dividend_yield=dividend_yield,
            )
        )
    frame["implied_volatility"] = ivs
    frame["theoretical_price"] = theoretical
    greeks_frame = pd.DataFrame(greek_rows)
    return pd.concat([frame.reset_index(drop=True), greeks_frame], axis=1).dropna(subset=["implied_volatility"])


def iv_smile_by_expiration(options: pd.DataFrame) -> pd.DataFrame:
    """Return IV smile points keyed by expiration and moneyness."""
    return options[["expiration", "option_type", "moneyness", "implied_volatility"]].sort_values(
        ["expiration", "option_type", "moneyness"]
    )


def iv_term_structure(options: pd.DataFrame) -> pd.DataFrame:
    """Calculate ATM implied volatility term structure by expiration."""
    frame = options.assign(atm_distance=(options["moneyness"] - 1.0).abs())
    atm = frame.sort_values("atm_distance").groupby(["expiration", "option_type"], as_index=False).head(5)
    return (
        atm.groupby(["expiration", "option_type"], as_index=False)["implied_volatility"]
        .mean()
        .sort_values(["expiration", "option_type"])
    )


def build_surface_grid(
    options: pd.DataFrame,
    option_type: str = "c",
    grid_size: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate implied volatility over moneyness and time-to-expiry."""
    opt = option_type.lower()[0]
    frame = options.loc[options["option_type"].str.lower().str[0] == opt].dropna(
        subset=["moneyness", "time_to_expiry", "implied_volatility"]
    )
    if len(frame) < 4:
        raise ValueError("Need at least four options to interpolate a volatility surface.")
    x = frame["moneyness"].to_numpy()
    y = frame["time_to_expiry"].to_numpy()
    z = frame["implied_volatility"].to_numpy()
    x_grid = np.linspace(np.nanpercentile(x, 5), np.nanpercentile(x, 95), grid_size)
    y_grid = np.linspace(np.nanpercentile(y, 5), np.nanpercentile(y, 95), grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = griddata((x, y), z, (xx, yy), method="linear")
    if np.isnan(zz).any():
        nearest = griddata((x, y), z, (xx, yy), method="nearest")
        zz = np.where(np.isnan(zz), nearest, zz)
    return xx, yy, zz
