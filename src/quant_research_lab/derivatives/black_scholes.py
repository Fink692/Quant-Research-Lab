"""Black-Scholes pricing and implied volatility."""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def _validate_option_type(option_type: str) -> str:
    opt = option_type.lower()[0]
    if opt not in {"c", "p"}:
        raise ValueError("option_type must be 'call'/'c' or 'put'/'p'.")
    return opt


def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return Black-Scholes d1 and d2."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")
    if time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("time_to_expiry and volatility must be positive.")
    numerator = math.log(spot / strike) + (
        risk_free_rate - dividend_yield + 0.5 * volatility**2
    ) * time_to_expiry
    denominator = volatility * math.sqrt(time_to_expiry)
    d1 = numerator / denominator
    return d1, d1 - denominator


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "c",
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option with continuous dividend yield."""
    opt = _validate_option_type(option_type)
    if time_to_expiry <= 0:
        return max(0.0, spot - strike) if opt == "c" else max(0.0, strike - spot)
    d1, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
    discount = math.exp(-risk_free_rate * time_to_expiry)
    dividend_discount = math.exp(-dividend_yield * time_to_expiry)
    if opt == "c":
        price = spot * dividend_discount * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    else:
        price = strike * discount * norm.cdf(-d2) - spot * dividend_discount * norm.cdf(-d1)
    return float(max(price, 0.0))


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "c",
    dividend_yield: float = 0.0,
    lower: float = 1e-4,
    upper: float = 5.0,
) -> float:
    """Solve Black-Scholes implied volatility with a robust bracketing method."""
    opt = _validate_option_type(option_type)
    if market_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return float("nan")
    intrinsic = max(0.0, spot - strike) if opt == "c" else max(0.0, strike - spot)
    if market_price < intrinsic:
        return float("nan")

    def objective(volatility: float) -> float:
        return (
            black_scholes_price(
                spot,
                strike,
                time_to_expiry,
                risk_free_rate,
                volatility,
                opt,
                dividend_yield,
            )
            - market_price
        )

    try:
        return float(brentq(objective, lower, upper, maxiter=100))
    except ValueError:
        return float("nan")
