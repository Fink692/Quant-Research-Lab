"""Black-Scholes Greek sensitivities."""

from __future__ import annotations

import math

from scipy.stats import norm

from quant_research_lab.derivatives.black_scholes import _validate_option_type, d1_d2


def delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "c",
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes delta."""
    opt = _validate_option_type(option_type)
    d1, _ = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
    discount = math.exp(-dividend_yield * time_to_expiry)
    return float(discount * norm.cdf(d1) if opt == "c" else discount * (norm.cdf(d1) - 1.0))


def gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes gamma."""
    d1, _ = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
    return float(
        math.exp(-dividend_yield * time_to_expiry)
        * norm.pdf(d1)
        / (spot * volatility * math.sqrt(time_to_expiry))
    )


def vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes vega for a 1.00 volatility point change."""
    d1, _ = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
    return float(
        spot * math.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * math.sqrt(time_to_expiry)
    )


def theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "c",
    dividend_yield: float = 0.0,
) -> float:
    """Calculate annual Black-Scholes theta."""
    opt = _validate_option_type(option_type)
    d1, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
    first = -(
        spot
        * math.exp(-dividend_yield * time_to_expiry)
        * norm.pdf(d1)
        * volatility
        / (2.0 * math.sqrt(time_to_expiry))
    )
    if opt == "c":
        second = dividend_yield * spot * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        third = -risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        second = -dividend_yield * spot * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        third = risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    return float(first + second + third)


def rho(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "c",
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes rho for a 1.00 interest-rate change."""
    opt = _validate_option_type(option_type)
    _, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
    if opt == "c":
        value = strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        value = (
            -strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        )
    return float(value)


def greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "c",
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Return all primary Black-Scholes Greeks."""
    return {
        "delta": delta(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
        ),
        "gamma": gamma(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield),
        "theta": theta(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
        ),
        "vega": vega(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield),
        "rho": rho(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
        ),
    }
