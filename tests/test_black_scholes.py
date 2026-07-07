from __future__ import annotations

import numpy as np

from quant_research_lab.derivatives.black_scholes import black_scholes_price, implied_volatility
from quant_research_lab.derivatives.greeks import delta, gamma, greeks


def test_black_scholes_known_call_price() -> None:
    price = black_scholes_price(100, 100, 1.0, 0.05, 0.20, "c")
    assert np.isclose(price, 10.4506, atol=1e-3)


def test_black_scholes_known_put_price() -> None:
    price = black_scholes_price(100, 100, 1.0, 0.05, 0.20, "p")
    assert np.isclose(price, 5.5735, atol=1e-3)


def test_implied_volatility_recovers_input() -> None:
    market_price = black_scholes_price(100, 105, 0.75, 0.03, 0.25, "c")
    iv = implied_volatility(market_price, 100, 105, 0.75, 0.03, "c")
    assert np.isclose(iv, 0.25, atol=1e-4)


def test_greeks_are_reasonable() -> None:
    values = greeks(100, 100, 1.0, 0.05, 0.20, "c")
    assert 0.60 < values["delta"] < 0.70
    assert values["gamma"] > 0
    assert np.isclose(delta(100, 100, 1.0, 0.05, 0.20, "c"), values["delta"])
    assert np.isclose(gamma(100, 100, 1.0, 0.05, 0.20), values["gamma"])
