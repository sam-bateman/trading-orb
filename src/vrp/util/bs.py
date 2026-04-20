"""Black-Scholes option pricing and Greeks.

European options on a non-dividend-paying underlying, risk-free rate
optionally zero (the default for the Strategy B backtest, since we're
analyzing a premium-collection strategy where the risk-free drift is
small relative to IV).

References:
- Black, Scholes (1973) "The Pricing of Options and Corporate Liabilities"
- Hull (2017) "Options, Futures, and Other Derivatives" (ch. 15, 17)
"""
from __future__ import annotations

import math

from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T ({T}) and sigma ({sigma}) must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.0,
             option_type: str = "call") -> float:
    """European Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0,
             option_type: str = "call") -> float:
    """European Black-Scholes delta."""
    d1, _ = _d1_d2(S, K, T, sigma, r)
    if option_type == "call":
        return float(norm.cdf(d1))
    if option_type == "put":
        return float(norm.cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def strike_from_delta(S: float, T: float, sigma: float, r: float,
                      option_type: str, target_delta: float) -> float:
    """Invert the delta formula: solve K such that delta(K) == target_delta."""
    if not (0.0 < abs(target_delta) < 1.0):
        raise ValueError(f"|target_delta| must be in (0, 1), got {target_delta}")
    if option_type == "put":
        if target_delta >= 0:
            raise ValueError("put target_delta must be negative")
        nd1 = norm.ppf(1.0 + target_delta)
    elif option_type == "call":
        if target_delta <= 0:
            raise ValueError("call target_delta must be positive")
        nd1 = norm.ppf(target_delta)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    K = S * math.exp(-sigma * math.sqrt(T) * nd1 + (r + 0.5 * sigma * sigma) * T)
    return float(K)
