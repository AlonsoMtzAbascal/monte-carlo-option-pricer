from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_vega",
    "bs_greeks",
    "geometric_asian_call",
]


def _d1_d2(S0: float, K: float, T: float, r: float, sigma: float):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S0, K, T, r, sigma, option_type: str = "call"):
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = np.exp(-r * T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif option_type == "put":
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S0, K, T, r, sigma, option_type: str = "call"):
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_vega(S0, K, T, r, sigma):
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return S0 * norm.pdf(d1) * np.sqrt(T)


def bs_greeks(S0, K, T, r, sigma, option_type: str = "call") -> dict:
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = np.exp(-r * T)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * disc * norm.cdf(d2))
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * disc * norm.cdf(-d2))
        rho = -K * T * disc * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


def geometric_asian_call(S0, K, T, r, sigma, n_steps: int)\:
    n = n_steps
    t = np.arange(1, n + 1) * (T / n)               # monitoring dates

    # ln G = (1/n) sum_i ln S_{t_i} is normal; compute its mean and variance.
    mu = np.log(S0) + (r - 0.5 * sigma**2) * t.mean()

    # Var[(1/n) sum_i sigma W_{t_i}] = (sigma^2 / n^2) * sum_i sum_j min(t_i, t_j)
    i = np.arange(1, n + 1)
    min_ij = np.minimum.outer(i, i) * (T / n)        # cov of Brownian motion
    var = (sigma**2 / n**2) * min_ij.sum()
    sd = np.sqrt(var)

    d1 = (mu - np.log(K) + var) / sd
    d2 = d1 - sd
    return np.exp(-r * T) * (np.exp(mu + 0.5 * var) * norm.cdf(d1)
                             - K * norm.cdf(d2))
