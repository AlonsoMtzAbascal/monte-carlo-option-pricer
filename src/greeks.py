from __future__ import annotations

import numpy as np

__all__ = ["pathwise_delta", "pathwise_vega", "fd_delta_crn"]


def _terminal_from_normals(S0, T, r, sigma, z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def pathwise_delta(S0, K, T, r, sigma, n_paths, rng):
    z = rng.standard_normal(n_paths)
    S_T = _terminal_from_normals(S0, T, r, sigma, z)
    indicator = (S_T > K).astype(float)
    estimator = np.exp(-r * T) * indicator * S_T / S0
    return estimator.mean(), estimator.std(ddof=1) / np.sqrt(n_paths)


def pathwise_vega(S0, K, T, r, sigma, n_paths, rng):
    z = rng.standard_normal(n_paths)
    S_T = _terminal_from_normals(S0, T, r, sigma, z)
    indicator = (S_T > K).astype(float)
    dST_dsigma = S_T * (np.log(S_T / S0) - (r + 0.5 * sigma**2) * T) / sigma
    estimator = np.exp(-r * T) * indicator * dST_dsigma
    return estimator.mean(), estimator.std(ddof=1) / np.sqrt(n_paths)


def fd_delta_crn(S0, K, T, r, sigma, n_paths, rng, h=1e-2):
    z = rng.standard_normal(n_paths)
    disc = np.exp(-r * T)

    S_up = _terminal_from_normals(S0 + h, T, r, sigma, z)
    S_dn = _terminal_from_normals(S0 - h, T, r, sigma, z)
    price_up = disc * np.maximum(S_up - K, 0.0)
    price_dn = disc * np.maximum(S_dn - K, 0.0)

    diff = (price_up - price_dn) / (2 * h)
    return diff.mean(), diff.std(ddof=1) / np.sqrt(n_paths)
