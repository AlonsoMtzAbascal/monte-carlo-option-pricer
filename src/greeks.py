"""
Monte Carlo Greeks (sensitivities) for a European call.

Two independent methods, both validated against the closed-form Greeks in
`black_scholes.py`:

* Pathwise differentiation -- differentiate the payoff with respect to the
  parameter *inside* the expectation. Low variance and unbiased for the call,
  whose payoff is (almost everywhere) differentiable in S0 and sigma.

* Finite differences with common random numbers (CRN) -- bump the parameter,
  re-simulate with the *same* random draws, and difference. Reusing the draws
  is essential: it cancels the simulation noise that would otherwise swamp a
  small bump, turning a hopeless estimator into an accurate one.

Delta = dPrice/dS0,  Vega = dPrice/dsigma.
"""
from __future__ import annotations

import numpy as np

__all__ = ["pathwise_delta", "pathwise_vega", "fd_delta_crn"]


def _terminal_from_normals(S0, T, r, sigma, z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def pathwise_delta(S0, K, T, r, sigma, n_paths, rng):
    """Pathwise estimator of Delta for a European call.

    Delta = e^{-rT} E[ 1{S_T > K} * dS_T/dS0 ] = e^{-rT} E[ 1{S_T > K} * S_T/S0 ].
    """
    z = rng.standard_normal(n_paths)
    S_T = _terminal_from_normals(S0, T, r, sigma, z)
    indicator = (S_T > K).astype(float)
    estimator = np.exp(-r * T) * indicator * S_T / S0
    return estimator.mean(), estimator.std(ddof=1) / np.sqrt(n_paths)


def pathwise_vega(S0, K, T, r, sigma, n_paths, rng):
    """Pathwise estimator of Vega for a European call.

    dS_T/dsigma = S_T * (ln(S_T/S0) - (r + 0.5 sigma^2) T) / sigma, so
    Vega = e^{-rT} E[ 1{S_T > K} * dS_T/dsigma ].
    """
    z = rng.standard_normal(n_paths)
    S_T = _terminal_from_normals(S0, T, r, sigma, z)
    indicator = (S_T > K).astype(float)
    dST_dsigma = S_T * (np.log(S_T / S0) - (r + 0.5 * sigma**2) * T) / sigma
    estimator = np.exp(-r * T) * indicator * dST_dsigma
    return estimator.mean(), estimator.std(ddof=1) / np.sqrt(n_paths)


def fd_delta_crn(S0, K, T, r, sigma, n_paths, rng, h=1e-2):
    """Central finite-difference Delta using common random numbers.

    The SAME normal draws price both the up-bumped and down-bumped spot, so the
    Monte Carlo noise is shared and cancels in the difference.
    """
    z = rng.standard_normal(n_paths)
    disc = np.exp(-r * T)

    S_up = _terminal_from_normals(S0 + h, T, r, sigma, z)
    S_dn = _terminal_from_normals(S0 - h, T, r, sigma, z)
    price_up = disc * np.maximum(S_up - K, 0.0)
    price_dn = disc * np.maximum(S_dn - K, 0.0)

    diff = (price_up - price_dn) / (2 * h)
    return diff.mean(), diff.std(ddof=1) / np.sqrt(n_paths)
