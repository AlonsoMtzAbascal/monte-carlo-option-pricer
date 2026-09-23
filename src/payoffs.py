"""
Payoff functions.

Each function maps simulated prices to the cash the option holder receives at
maturity (undiscounted). European payoffs act on the terminal price only;
path-dependent payoffs act on the full simulated path.

Separating payoffs from the simulation engine is deliberate: the same set of
simulated paths can be re-priced under any payoff, and new products can be
added here without touching the simulator or the pricer.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "european_call",
    "european_put",
    "arithmetic_asian_call",
    "geometric_asian_call_payoff",
    "down_and_out_call",
    "up_and_out_call",
]


# --- European (depend on terminal price only) --------------------------------

def european_call(S_T, K):
    return np.maximum(S_T - K, 0.0)


def european_put(S_T, K):
    return np.maximum(K - S_T, 0.0)


# --- Asian (depend on the average price along the path) ----------------------

def arithmetic_asian_call(paths, K):
    """Average taken over monitoring dates (excludes the t=0 starting price).

    The arithmetic average of log-normals is not log-normal, so this contract
    has no closed-form price -- it is exactly the case where Monte Carlo earns
    its keep.
    """
    avg = paths[:, 1:].mean(axis=1)
    return np.maximum(avg - K, 0.0)


def geometric_asian_call_payoff(paths, K):
    """Geometric-average Asian call payoff (has a closed form; see black_scholes)."""
    log_avg = np.log(paths[:, 1:]).mean(axis=1)
    geo_avg = np.exp(log_avg)
    return np.maximum(geo_avg - K, 0.0)


# --- Barrier (knocked out if the path crosses a level) -----------------------

def down_and_out_call(paths, K, barrier):
    """Call that pays off only if the price never falls to/through `barrier`."""
    survived = paths.min(axis=1) > barrier
    return np.where(survived, np.maximum(paths[:, -1] - K, 0.0), 0.0)


def up_and_out_call(paths, K, barrier):
    """Call that pays off only if the price never rises to/through `barrier`."""
    survived = paths.max(axis=1) < barrier
    return np.where(survived, np.maximum(paths[:, -1] - K, 0.0), 0.0)
