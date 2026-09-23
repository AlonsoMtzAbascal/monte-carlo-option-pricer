"""
Geometric Brownian Motion simulation under the risk-neutral measure.

Two entry points:

* `simulate_terminal`  -- samples S_T directly. This is *exact*: there is no
  time-discretisation error, because the log-normal law of S_T is known in
  closed form. Used for European options, whose payoff depends only on S_T.

* `simulate_paths`     -- samples the whole trajectory on a time grid. Needed
  for path-dependent options (Asian, barrier) whose payoff depends on the
  route, not just the destination. Each step uses the exact log-normal
  transition, so the only discretisation effect is that the path is observed
  at finitely many dates (which is what a discretely monitored contract
  actually specifies anyway).

Everything is vectorised over simulations with NumPy -- there is no Python
loop over the Monte Carlo paths.
"""
from __future__ import annotations

import numpy as np

__all__ = ["simulate_terminal", "simulate_paths"]


def _draw_normals(shape, rng: np.random.Generator, antithetic: bool):
    """Draw standard normals, optionally as antithetic pairs (Z, -Z).

    Antithetic sampling generates half the independent draws and mirrors them.
    Because a call payoff is a monotone function of Z, the payoffs of Z and -Z
    are negatively correlated, so their average has lower variance than two
    independent draws -- variance reduction at essentially zero extra cost.
    """
    if not antithetic:
        return rng.standard_normal(shape)

    n = shape[0]
    if n % 2 != 0:
        raise ValueError("antithetic sampling requires an even number of paths")
    half = (n // 2,) + tuple(shape[1:])
    z = rng.standard_normal(half)
    return np.concatenate([z, -z], axis=0)


def simulate_terminal(S0, T, r, sigma, n_paths, rng, antithetic=False):
    """Return an array of shape (n_paths,) of simulated terminal prices S_T."""
    z = _draw_normals((n_paths,), rng, antithetic)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def simulate_paths(S0, T, r, sigma, n_paths, n_steps, rng, antithetic=False):
    """Return an array of shape (n_paths, n_steps + 1) of simulated paths.

    Column 0 is S0; column j is the price at time j * T / n_steps.
    """
    dt = T / n_steps
    z = _draw_normals((n_paths, n_steps), rng, antithetic)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_increments = drift + diffusion
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)
    # Prepend the known starting value S0 as the first column.
    return np.concatenate([np.full((n_paths, 1), S0), paths], axis=1)
