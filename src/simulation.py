from __future__ import annotations

import numpy as np

__all__ = ["simulate_terminal", "simulate_paths"]


def _draw_normals(shape, rng: np.random.Generator, antithetic: bool):
    if not antithetic:
        return rng.standard_normal(shape)

    n = shape[0]
    if n % 2 != 0:
        raise ValueError("antithetic sampling requires an even number of paths")
    half = (n // 2,) + tuple(shape[1:])
    z = rng.standard_normal(half)
    return np.concatenate([z, -z], axis=0)


def simulate_terminal(S0, T, r, sigma, n_paths, rng, antithetic=False):
    z = _draw_normals((n_paths,), rng, antithetic)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def simulate_paths(S0, T, r, sigma, n_paths, n_steps, rng, antithetic=False):
    dt = T / n_steps
    z = _draw_normals((n_paths, n_steps), rng, antithetic)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_increments = drift + diffusion
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)
    # Prepend the known starting value S0 as the first column.
    return np.concatenate([np.full((n_paths, 1), S0), paths], axis=1)
