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


def european_call(S_T, K):
    return np.maximum(S_T - K, 0.0)


def european_put(S_T, K):
    return np.maximum(K - S_T, 0.0)


def arithmetic_asian_call(paths, K):
    avg = paths[:, 1:].mean(axis=1)
    return np.maximum(avg - K, 0.0)


def geometric_asian_call_payoff(paths, K):
    log_avg = np.log(paths[:, 1:]).mean(axis=1)
    geo_avg = np.exp(log_avg)
    return np.maximum(geo_avg - K, 0.0)


def down_and_out_call(paths, K, barrier):
    survived = paths.min(axis=1) > barrier
    return np.where(survived, np.maximum(paths[:, -1] - K, 0.0), 0.0)


def up_and_out_call(paths, K, barrier):
    survived = paths.max(axis=1) < barrier
    return np.where(survived, np.maximum(paths[:, -1] - K, 0.0), 0.0)
