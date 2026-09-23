"""
Correctness tests.

These pin the library to closed-form answers and to model-free identities
(put-call parity), so a future change that breaks pricing fails loudly.

Run with:  pytest -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import black_scholes as bs
from src import payoffs as pf
from src import pricer

PARAMS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def test_put_call_parity():
    """C - P = S0 - K e^{-rT} must hold exactly for the closed forms."""
    c = bs.bs_price(option_type="call", **PARAMS)
    p = bs.bs_price(option_type="put", **PARAMS)
    lhs = c - p
    rhs = PARAMS["S0"] - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
    assert abs(lhs - rhs) < 1e-10


def test_mc_matches_bs_within_ci():
    """Crude MC call price must bracket the closed form within its 95% CI."""
    rng = np.random.default_rng(0)
    res = pricer.price_european(n_paths=400_000, rng=rng, option_type="call",
                                **PARAMS)
    exact = bs.bs_price(option_type="call", **PARAMS)
    lo, hi = res.ci95
    assert lo <= exact <= hi


def test_control_variate_reduces_error():
    """Control variate must have a smaller standard error than crude MC."""
    rng = np.random.default_rng(1)
    crude = pricer.price_european(n_paths=100_000, rng=rng, **PARAMS)
    ctrl = pricer.price_european_control(n_paths=100_000, rng=rng, **PARAMS)
    assert ctrl.std_error < crude.std_error


def test_geometric_asian_mc_matches_closed_form():
    """Path engine: geometric Asian MC must match its closed form within CI."""
    rng = np.random.default_rng(2)
    n_steps = 100
    res = pricer.price_path_dependent(
        pf.geometric_asian_call_payoff, n_paths=400_000, n_steps=n_steps,
        rng=rng, method_name="geo", **PARAMS)
    exact = bs.geometric_asian_call(n_steps=n_steps, **PARAMS)
    lo, hi = res.ci95
    assert lo <= exact <= hi


def test_barrier_cheaper_than_vanilla():
    """A knock-out barrier call cannot be worth more than the vanilla call."""
    rng = np.random.default_rng(3)
    barrier = pricer.price_path_dependent(
        pf.down_and_out_call, n_paths=200_000, n_steps=100, rng=rng,
        method_name="barrier", barrier=85.0, **PARAMS)
    vanilla = bs.bs_price(option_type="call", **PARAMS)
    assert barrier.price < vanilla


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
