"""
The Monte Carlo pricing engine.

Every estimate is returned as an `MCResult`, which carries not just the price
but its Monte Carlo standard error and a 95% confidence interval. Reporting a
simulated price without an error bar is the single most common mistake in a
Monte Carlo project: the whole point of the method is that the answer is a
random estimate, and its uncertainty is quantifiable and must be quoted.

Estimators provided
-------------------
* `price_european`          -- plain (crude) Monte Carlo.
* `price_european_control`  -- control-variate estimator using the underlying
                               S_T (whose risk-neutral expectation is known
                               exactly) to cancel variance.
* `price_path_dependent`    -- crude estimator for any path payoff.
* `price_arithmetic_asian_control` -- control-variate estimator for the
                               arithmetic Asian using the geometric Asian
                               (which has a closed form) as the control.

Antithetic variates are handled inside the simulator and switched on with the
`antithetic=True` flag on the European/path estimators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs
from . import payoffs as pf
from .simulation import simulate_paths, simulate_terminal

__all__ = [
    "MCResult",
    "price_european",
    "price_european_control",
    "price_path_dependent",
    "price_arithmetic_asian_control",
]


@dataclass
class MCResult:
    """Result of a Monte Carlo pricing run."""
    price: float          # discounted mean payoff
    std_error: float      # standard error of that mean
    n_paths: int
    method: str

    @property
    def ci95(self):
        """95% confidence interval for the true price."""
        half = 1.96 * self.std_error
        return (self.price - half, self.price + half)

    def __str__(self):
        lo, hi = self.ci95
        return (f"{self.method:<28s} price = {self.price:8.4f}  "
                f"SE = {self.std_error:7.4f}  "
                f"95% CI = [{lo:8.4f}, {hi:8.4f}]  (n={self.n_paths:,})")


def _summarise(discounted_payoffs: np.ndarray, method: str) -> MCResult:
    """Turn a vector of discounted payoffs into price + standard error."""
    n = discounted_payoffs.size
    price = discounted_payoffs.mean()
    # SE of the sample mean = sample std / sqrt(n). This is the 1/sqrt(n) law.
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n)
    return MCResult(price=price, std_error=std_error, n_paths=n, method=method)


# --- European -----------------------------------------------------------------

def price_european(S0, K, T, r, sigma, n_paths, rng,
                   option_type="call", antithetic=False) -> MCResult:
    """Crude (or antithetic) Monte Carlo price of a European option."""
    S_T = simulate_terminal(S0, T, r, sigma, n_paths, rng, antithetic=antithetic)
    payoff = pf.european_call(S_T, K) if option_type == "call" \
        else pf.european_put(S_T, K)
    discounted = np.exp(-r * T) * payoff
    method = "European MC (antithetic)" if antithetic else "European MC (crude)"
    return _summarise(discounted, method)


def price_european_control(S0, K, T, r, sigma, n_paths, rng,
                           option_type="call") -> MCResult:
    """Control-variate estimator for a European option.

    Control:  X = S_T, with known E[S_T] = S0 * exp(r T) under the risk-neutral
    measure. The estimator prices Y - c (X - E[X]); the optimal c minimises
    variance and is estimated from the sample covariance. Because X is strongly
    correlated with the call payoff, most of the payoff's variance is removed.
    """
    S_T = simulate_terminal(S0, T, r, sigma, n_paths, rng)
    payoff = pf.european_call(S_T, K) if option_type == "call" \
        else pf.european_put(S_T, K)
    discounted = np.exp(-r * T) * payoff

    control = S_T
    expected_control = S0 * np.exp(r * T)

    cov = np.cov(discounted, control, ddof=1)
    c_star = cov[0, 1] / cov[1, 1]                    # variance-minimising coeff
    adjusted = discounted - c_star * (control - expected_control)
    return _summarise(adjusted, "European MC (control variate)")


# --- Path-dependent -----------------------------------------------------------

def price_path_dependent(payoff_fn, S0, K, T, r, sigma, n_paths, n_steps, rng,
                         antithetic=False, method_name="Path-dependent MC",
                         **payoff_kwargs) -> MCResult:
    """Crude (or antithetic) Monte Carlo for any payoff acting on full paths.

    `payoff_fn(paths, K, **payoff_kwargs)` returns undiscounted payoffs.
    """
    paths = simulate_paths(S0, T, r, sigma, n_paths, n_steps, rng,
                           antithetic=antithetic)
    payoff = payoff_fn(paths, K, **payoff_kwargs)
    discounted = np.exp(-r * T) * payoff
    return _summarise(discounted, method_name)


def price_arithmetic_asian_control(S0, K, T, r, sigma, n_paths, n_steps, rng) -> MCResult:
    """Control-variate estimator for the arithmetic Asian call.

    The arithmetic Asian has no closed form, but the geometric Asian does, and
    the two payoffs are almost perfectly correlated (both are averages of the
    same path). Using the geometric Asian as a control variate -- the
    Kemna-Vorst technique -- removes the vast majority of the variance. This is
    the headline variance-reduction result of the path-dependent section.
    """
    paths = simulate_paths(S0, T, r, sigma, n_paths, n_steps, rng)

    arith = np.exp(-r * T) * pf.arithmetic_asian_call(paths, K)
    geo = np.exp(-r * T) * pf.geometric_asian_call_payoff(paths, K)
    expected_geo = bs.geometric_asian_call(S0, K, T, r, sigma, n_steps)

    cov = np.cov(arith, geo, ddof=1)
    c_star = cov[0, 1] / cov[1, 1]
    adjusted = arith - c_star * (geo - expected_geo)
    return _summarise(adjusted, "Arithmetic Asian MC (control)")
