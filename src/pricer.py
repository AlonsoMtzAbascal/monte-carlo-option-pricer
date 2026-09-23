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
    price: float          # discounted mean payoff
    std_error: float      # standard error of that mean
    n_paths: int
    method: str

    @property
    def ci95(self):
        half = 1.96 * self.std_error
        return (self.price - half, self.price + half)

    def __str__(self):
        lo, hi = self.ci95
        return (f"{self.method:<28s} price = {self.price:8.4f}  "
                f"SE = {self.std_error:7.4f}  "
                f"95% CI = [{lo:8.4f}, {hi:8.4f}]  (n={self.n_paths:,})")


def _summarise(discounted_payoffs: np.ndarray, method: str) -> MCResult:
    n = discounted_payoffs.size
    price = discounted_payoffs.mean()
    # SE of the sample mean = sample std / sqrt(n). This is the 1/sqrt(n) law.
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n)
    return MCResult(price=price, std_error=std_error, n_paths=n, method=method)

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


def price_path_dependent(payoff_fn, S0, K, T, r, sigma, n_paths, n_steps, rng,
                         antithetic=False, method_name="Path-dependent MC",
                         **payoff_kwargs) -> MCResult:
    paths = simulate_paths(S0, T, r, sigma, n_paths, n_steps, rng,
                           antithetic=antithetic)
    payoff = payoff_fn(paths, K, **payoff_kwargs)
    discounted = np.exp(-r * T) * payoff
    return _summarise(discounted, method_name)


def price_arithmetic_asian_control(S0, K, T, r, sigma, n_paths, n_steps, rng) -> MCResult:
    paths = simulate_paths(S0, T, r, sigma, n_paths, n_steps, rng)

    arith = np.exp(-r * T) * pf.arithmetic_asian_call(paths, K)
    geo = np.exp(-r * T) * pf.geometric_asian_call_payoff(paths, K)
    expected_geo = bs.geometric_asian_call(S0, K, T, r, sigma, n_steps)

    cov = np.cov(arith, geo, ddof=1)
    c_star = cov[0, 1] / cov[1, 1]
    adjusted = arith - c_star * (geo - expected_geo)
    return _summarise(adjusted, "Arithmetic Asian MC (control)")
