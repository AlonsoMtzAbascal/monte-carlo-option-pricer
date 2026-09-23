"""Monte Carlo option pricing engine with variance reduction.

A small, validated library for pricing options under the Black-Scholes model
by Monte Carlo simulation, with closed-form benchmarks, convergence
diagnostics, variance-reduction techniques, and path-dependent payoffs.
"""
from . import black_scholes, greeks, payoffs, pricer, simulation

__all__ = ["black_scholes", "simulation", "payoffs", "pricer", "greeks"]
__version__ = "0.1.0"
