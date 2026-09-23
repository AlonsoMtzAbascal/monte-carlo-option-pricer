"""
Experiment 1 -- Validation against closed form.

Question answered: does the Monte Carlo price agree with Black-Scholes where
Black-Scholes is exact?

We price European calls across a range of strikes with crude Monte Carlo and
overlay them on the analytic Black-Scholes curve. The lower panel shows the
error measured in units of the Monte Carlo standard error (a z-score). If the
engine is unbiased, roughly 95% of those z-scores fall within +/-1.96 -- i.e.
the discrepancy is pure sampling noise, not a bug.

As a bonus, the script also validates Monte Carlo Greeks (pathwise and
finite-difference-with-common-random-numbers) against the closed-form Greeks.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import config as C
from src import black_scholes as bs
from src import greeks, pricer


def main():
    p = C.BASE
    n_paths = 60_000
    strikes = np.arange(60, 141, 2.5)
    # Independent RNG stream per strike, so the z-scores are i.i.d. and the
    # coverage fraction below is a meaningful test of unbiasedness.
    child_rngs = [np.random.default_rng(s)
                  for s in np.random.SeedSequence(C.MASTER_SEED).spawn(len(strikes))]

    mc_prices, mc_ses, bs_prices, zscores = [], [], [], []
    for K, rng in zip(strikes, child_rngs):
        res = pricer.price_european(p["S0"], K, p["T"], p["r"], p["sigma"],
                                    n_paths, rng)
        exact = bs.bs_price(p["S0"], K, p["T"], p["r"], p["sigma"], "call")
        mc_prices.append(res.price)
        mc_ses.append(res.std_error)
        bs_prices.append(exact)
        zscores.append((res.price - exact) / res.std_error)

    mc_prices = np.array(mc_prices)
    mc_ses = np.array(mc_ses)
    bs_prices = np.array(bs_prices)
    zscores = np.array(zscores)

    # --- table ---------------------------------------------------------------
    lines = ["Strike     BS price     MC price     MC SE      error/SE (z)"]
    lines.append("-" * 60)
    for K, b, m, se, z in zip(strikes, bs_prices, mc_prices, mc_ses, zscores):
        lines.append(f"{K:6.0f} {b:12.4f} {m:12.4f} {se:10.4f} {z:14.2f}")
    frac = np.mean(np.abs(zscores) <= 1.96)
    lines.append("-" * 60)
    lines.append(f"Fraction of |z| <= 1.96: {frac:.0%} "
                 f"(expect ~95% if the estimator is unbiased)")
    table = "\n".join(lines)
    print(table)
    with open(f"{C.RESULTS_DIR}/validation.txt", "w") as f:
        f.write(table + "\n")

    # --- Greeks validation ---------------------------------------------------
    rng = np.random.default_rng(C.MASTER_SEED + 1)
    K = p["K"]
    pw_d, pw_d_se = greeks.pathwise_delta(p["S0"], K, p["T"], p["r"], p["sigma"],
                                          400_000, rng)
    fd_d, fd_d_se = greeks.fd_delta_crn(p["S0"], K, p["T"], p["r"], p["sigma"],
                                        400_000, rng)
    pw_v, pw_v_se = greeks.pathwise_vega(p["S0"], K, p["T"], p["r"], p["sigma"],
                                         400_000, rng)
    exact = bs.bs_greeks(p["S0"], K, p["T"], p["r"], p["sigma"], "call")
    glines = ["", "Greeks validation at K = 100 (n = 400,000)",
              "-" * 60,
              f"Delta  closed form = {exact['delta']:.4f}",
              f"       pathwise    = {pw_d:.4f} +/- {pw_d_se:.4f}",
              f"       finite diff = {fd_d:.4f} +/- {fd_d_se:.4f}",
              f"Vega   closed form = {exact['vega']:.4f}",
              f"       pathwise    = {pw_v:.4f} +/- {pw_v_se:.4f}"]
    gtable = "\n".join(glines)
    print(gtable)
    with open(f"{C.RESULTS_DIR}/validation.txt", "a") as f:
        f.write(gtable + "\n")

    # --- figure --------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4),
                                   gridspec_kw={"height_ratios": [3, 1.4]},
                                   sharex=True)

    ax1.plot(strikes, bs_prices, "-", color=C.NAVY, lw=2,
             label="Black-Scholes (closed form)", zorder=1)
    ax1.errorbar(strikes, mc_prices, yerr=1.96 * mc_ses, fmt="o", color=C.CRIMSON,
                 ms=5, capsize=3, lw=1.2, label="Monte Carlo (95% CI)", zorder=2)
    ax1.set_ylabel("Call price")
    ax1.set_title("Monte Carlo prices track the closed form across strikes")
    ax1.legend(loc="upper right")

    ax2.axhspan(-1.96, 1.96, color=C.TEAL, alpha=0.15, label="+/-1.96 band")
    ax2.axhline(0, color=C.GREY, lw=0.8)
    ax2.plot(strikes, zscores, "o-", color=C.NAVY, ms=4, lw=1)
    ax2.set_ylabel("error / SE")
    ax2.set_xlabel("Strike K")
    ax2.set_ylim(-3, 3)

    fig.savefig(f"{C.FIG_DIR}/01_validation.png", bbox_inches="tight")
    print(f"\nSaved figure -> figures/01_validation.png")


if __name__ == "__main__":
    main()
