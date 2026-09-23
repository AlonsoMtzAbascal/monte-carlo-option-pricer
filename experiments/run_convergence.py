"""
Experiment 2 -- Convergence and the 1/sqrt(N) law.

Two questions answered:
  (a) does the estimate converge to the true (Black-Scholes) price as the
      number of paths grows?
  (b) does the error shrink at the theoretical Monte Carlo rate of 1/sqrt(N)?

Left panel: the price estimate with its 95% confidence band as N increases,
against the exact price. The band narrows and the estimate homes in.

Right panel: standard error vs N on log-log axes. Monte Carlo error scales as
sigma_payoff / sqrt(N), so on log-log axes the points lie on a straight line of
slope -1/2. We fit the slope and print it -- getting -0.5 back is the
quantitative confirmation that the estimator behaves as theory predicts.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import config as C
from src import black_scholes as bs
from src import pricer


def main():
    p = C.BASE
    rng = np.random.default_rng(C.MASTER_SEED)
    exact = bs.bs_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"], "call")

    Ns = np.array([250, 500, 1_000, 2_500, 5_000, 10_000, 25_000,
                   50_000, 100_000, 250_000, 500_000, 1_000_000])
    prices, ses = [], []
    for N in Ns:
        res = pricer.price_european(p["S0"], p["K"], p["T"], p["r"], p["sigma"],
                                    int(N), rng)
        prices.append(res.price)
        ses.append(res.std_error)
    prices, ses = np.array(prices), np.array(ses)

    # Fit the log-log slope of SE vs N.
    slope, intercept = np.polyfit(np.log(Ns), np.log(ses), 1)
    print(f"Exact (Black-Scholes) price: {exact:.4f}")
    print(f"Fitted log-log slope of SE vs N: {slope:.3f} "
          f"(theory: -0.500)")
    with open(f"{C.RESULTS_DIR}/convergence.txt", "w") as f:
        f.write(f"Exact price: {exact:.6f}\n")
        f.write(f"Fitted SE-vs-N log-log slope: {slope:.4f} (theory -0.5)\n\n")
        f.write("N, price, std_error\n")
        for N, pr, se in zip(Ns, prices, ses):
            f.write(f"{int(N)}, {pr:.6f}, {se:.6f}\n")

    # --- figure --------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.axhline(exact, color=C.NAVY, lw=2, label=f"Exact = {exact:.3f}")
    ax1.fill_between(Ns, prices - 1.96 * ses, prices + 1.96 * ses,
                     color=C.CRIMSON, alpha=0.18, label="95% CI")
    ax1.plot(Ns, prices, "o-", color=C.CRIMSON, ms=4, lw=1, label="MC estimate")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of paths N")
    ax1.set_ylabel("Call price")
    ax1.set_title("Estimate converges to the closed form")
    ax1.legend(loc="upper right")

    ax2.loglog(Ns, ses, "o", color=C.CRIMSON, ms=5, label="observed SE")
    fit = np.exp(intercept) * Ns.astype(float) ** slope
    ax2.loglog(Ns, fit, "-", color=C.NAVY, lw=2,
               label=f"fit: slope = {slope:.2f}")
    # reference line of exact slope -1/2 anchored at the first point
    ref = ses[0] * (Ns / Ns[0]) ** (-0.5)
    ax2.loglog(Ns, ref, "--", color=C.GREY, lw=1.3, label="slope = -1/2 (theory)")
    ax2.set_xlabel("Number of paths N")
    ax2.set_ylabel("Standard error")
    ax2.set_title("Error decays as 1/sqrt(N)")
    ax2.legend(loc="lower left")

    fig.savefig(f"{C.FIG_DIR}/02_convergence.png", bbox_inches="tight")
    print("Saved figure -> figures/02_convergence.png")


if __name__ == "__main__":
    main()
