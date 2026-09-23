from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import config as C
from src import black_scholes as bs
from src import payoffs as pf
from src import pricer


def main():
    p = C.BASE
    rng = np.random.default_rng(C.MASTER_SEED)
    n_steps = 252          # daily monitoring
    n_paths = 100_000

    geo_cf = bs.geometric_asian_call(p["S0"], p["K"], p["T"], p["r"],
                                     p["sigma"], n_steps)
    geo_mc = pricer.price_path_dependent(
        pf.geometric_asian_call_payoff, p["S0"], p["K"], p["T"], p["r"],
        p["sigma"], n_paths, n_steps, rng, method_name="Geometric Asian MC")

    arith_crude = pricer.price_path_dependent(
        pf.arithmetic_asian_call, p["S0"], p["K"], p["T"], p["r"], p["sigma"],
        n_paths, n_steps, rng, method_name="Arithmetic Asian MC (crude)")
    arith_ctrl = pricer.price_arithmetic_asian_control(
        p["S0"], p["K"], p["T"], p["r"], p["sigma"], n_paths, n_steps, rng)
    asian_vrf = (arith_crude.std_error / arith_ctrl.std_error) ** 2

    barrier = 85.0
    barrier_mc = pricer.price_path_dependent(
        pf.down_and_out_call, p["S0"], p["K"], p["T"], p["r"], p["sigma"],
        n_paths, n_steps, rng, method_name="Down-and-out call MC",
        barrier=barrier)
    vanilla = bs.bs_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"], "call")

    lines = [
        f"Path-dependent pricing (N = {n_paths:,}, {n_steps} steps)", "=" * 64,
        "",
        "Geometric Asian call  (closed form exists -> engine validation):",
        f"    closed form : {geo_cf:.4f}",
        f"    {geo_mc}",
        f"    agree within CI: {abs(geo_mc.price - geo_cf) < 1.96 * geo_mc.std_error}",
        "",
        "Arithmetic Asian call  (NO closed form -> Monte Carlo required):",
        f"    {arith_crude}",
        f"    {arith_ctrl}",
        f"    variance reduction factor from control variate: {asian_vrf:.0f}x",
        "",
        "Down-and-out barrier call  (barrier = 85):",
        f"    {barrier_mc}",
        f"    vanilla call for reference: {vanilla:.4f}",
        f"    barrier discount: {vanilla - barrier_mc.price:.4f} "
        f"(knock-out risk makes it cheaper)",
    ]
    table = "\n".join(lines)
    print(table)
    with open(f"{C.RESULTS_DIR}/path_dependent.txt", "w") as f:
        f.write(table + "\n")

    demo_rng = np.random.default_rng(1)
    from src.simulation import simulate_paths
    demo = simulate_paths(p["S0"], p["T"], p["r"], p["sigma"], 60, n_steps,
                          demo_rng)
    t = np.linspace(0, p["T"], n_steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    knocked = demo.min(axis=1) <= barrier
    for path, ko in zip(demo, knocked):
        ax1.plot(t, path, lw=0.7, alpha=0.5,
                 color=(C.CRIMSON if ko else C.TEAL))
    ax1.axhline(barrier, color=C.NAVY, lw=1.8, ls="--", label=f"barrier = {barrier}")
    ax1.axhline(p["K"], color=C.GREY, lw=1.2, ls=":", label=f"strike = {p['K']:.0f}")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Price")
    ax1.set_title("Sample paths (red = knocked out)")
    ax1.legend(loc="upper left", fontsize=9)

    methods = ["Crude", "Control\nvariate"]
    ses = [arith_crude.std_error, arith_ctrl.std_error]
    bars = ax2.bar(methods, ses, color=[C.GREY, C.TEAL], edgecolor="white")
    for b, se in zip(bars, ses):
        ax2.text(b.get_x() + b.get_width() / 2, se, f"{se:.4f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.set_ylabel("Standard error")
    ax2.set_title(f"Arithmetic Asian: {asian_vrf:.0f}x variance reduction")

    fig.savefig(f"{C.FIG_DIR}/04_path_dependent.png", bbox_inches="tight")
    print("\nSaved figure -> figures/04_path_dependent.png")


if __name__ == "__main__":
    main()
