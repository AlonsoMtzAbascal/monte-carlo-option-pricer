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

    n_paths = 20_000     # per estimate
    n_reps = 500         # independent replications

    crude, anti, ctrl = [], [], []
    for _ in range(n_reps):
        crude.append(pricer.price_european(
            p["S0"], p["K"], p["T"], p["r"], p["sigma"], n_paths, rng).price)
        anti.append(pricer.price_european(
            p["S0"], p["K"], p["T"], p["r"], p["sigma"], n_paths, rng,
            antithetic=True).price)
        ctrl.append(pricer.price_european_control(
            p["S0"], p["K"], p["T"], p["r"], p["sigma"], n_paths, rng).price)
    crude, anti, ctrl = map(np.array, (crude, anti, ctrl))

    def summary(name, x):
        return dict(name=name, mean=x.mean(), std=x.std(ddof=1),
                    bias=x.mean() - exact)

    rows = [summary("Crude", crude), summary("Antithetic", anti),
            summary("Control variate", ctrl)]
    v_crude = rows[0]["std"] ** 2

    lines = [f"Exact price: {exact:.4f}    "
             f"(N = {n_paths:,} paths, {n_reps} replications)", "-" * 68,
             f"{'method':<18s}{'mean':>10s}{'std of est':>14s}"
             f"{'bias':>10s}{'VRF':>8s}"]
    for row in rows:
        vrf = v_crude / row["std"] ** 2
        lines.append(f"{row['name']:<18s}{row['mean']:>10.4f}"
                     f"{row['std']:>14.5f}{row['bias']:>10.5f}{vrf:>8.1f}")
    table = "\n".join(lines)
    print(table)
    with open(f"{C.RESULTS_DIR}/variance_reduction.txt", "w") as f:
        f.write(table + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    colors = [C.GREY, C.AMBER, C.TEAL]
    bins = np.linspace(min(crude.min(), anti.min(), ctrl.min()),
                       max(crude.max(), anti.max(), ctrl.max()), 45)
    for (arr, col, row) in zip([crude, anti, ctrl], colors, rows):
        ax1.hist(arr, bins=bins, alpha=0.55, color=col, label=row["name"],
                 density=True)
    ax1.axvline(exact, color=C.NAVY, lw=2, ls="--", label=f"exact = {exact:.3f}")
    ax1.set_xlabel("Price estimate")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Spread of {n_reps} estimates (tighter = better)")
    ax1.legend(loc="upper right", fontsize=9)

    names = [r["name"] for r in rows]
    vrfs = [v_crude / r["std"] ** 2 for r in rows]
    bars = ax2.bar(names, vrfs, color=colors, edgecolor="white")
    for b, v in zip(bars, vrfs):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}x",
                 ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Variance reduction factor (vs crude)")
    ax2.set_title("Equivalent speed-up over crude Monte Carlo")
    ax2.set_ylim(0, max(vrfs) * 1.25)

    fig.savefig(f"{C.FIG_DIR}/03_variance_reduction.png", bbox_inches="tight")
    print("Saved figure -> figures/03_variance_reduction.png")


if __name__ == "__main__":
    main()
