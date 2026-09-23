from __future__ import annotations

import os

import matplotlib as mpl

BASE = dict(
    S0=100.0,     # spot
    K=100.0,      # strike (at-the-money)
    T=1.0,        # 1 year to maturity
    r=0.05,       # 5% risk-free rate
    sigma=0.20,   # 20% volatility
)

MASTER_SEED = 20240517

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NAVY = "#1f3b57"
TEAL = "#2a9d8f"
AMBER = "#e9a13b"
CRIMSON = "#c1443c"
GREY = "#8a8f98"

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": True,
})
