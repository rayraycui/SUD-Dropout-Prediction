#!/usr/bin/env python3
"""
06_covid_temporal.py: COVID-era transportability on the trained substances:
fit on big-6 episodes from 2015-2018, test on 2019 (pre-COVID, held-out), 2020
(COVID onset) and 2022 (post-COVID).

Training assembly: "year-restricted JSH" (src/jsh_year_restricted.py), the
canonical joint-stratified holdout with the big-6 pool filtered to YEAR <= 2018
before the split and before the 1,000,000-row cap. Same features, same joint
(subname x y) stratification, same estimator, same seed 0 as src/jsh_fit.py;
the year filter is the only difference. No 70/30 holdout is taken: the test
years are excluded from the pool, so train and test are disjoint already (see
src/jsh_year_restricted.py). This script carries no sampling scheme
of its own.

This is the trained-substance series in Fig. 4b. Its unseen-class counterpart is
a separate within-unseen replication computed in src/07b_unseen_deployability.py
(covid_temporal(), seed 42, outside JSH by decision).

Usage:  python src/06_covid_temporal.py
Inputs: data/teds_d_analysis_2015_2022.parquet
Output: data/p1_covid_temporal.csv, figures/p1_covid_temporal.png
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "48")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jsh_year_restricted import fit_jsh_years, score, SEED

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def goodness_arrow(ax, text="higher = better", loc="upper left", axis="y", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams["legend.fontsize"]
    pos = {"upper left": (0.02, 0.98), "upper right": (0.98, 0.98),
           "lower left": (0.02, 0.02), "lower right": (0.98, 0.02)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    arrow = "↑ " if axis == "y" else "→ "
    ax.text(pos[0], pos[1], arrow + text, transform=ax.transAxes,
            fontsize=fontsize, color=META_GREY, ha=ha, va=va)


DATA = "data/teds_d_analysis_2015_2022.parquet"
BOOT_N = 300
TESTCAP = 150_000

d = pd.read_parquet(DATA)
big6 = d[d.is_big6]


def boot_ci(y, p, seed=SEED):
    r = np.random.default_rng(seed)
    if len(y) > TESTCAP:
        idx = r.choice(len(y), TESTCAP, replace=False)
        y, p = y[idx], p[idx]
    n = len(y)
    a = []
    for _ in range(BOOT_N):
        b = r.integers(0, n, n)
        yb = y[b]
        if yb.min() != yb.max():
            a.append(roc_auc_score(yb, p[b]))
    return round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)


m2, cap, _pool = fit_jsh_years(d, d.YEAR <= 2018, label="(big-6, YEAR<=2018)")

rows = []
tests = {"2019 (pre-COVID, held-out)": big6[big6.YEAR == 2019],
         "2020 (COVID)": big6[big6.YEAR == 2020],
         "2022 (post-COVID)": big6[big6.YEAR == 2022]}
for label, te2 in tests.items():
    p = score(m2, te2)
    y = te2.y.values
    auc = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p)
    rows.append({"test_era": label, "n": len(te2), "dropout": round(y.mean(), 3),
                 "auc": round(auc, 4), "ci_lo": lo, "ci_hi": hi})
    print(f"  train 2015-18 \u2192 {label}: AUROC={auc:.4f} [{lo},{hi}] n={len(te2):,}")
covid = pd.DataFrame(rows)
covid.to_csv("data/p1_covid_temporal.csv", index=False)

apply_figure_style()
covid = pd.read_csv("data/p1_covid_temporal.csv")
covid["order"] = range(len(covid))
covid = covid.sort_values("order")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(covid))
yerr = np.clip(np.vstack([covid.auc - covid.ci_lo, covid.ci_hi - covid.auc]), 0, None)
colors = ["#9ab8d6", "#c0504d", "#1f4e79"]
ax.errorbar(x, covid.auc, yerr=yerr, fmt="-", color="#1f4e79", capsize=3, lw=1.5, zorder=3)
for xi, (_, r), c in zip(x, covid.iterrows(), colors):
    ax.scatter([xi], [r.auc], color=c, s=90, zorder=4)
    ax.annotate(f"{r.auc:.3f}", (xi, r.auc), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8)
ax.axvspan(0.5, 2.5, color="#c0504d", alpha=0.06, zorder=0)
ax.text(1.5, 0.792, "COVID era", ha="center", fontsize=8, color="#c0504d")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.test_era}\n(drop={r.dropout:.3f})" for _, r in covid.iterrows()],
                   fontsize=7.5)
ax.set_ylabel("AUROC (95% CI)")
ax.set_ylim(0.755, 0.795)
ax.set_title("Pre-COVID model (train 2015\u20132018) transports across the pandemic", fontsize=9.5)
ax.set_xlabel("Test era")
ax.yaxis.set_label_coords(-0.10, 0.5)
goodness_arrow(ax, "higher = better", axis="y")
fig.tight_layout()
fig.savefig("figures/p1_covid_temporal.png", dpi=300, bbox_inches="tight")
