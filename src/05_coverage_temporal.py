#!/usr/bin/env python3
"""
05_coverage_temporal.py: coverage under temporal shift: fit on big-6 episodes
from years BEFORE 2022, test on the unseen substance classes as they appear in
2022.

Training assembly: "year-restricted JSH" (src/jsh_year_restricted.py), the
canonical joint-stratified holdout with the big-6 pool filtered to YEAR < 2022
before the split and before the 1,000,000-row cap. Same features, same joint
(subname x y) stratification, same estimator, same seed 0 as src/jsh_fit.py;
the year filter is the only difference. No 70/30 holdout is taken: the test
years are excluded from the pool, so train and test are disjoint already (see
src/jsh_year_restricted.py). This script carries no sampling scheme
of its own.

Reads the committed analytic table rather than rebuilding it from the raw PUF
CSVs (data/raw/ is not distributed); the table is produced by
src/01_build_analysis_table.py and carries the YEAR column this arm needs.

Usage:  python src/05_coverage_temporal.py
Inputs: data/teds_d_analysis_2015_2022.parquet, data/p1_coverage_ci.csv
Output: data/p1_coverage_temporal2022.csv, figures/p1_coverage_temporal.png
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
TESTCAP = 100_000

d = pd.read_parquet(DATA)

# TEMPORAL: year-restricted JSH on big-6 years < 2022, test unseen classes IN 2022
m, cap, _pool = fit_jsh_years(d, d.YEAR < 2022, label="(big-6, YEAR<2022)")
test22_small = d[(~d.is_big6) & (d.YEAR == 2022)]
print(f"test = unseen classes in 2022, n={len(test22_small):,}")


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


rows = []
for s in test22_small.subname.unique():
    sd = test22_small[test22_small.subname == s]
    if sd.y.nunique() < 2 or len(sd) < 30:
        continue
    p = score(m, sd)
    y = sd.y.values
    lo, hi = boot_ci(y, p, SEED)
    rows.append({"substance": s, "n": len(sd), "dropout": round(y.mean(), 3),
                 "auc": round(roc_auc_score(y, p), 4), "ci_lo": lo, "ci_hi": hi})
cov22 = pd.DataFrame(rows).sort_values("auc", ascending=False)
cov22.to_csv("data/p1_coverage_temporal2022.csv", index=False)
print(cov22.to_string(index=False))
print(f"\ntemporal mean AUROC={cov22.auc.mean():.4f} | n-weighted={np.average(cov22.auc, weights=cov22.n):.4f}")

apply_figure_style()
cov = pd.read_csv("data/p1_coverage_ci.csv").set_index("substance")
cov22 = cov22.set_index("substance")
common = [s for s in cov22.index if s in cov.index]
order = cov22.loc[common].sort_values("auc").index.tolist()

fig, ax = plt.subplots(figsize=(8.5, 6))
y = np.arange(len(order))
off = 0.16
c_pool = "#9ab8d6"
c_temp = "#1f4e79"
for lbl, df, c, dy in [("same-pool (all yrs)", cov, c_pool, +off),
                        ("temporal (train<2022 \u2192 test 2022)", cov22, c_temp, -off)]:
    r = df.loc[order]
    xerr = np.clip(np.vstack([r.auc - r.ci_lo, r.ci_hi - r.auc]), 0, None)
    ax.errorbar(r.auc, y + dy, xerr=xerr, fmt="o", color=c, capsize=2, ms=4, lw=1, label=lbl)
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_xlabel("AUROC on held-out small class (95% CI)")
ax.set_title("Coverage is stable under temporal shift", fontsize=10)
ax.set_xlim(0.68, 0.90)
ax.margins(y=0.03)
ax.xaxis.labelpad = 8
ax.legend(frameon=False, loc="lower right", fontsize=7)
goodness_arrow(ax, "higher = better", axis="x")
fig.tight_layout()
fig.savefig("figures/p1_coverage_temporal.png", dpi=300, bbox_inches="tight")
