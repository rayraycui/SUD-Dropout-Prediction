#!/usr/bin/env python3
"""
04_coverage_unseen.py: coverage of the pooled model over the twelve unseen
substance classes, with bootstrap CIs per class.

Training assembly: the canonical joint-stratified holdout (JSH) defined in
src/jsh_fit.py. This script carries no sampling scheme and no seed of its own,
so the coverage AUROCs come from the same fitted model as the calibration,
decision-curve and fairness arms.

Usage:  python src/04_coverage_unseen.py
Inputs: data/teds_d_analysis_2015_2022.parquet
        data/p1_phase1_pooled_jsh.csv  (per-substance reference line)
Output: data/p1_coverage_ci.csv, figures/p1_coverage_ci.png
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

from jsh_fit import fit_jsh, score, SEED

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

clf, _holdout, small, cap, _train = fit_jsh(DATA)


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
for s in small.subname.unique():
    sd = small[small.subname == s]
    if sd.y.nunique() < 2 or len(sd) < 30:
        continue
    p = score(clf, sd)
    y = sd.y.values
    auc = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p, SEED)
    rows.append({"substance": s, "n": len(sd), "dropout": round(y.mean(), 3),
                 "auc": round(auc, 4), "ci_lo": lo, "ci_hi": hi})
cov = pd.DataFrame(rows).sort_values("auc", ascending=False)
cov.to_csv("data/p1_coverage_ci.csv", index=False)
print(cov.to_string(index=False))
print(f"\nunweighted mean AUROC={cov.auc.mean():.4f} | n-weighted={np.average(cov.auc, weights=cov.n):.4f}")

cov = pd.read_csv("data/p1_coverage_ci.csv").sort_values("auc")
# Reference line: the pooled model's mean AUROC across the six trained substances
# under the same JSH fit (data/p1_phase1_pooled_jsh.csv), not a hardcoded constant.
big6_ref = float(pd.read_csv("data/p1_phase1_pooled_jsh.csv").auc.mean())

apply_figure_style()
fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(cov))
xerr = np.clip(np.vstack([cov.auc - cov.ci_lo, cov.ci_hi - cov.auc]), 0, None)
ax.errorbar(cov.auc, y, xerr=xerr, fmt="o", color="#1f4e79", capsize=2, ms=5, lw=1)
ax.axvline(big6_ref, ls="--", color="#b0651a", lw=1.2)
ax.text(0.845, 3.2, f"big-6 pooled\nreference ({big6_ref:.3f})", ha="right", va="center",
        fontsize=7, color="#b0651a")
ax.set_yticks(y)
ax.set_yticklabels([f"{s}  (n={n:,})" for s, n in zip(cov.substance, cov.n)])
ax.set_xlabel("AUROC on held-out small class (95% CI)")
ax.set_title("Coverage: one pooled model predicts dropout for 12 unseen substance classes", fontsize=9)
ax.set_xlim(0.72, 0.86)
ax.margins(y=0.03)
ax.xaxis.labelpad = 8
goodness_arrow(ax, "higher = better", axis="x")
fig.tight_layout()
fig.savefig("figures/p1_coverage_ci.png", dpi=300, bbox_inches="tight")
r_ = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
sp = [(s, s.get_window_extent(r_)) for a in fig.axes for s in a.spines.values() if s.get_visible()]
tl = {a: set(a.get_xticklabels() + a.get_yticklabels()) for a in fig.axes}
ov = [(a, b) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
ov += [(t, s) for t, bt in texts for s, bs in sp if bt.overlaps(bs) and t not in tl[s.axes]]
print("overlaps:", len(ov))
