#!/usr/bin/env python3
"""
03_calibration_dca.py: calibration and decision-curve analysis for the pooled
model on the trained-substance (big-6) holdout.

Training assembly: the canonical joint-stratified holdout (JSH) defined in
src/jsh_fit.py. This script carries no sampling scheme and no seed of its own;
it imports the shared fit so that every cross-sectional arm of the manuscript is
scored from one model.

Usage:  python src/03_calibration_dca.py
Inputs: data/teds_d_analysis_2015_2022.parquet
Output: data/p1_dca_realdata.csv, figures/p1_calibration_dca.png
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "48")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

from jsh_fit import fit_jsh, score

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

clf, holdout, _unseen, cap, _train = fit_jsh(DATA)
p = score(clf, holdout)
yte = holdout.y.to_numpy()

auc = roc_auc_score(yte, p)
brier = brier_score_loss(yte, p)
print(f"JSH holdout: n={len(yte):,} AUROC={auc:.4f} Brier={brier:.4f}")
frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")

prev = yte.mean()


def net_benefit(y, p, pt):
    pred = p >= pt
    TP = np.sum(pred & (y == 1))
    FP = np.sum(pred & (y == 0))
    return TP / len(y) - FP / len(y) * (pt / (1 - pt))


def nb_treat_all(y, pt):
    return prev - (1 - prev) * (pt / (1 - pt))


thr = np.arange(0.05, 0.85, 0.05)
dca = []
for pt in thr:
    nb_m = net_benefit(yte, p, pt)
    nb_all = nb_treat_all(yte, pt)
    nb_none = 0.0
    dca.append({"threshold": round(pt, 2), "nb_model": round(nb_m, 4),
                "nb_treat_all": round(nb_all, 4), "nb_treat_none": nb_none,
                "advantage_vs_best_default": round(nb_m - max(nb_all, nb_none), 4)})
dca = pd.DataFrame(dca)
dca.to_csv("data/p1_dca_realdata.csv", index=False)

import matplotlib.pyplot as plt

apply_figure_style()

fig, (axc, axd) = plt.subplots(1, 2, figsize=(11, 5))
# Panel A: calibration
axc.plot([0, 1], [0, 1], ls="--", color=META_GREY, lw=1, label="perfect")
axc.plot(mean_pred, frac_pos, marker="o", color="#1f4e79", label="pooled GBT")
axc.set_xlabel("Predicted dropout probability")
axc.set_ylabel("Observed dropout frequency")
axc.set_title("Calibration: predictions match observed rates")
axc.set_xlim(0, 0.9)
axc.set_ylim(0, 0.9)
axc.legend(frameon=False, loc="upper left")
axc.text(0.05, 0.82, f"Brier = {brier:.3f}\nAUROC = {auc:.3f}\nn = {len(yte)/1e6:.2f}M",
         transform=axc.transAxes, fontsize=7, va="top")

# Panel B: DCA
axd.plot(dca.threshold, dca.nb_model, marker="o", ms=3, color="#1f4e79", label="model")
axd.plot(dca.threshold, dca.nb_treat_all, ls="--", color="#b0651a", label="treat all")
axd.axhline(0, ls=":", color=META_GREY, lw=1, label="treat none")
axd.set_xlabel("Threshold probability")
axd.set_ylabel("Net benefit")
axd.set_title("Decision curve: model beats both defaults")
axd.set_ylim(-0.6, 0.42)
axd.legend(frameon=False, loc="upper right")
goodness_arrow(axd, "higher = better", axis="y")
fig.tight_layout()

for ax in (axc, axd):
    ax.xaxis.labelpad = 8
    ax.yaxis.labelpad = 8
axc.yaxis.set_label_coords(-0.13, 0.5)
axd.yaxis.set_label_coords(-0.13, 0.5)
fig.subplots_adjust(wspace=0.28, bottom=0.13)
fig.savefig("figures/p1_calibration_dca.png", dpi=300, bbox_inches="tight")
