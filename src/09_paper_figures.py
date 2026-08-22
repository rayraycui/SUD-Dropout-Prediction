#!/usr/bin/env python3
"""Render the six manuscript figures at true IEEE printed width.

The exploratory figures in src/02b..08 are built on wide canvases for
on-screen review. LaTeX rescales an 8-12 inch canvas into a 3.5 inch
column, which shrinks all text by ~2-3x and prints labels below the 9 pt
body size. This script rebuilds the same results at the width they are
actually printed at (3.5 in single column), so no rescaling occurs and
tick/label sizes in the PDF match what is set here.

Inputs  (all committed under data/)
  p1_coverage_ci.csv               -> Fig 1
  p1_phase1_single_pooled.csv -> Fig 2
  fairness_preds.parquet           -> Fig 3 (calibration), Fig 5 reference line
  p1_dca_realdata.csv              -> Fig 3 (decision curve)
  p1_covid_temporal.csv            -> Fig 4
  p1_fairness_subgroups.csv        -> Fig 5
  p1_fairness_states.csv           -> Fig 5 (state range in title)
  p1_tabpfn_uncapped.csv           -> Fig 6

Outputs: paper/fig1_coverage.png ... paper/fig6_tabpfn.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, brier_score_loss

COL = 3.5          # IEEE single-column width, inches
TS, LS, AS_ = 9.5, 9.0, 8.2                  # title / label / annotation pt
BLUE, GREY, LGREY = "#2b5d8a", "#7a7a7a", "#c9c9c9"
RED, ORANGE = "#c0504d", "#c9722a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": AS_, "ytick.labelsize": AS_, "axes.labelsize": LS,
    "axes.titlesize": TS, "legend.fontsize": AS_,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "figure.dpi": 400,
})

def err(v, lo, hi):
    """Asymmetric error bars, clamped at 0 (bootstrap CIs can sit off-point)."""
    return [np.maximum(0, v - lo), np.maximum(0, hi - v)]

def _quad(t, r):
    """True (possibly rotated) corner polygon of a Text, in display coords.

    Text.get_window_extent returns an AXIS-ALIGNED box. For text rotated off the
    axes (the 45-degree category labels below) that box is far larger than the
    glyphs, so adjacent labels appear to collide when they do not. Rotating the
    unrotated extent about the text's anchor gives the real footprint.
    """
    ang = np.deg2rad(t.get_rotation())
    if abs(ang) < 1e-9:
        b = t.get_window_extent(r)
        return [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]
    saved = t.get_rotation()
    t.set_rotation(0)
    b = t.get_window_extent(r)
    t.set_rotation(saved)
    ax_, ay_ = t.get_transform().transform(t.get_position())
    ca, sa = np.cos(ang), np.sin(ang)
    pts = []
    for px, py in [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]:
        dx, dy = px - ax_, py - ay_
        pts.append((ax_ + dx * ca - dy * sa, ay_ + dx * sa + dy * ca))
    return pts


def _overlaps(p, q):
    """Separating-axis test on two convex polygons."""
    for poly in (p, q):
        for i in range(len(poly)):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % len(poly)]
            nx, ny = -(y1 - y0), (x1 - x0)
            n = np.hypot(nx, ny)
            if n == 0:
                continue
            nx, ny = nx / n, ny / n
            a = [px * nx + py * ny for px, py in p]
            b = [px * nx + py * ny for px, py in q]
            if max(a) <= min(b) + 0.5 or max(b) <= min(a) + 0.5:
                return False
    return True


def overlap_count(fig, name):
    """Text-overlap check: text-vs-text and text-vs-spine, rotation-aware."""
    fig.canvas.draw()  # extents are only correct once the canvas has been drawn
    r = fig.canvas.get_renderer()
    texts = [(t, _quad(t, r)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    spines = []
    for ax in fig.axes:
        for s in ax.spines.values():
            if s.get_visible():
                b = s.get_window_extent(r)
                spines.append((s, [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]))
    tl = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
          for ax in fig.axes}
    # Matplotlib keeps a Text for every tick the locator produced, including ticks
    # OUTSIDE the view interval (e.g. a y-tick at -0.2 on an axis limited to
    # -0.045..0.945). Those glyphs are never rendered but still report an extent,
    # and they collide with real labels in other panels. Drop them: only text that
    # actually appears in the PNG can overlap.
    ghosts = set()
    for ax in fig.axes:
        for axis, lo_hi in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lo_hi)
            for t in axis.get_ticklabels(which="both"):
                v = t.get_position()[0 if axis is ax.xaxis else 1]
                if not (lo - 1e-9 <= v <= hi + 1e-9):
                    ghosts.add(t)
    texts = [(t, q) for t, q in texts if t not in ghosts]
    ov = [(a, b) for i, (a, qa) in enumerate(texts) for b, qb in texts[i + 1:] if _overlaps(qa, qb)]
    ov += [(t, s) for t, qt in texts for s, qs in spines
           if _overlaps(qt, qs) and t not in tl[s.axes]]
    print(f"{name}: text overlaps = {len(ov)}")
    for a, b in ov:
        print("   ", repr(a.get_text())[:44], "<->",
              repr(getattr(b, "get_text", lambda: "<spine>")())[:44])
    return len(ov)


def save(fig, path):
    fig.savefig(path)   # no bbox='tight': it would expand past COL
    n = overlap_count(fig, path.rsplit("/", 1)[-1])
    plt.close(fig)
    print("wrote", path)
    return n

SHORT = {"Alcohol": "Alcohol", "Heroin": "Heroin", "Cannabis": "Cannabis",
         "Methamphetamine": "Metham.", "Other_opioids": "Oth. opioid",
         "Cocaine": "Cocaine"}
SMALL = {"Other_sedatives": "Oth. sedatives", "Other_amphetamines": "Oth. amphet.",
         "Other_stimulants": "Oth. stimulants", "Non_rx_methadone": "Non-Rx methadone",
         "Other_tranquilizers": "Oth. tranq.", "Other_hallucinogens": "Oth. halluc.",
         "Other_sedatives_hypnotics": "Oth. sed./hyp.", "Inhalants": "Inhalants",
         "PCP": "PCP", "OTC": "OTC", "Barbiturates": "Barbiturates",
         "Benzodiazepines": "Benzodiaz."}
ORDER = ["Alcohol", "Heroin", "Cannabis", "Methamphetamine", "Other_opioids", "Cocaine"]

# ---------------------------------------------------------------- Fig 1
cv = pd.read_csv("data/p1_coverage_ci.csv").sort_values("auc")
y = np.arange(len(cv))
fig, ax = plt.subplots(figsize=(COL, 2.05), constrained_layout=True)
ax.errorbar(cv.auc, y, xerr=err(cv.auc.values, cv.ci_lo.values, cv.ci_hi.values),
            fmt="o", ms=2.3, color=BLUE, ecolor=BLUE, elinewidth=0.7, capsize=1.0, lw=0)
# Reference: the pooled model's mean AUROC across the six trained substances under
# the same canonical (JSH) fit: read from data/p1_phase1_pooled_jsh.csv, not hardcoded.
BIG6_REF = float(pd.read_csv("data/p1_phase1_pooled_jsh.csv").auc.mean())
ax.axvline(BIG6_REF, ls="--", lw=0.85, color=ORANGE)
# Horizontal, tucked under the foot of the line. A rotated label beside the line
# cannot work here: 8 of the 12 CIs cross it (ci_hi up to 0.835), so a label tall
# enough to read always strikes an error bar. The band below row 0 is empty.
ax.text(BIG6_REF + 0.0030, -0.52, "big-6 pooled ref.", color=ORANGE,
        fontsize=AS_ - 1.0, ha="left", va="bottom")
ax.set_yticks(y)
ax.set_yticklabels([f"{SMALL[s]} ({n:,})" for s, n in zip(cv.substance, cv.n)],
                   fontsize=AS_ - 1.0)
ax.set_ylim(-0.6, len(cv) - 0.4)
ax.set_xlim(0.730, 0.845)
ax.set_xticks([0.74, 0.78, 0.82])
ax.tick_params(axis="x", labelsize=AS_ - 1.8)
ax.set_xlabel("AUROC, unseen class (95% CI)", fontsize=LS - 1.2)
fig.suptitle("Pooled model covers 12 unseen substance classes",
             fontsize=TS - 0.8, x=0.005, ha="left")
save(fig, "paper/fig1_coverage.png")
print(f"fig1: big-6 pooled reference = {BIG6_REF:.4f}")
for s_, n_, a_, lo_, hi_ in zip(cv.substance, cv.n, cv.auc, cv.ci_lo, cv.ci_hi):
    print(f"   {SMALL[s_]:<18s} n={n_:>6,}  AUROC={a_:.4f} [{lo_:.4f}, {hi_:.4f}]")

# ---------------------------------------------------------------- Fig 2
rg = pd.read_csv("data/p1_phase1_single_pooled.csv")
gb = rg[rg.family == "GBT"]
w = 0.26
fig, ax = plt.subplots(figsize=(COL, 1.72), constrained_layout=True)
for i, (reg, c) in enumerate(zip(["single", "pooled"], [BLUE, LGREY])):
    d = gb[gb.regime == reg].set_index("substance").loc[ORDER]
    ax.bar(np.arange(6) + (i - 0.5) * w, d.auc, w, color=c, label=reg,
           yerr=err(d.auc.values, d.ci_lo.values, d.ci_hi.values),
           error_kw=dict(lw=0.6, capsize=1.0))
ax.set_xticks(np.arange(6))
ax.set_xticklabels([SHORT[s] for s in ORDER], rotation=25, ha="right",
                   fontsize=AS_ - 1.6)
ax.set_ylim(0.70, 0.828)
ax.set_yticks([0.70, 0.74, 0.78, 0.82])
ax.tick_params(axis="y", labelsize=AS_ - 1.5)
ax.set_ylabel("AUROC (held-out)", fontsize=LS - 1.0)
ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=AS_ - 1.6,
          handlelength=0.8, columnspacing=0.7, borderpad=0.05)
fig.suptitle("Substance-specific vs pooled training (GBT)", fontsize=TS - 0.8, x=0.005, ha="left")
save(fig, "paper/fig2_regimes.png")

# ---------------------------------------------------------------- Fig 3
fp = pd.read_parquet("data/fairness_preds.parquet", columns=["y", "p"])
overall_auc = roc_auc_score(fp.y, fp.p)
overall_brier = brier_score_loss(fp.y, fp.p)
print(f"pooled GBT held-out: n={len(fp)} AUROC={overall_auc:.4f} Brier={overall_brier:.4f}")
qb = pd.qcut(fp.p, 10, labels=False, duplicates="drop")
cal = fp.groupby(qb).agg(pm=("p", "mean"), om=("y", "mean")).reset_index(drop=True)
dc = pd.read_csv("data/p1_dca_realdata.csv")

fig, axes = plt.subplots(1, 2, figsize=(COL, 1.52), constrained_layout=True)
a = axes[0]
a.plot([0, 0.9], [0, 0.9], ls="--", lw=0.7, color=GREY)
a.plot(cal.pm, cal.om, "-o", ms=2.0, lw=0.9, color=BLUE)
a.set_xlabel("Predicted", fontsize=LS - 1.5)
a.set_ylabel("Observed", fontsize=LS - 1.5)
a.set_title("Calibration", fontsize=TS - 1.2, loc="left")
a.tick_params(labelsize=AS_ - 2.0)
a.text(0.05, 0.97, f"Brier {overall_brier:.3f}\nAUROC {overall_auc:.3f}",
       fontsize=AS_ - 2.0, va="top", transform=a.transAxes)
b = axes[1]
b.plot(dc.threshold, dc.nb_model, "-", lw=1.1, color=BLUE, label="model")
b.plot(dc.threshold, dc.nb_treat_all, "--", lw=0.9, color=ORANGE, label="treat all")
b.plot(dc.threshold, dc.nb_treat_none, ":", lw=0.9, color=GREY, label="treat none")
b.set_ylim(-0.32, 0.42)
b.set_xlabel("Threshold", fontsize=LS - 1.5)
b.set_ylabel("Net benefit", fontsize=LS - 1.5)
b.set_title("Decision curve", fontsize=TS - 1.2, loc="left")
b.tick_params(labelsize=AS_ - 2.0)
b.legend(frameon=False, fontsize=AS_ - 2.2, loc="lower left", handlelength=1.1,
         borderpad=0.05, labelspacing=0.15)
save(fig, "paper/fig3_calib_dca.png")

# ---------------------------------------------------------------- Fig 4
cd = pd.read_csv("data/p1_covid_temporal.csv")
x = np.arange(len(cd))
fig, ax = plt.subplots(figsize=(COL, 1.75), constrained_layout=True)
ax.axvspan(0.5, 2.5, color=RED, alpha=0.07, lw=0)
ax.errorbar(x, cd.auc, yerr=err(cd.auc.values, cd.ci_lo.values, cd.ci_hi.values),
            fmt="-o", ms=3.6, color=BLUE, ecolor=BLUE, elinewidth=0.9,
            capsize=1.8, lw=1.1)
for xi, a_ in zip(x, cd.auc):
    ax.annotate(f"{a_:.3f}", (xi, a_), textcoords="offset points", xytext=(0, 5.5),
                ha="center", fontsize=AS_ - 1.2)
ax.set_ylim(0.7625, 0.7925)
ax.set_yticks([0.765, 0.775, 0.785])
ax.tick_params(axis="y", labelsize=AS_ - 1.5)
ax.text(2.42, 0.7918, "COVID era", color=RED, fontsize=AS_ - 1.4,
        ha="right", va="top")
ax.set_xticks(x)
ax.set_xticklabels(["2019\npre-COVID", "2020\nCOVID", "2022\npost-COVID"],
                   fontsize=AS_ - 1.5)
ax.set_xlim(-0.45, 2.45)
ax.set_ylabel("AUROC (95% CI)", fontsize=LS - 1.0)
fig.suptitle("Model trained on 2015\u20132018 transports across the pandemic",
             fontsize=TS - 0.8, x=0.005, ha="left")
save(fig, "paper/fig4_covid.png")

# ---------------------------------------------------------------- Fig 5
fs = pd.read_csv("data/p1_fairness_subgroups.csv")
fst = pd.read_csv("data/p1_fairness_states.csv")
fig, axes = plt.subplots(1, 2, figsize=(COL, 2.10), constrained_layout=True)
for ax, dim, ttl in zip(axes, ["RACE", "AGE"], ["Race/ethnicity", "Age band"]):
    d = fs[fs.dim == dim].sort_values("auc")
    yy = np.arange(len(d))
    ax.errorbar(d.auc, yy, xerr=err(d.auc.values, d.ci_lo.values, d.ci_hi.values),
                fmt="o", ms=1.9, color=BLUE, ecolor=BLUE, elinewidth=0.6,
                capsize=0.8, lw=0)
    ax.axvline(overall_auc, ls="--", lw=0.7, color=RED)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.group, fontsize=AS_ - 2.4)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("AUROC (95% CI)", fontsize=LS - 1.6)
    ax.set_xticks([0.74, 0.79, 0.84])
    ax.tick_params(axis="x", labelsize=AS_ - 2.2)
    ax.set_title(ttl, fontsize=TS - 1.4, loc="left")
fig.suptitle(f"Discrimination varies (states {fst.auc.min():.2f}\u2013{fst.auc.max():.2f}); "
             "calibration uniform", fontsize=TS - 0.8, x=0.005, ha="left")
save(fig, "paper/fig5_fairness.png")

# ---------------------------------------------------------------- Fig 6
mm = pd.read_csv("data/p1_tabpfn_uncapped.csv")
mm["lab"] = [f"{SHORT.get(s, s.replace('_', ' ')[:11])} {r}" if r != "coverage"
             else f"{SMALL.get(s, s)[:13]} cov." for s, r in zip(mm.substance, mm.regime)]
mm = mm.sort_values("delta").reset_index(drop=True)
print(f"uncapped: {(mm.delta > 0).sum()}/{len(mm)} favour TabPFN, mean delta={mm.delta.mean():+.4f}; "
      f"matched mean={mm.delta_matched.mean():+.4f}; "
      f"CIs including zero={((mm.d_lo <= 0) & (mm.d_hi >= 0)).sum()}")
yy = np.arange(len(mm))
fig, ax = plt.subplots(figsize=(COL, 2.55), constrained_layout=True)
ax.axvline(0, lw=0.8, color="#444")
for i, (dm, d_) in enumerate(zip(mm.delta_matched, mm.delta)):
    ax.annotate("", xy=(d_, i), xytext=(dm, i),
                arrowprops=dict(arrowstyle="-|>", lw=0.45, color="#b9c6d4",
                                shrinkA=1.0, shrinkB=1.0, mutation_scale=3.5))
ax.scatter(mm.delta_matched, yy, s=4.5, color="#8fa9c4", zorder=3)
for i, r in mm.iterrows():
    c = RED if r.sig == "yes" else BLUE
    ax.plot([r.d_lo, r.d_hi], [i, i], lw=0.7, color=c, zorder=4)
    ax.plot([r.delta], [i], "o", ms=2.1, color=c, zorder=5)
ax.set_yticks(yy)
ax.set_yticklabels(mm.lab, fontsize=AS_ - 3.1)
ax.set_ylim(-0.9, len(mm) + 3.4)
ax.set_xlabel("TabPFN-3 minus GBT, AUROC", fontsize=LS - 1.2)
ax.set_xlim(-0.020, 0.041)
ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04])
ax.tick_params(axis="x", labelsize=AS_ - 2.0)
handles = [Line2D([], [], marker="o", ls="none", ms=3.0, color="#8fa9c4",
                  label="matched 10k training size"),
           Line2D([], [], marker="o", ms=3.0, color=BLUE, lw=0.9,
                  label="maximum training size (95% CI)"),
           Line2D([], [], marker="o", ms=3.0, color=RED, lw=0.9,
                  label="CI excludes zero")]
ax.legend(handles=handles, frameon=False, fontsize=AS_ - 2.8, loc="upper left",
          bbox_to_anchor=(0.0, 0.995), handlelength=1.1, borderpad=0.05,
          labelspacing=0.15)
fig.suptitle("TabPFN-3 advantage vanishes at full data",
             fontsize=TS - 0.8, x=0.005, ha="left")
save(fig, "paper/fig6_tabpfn.png")
