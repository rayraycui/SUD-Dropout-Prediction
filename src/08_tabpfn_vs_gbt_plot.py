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


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

apply_figure_style()

df = pd.read_csv("data/p1_tabpfn_vs_gbt.csv")

reg = df[df.regime != "coverage"].copy()
cov = df[df.regime == "coverage"].copy()

piv = reg.pivot_table(index=["substance", "regime"], columns="family", values="auc")
piv["delta"] = piv["TabPFN"] - piv["GBT"]

covp = cov.pivot_table(index="substance", columns="family", values="auc")
covp["delta"] = covp["TabPFN"] - covp["GBT"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel A: big-6 regimes, grouped by regime
ax = axes[0]
order = [("single", "#1f4e79"), ("pooled", "#4a90b8")]
subs = ["Alcohol", "Cannabis", "Cocaine", "Heroin", "Methamphetamine", "Other_opioids"]
sublab = ["Alcohol", "Cannabis", "Cocaine", "Heroin", "Meth", "Other op."]
x = np.arange(len(subs)); w = 0.13
for i, (rg, c) in enumerate(order):
    s = reg[reg.regime == rg]
    g = [s[(s.substance == sb) & (s.family == "GBT")].auc.values[0] for sb in subs]
    t = [s[(s.substance == sb) & (s.family == "TabPFN")].auc.values[0] for sb in subs]
    ax.bar(x + i * 2 * w - 2.5 * w, g, w, color=c, alpha=0.45, label=f"{rg} GBT" if i == 0 else None)
    ax.bar(x + i * 2 * w - 2.5 * w + w, t, w, color=c, label=f"{rg} TabPFN" if i == 0 else None)
leg = [Patch(fc="#1f4e79", label="single"), Patch(fc="#4a90b8", label="pooled"),
       Patch(fc="grey", alpha=0.45, label="GBT (light)"), Patch(fc="grey", label="TabPFN (solid)")]
ax.legend(handles=leg, fontsize=7, ncol=2, loc="lower center")
ax.set_xticks(x); ax.set_xticklabels(sublab, fontsize=8, rotation=15)
ax.set_ylabel("AUROC (matched 10k train / 2k test)"); ax.set_ylim(0.68, 0.81)
ax.set_title("A. Big-6 regimes: TabPFN-3 (solid) vs GBT (light)", fontsize=9.5)

# Panel B: paired dot plot of all 24 deltas
ax = axes[1]
cells = []
for _, r in piv.reset_index().iterrows():
    cells.append((f"{r.substance[:6]}·{r.regime}", r.delta, r.regime))
for _, r in covp.reset_index().iterrows():
    cells.append((r.substance.replace('COV:', '')[:10] + "·cov", r.delta, "coverage"))
cells.sort(key=lambda z: z[1])
cmap = {"single": "#1f4e79", "pooled": "#4a90b8", "coverage": "#7ba05b"}
yv = np.arange(len(cells))
ax.barh(yv, [c[1] for c in cells], color=[cmap[c[2]] for c in cells])
ax.set_yticks(yv); ax.set_yticklabels([c[0] for c in cells], fontsize=6)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.mean([c[1] for c in cells]), color="grey", ls="--", lw=1,
           label=f"mean +{np.mean([c[1] for c in cells]):.3f}")
ax.set_xlabel("TabPFN-3 − GBT (AUROC)"); ax.set_title("B. All 24 cells: TabPFN-3 advantage", fontsize=9.5)
ax.legend(handles=[Patch(fc=cmap[k], label=k) for k in cmap] +
          [plt.Line2D([0], [0], color="grey", ls="--", label=f"mean +{np.mean([c[1] for c in cells]):.3f}")],
          fontsize=6.5, loc="lower right")
fig.suptitle("TabPFN-3 outperforms GBT in every matched-data comparison (24/24)", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig("figures/p1_tabpfn_vs_gbt.png", dpi=300, bbox_inches="tight")