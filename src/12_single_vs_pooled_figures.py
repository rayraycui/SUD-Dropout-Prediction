"""
Manuscript figures 1-3, rendered from the tables written by
src/11_single_vs_pooled_uncapped.py. No model fitting happens here.

  figures/fig1_single_vs_pooled.png   per-substance vs pooled AUROC, all 17 substances
  figures/fig2_pooling_delta.png      pooling effect with bootstrap intervals, 11 lower-volume substances
"""
import os

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOCAL, GREY, META = "#2f5f98", "#b9bcc2", "#6d727a"
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "legend.fontsize": 9.0, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": None,
})

SHORT = {"Methamphetamine": "Methamph.", "Other_opioids": "Oth. opioids",
         "Benzodiazepines": "Benzodiaz.", "Other_amphetamines": "Oth. amphet.",
         "Non_rx_methadone": "Non-Rx methadone", "Other_hallucinogens": "Oth. halluc.",
         "Other_stimulants": "Oth. stimulants", "Other_tranquilizers": "Oth. tranq.",
         "Other_sedatives_hypnotics": "Oth. sed./hyp.", "OTC": "OTC meds"}

os.makedirs("figures", exist_ok=True)

C = pd.read_csv("data/p1_plan_eval_17substance_gbt_vs_logistic.csv")
C = C.sort_values("test_n", ascending=False).reset_index(drop=True)
C["lab"] = [SHORT.get(s, s.replace("_", " ")) for s in C.substance]
S = C[C.group == "small"].reset_index(drop=True)

# ---- Figure 1: per-substance vs pooled, every substance ---------------------
fig, ax = plt.subplots(figsize=(7.16, 2.86))
y = np.arange(len(C))[::-1]
for yy, r in zip(y, C.itertuples()):
    ax.plot([r.gbt_single, r.gbt_pooled], [yy, yy], color="#d0d3d8", lw=1.4,
            zorder=1, solid_capstyle="round")
ax.scatter(C.gbt_single, y, s=30, color=GREY, zorder=3, label="one model per substance")
ax.scatter(C.gbt_pooled, y, s=30, color=FOCAL, zorder=3, label="one pooled model")
ax.set_yticks(y); ax.set_yticklabels(C.lab)
ax.set_xlabel("AUROC"); ax.margins(y=0.02)
ax.axhline(y[5] - 0.5, color="#9aa0a8", lw=0.8, ls=(0, (4, 3)), zorder=0)
ax.text(0.868, y[5] - 0.32, "six highest-volume substances above", fontsize=8.2,
        color=META, va="bottom", ha="right")
ax.set_title("One pooled model stays close to per-substance models, and wins on low volume",
             loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.885), frameon=False,
          handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
ax.set_xlim(0.700, 0.872)
fig.tight_layout()
fig.savefig("figures/fig1_single_vs_pooled.png")
plt.close(fig)

# ---- Figure 2: pooling effect with intervals, small classes -----------------
fig, ax = plt.subplots(figsize=(7.16, 2.68))
y = np.arange(len(S))[::-1]; off = 0.19
for yy, r in zip(y, S.itertuples()):
    ax.plot([r.gbt_ci_lo, r.gbt_ci_hi], [yy + off] * 2, color=FOCAL, lw=1.3, zorder=2)
    ax.plot([r.lr_ci_lo, r.lr_ci_hi], [yy - off] * 2, color=GREY, lw=1.3, zorder=2)
ax.scatter(S.gbt_delta, y + off, s=26, color=FOCAL, zorder=3, label="gradient boosting")
ax.scatter(S.lr_delta, y - off, s=26, color=GREY, zorder=3, label="logistic regression")
ax.axvline(0, color=META, lw=1.0, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([f"{l} ({n:,})" for l, n in zip(S.lab, S.train_n)])
ax.set_xlabel("Change in AUROC from pooling (pooled minus per-substance)")
ax.set_title("Whether pooling helps depends on how much data the substance has", loc="left")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.62), frameon=False,
          handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
ax.text(0.004, y[0] + 0.62, "pooling better \u2192", fontsize=8.2, color=META, ha="left")
ax.text(-0.004, y[0] + 0.62, "\u2190 per-substance better", fontsize=8.2, color=META, ha="right")
ax.set_xlim(-0.115, 0.168); ax.margins(y=0.07)
fig.tight_layout()
fig.savefig("figures/fig2_pooling_delta.png")
plt.close(fig)

print("wrote figures/fig1_single_vs_pooled.png, figures/fig2_pooling_delta.png")
