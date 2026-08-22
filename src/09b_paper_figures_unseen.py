#!/usr/bin/env python3
"""
09b_paper_figures_unseen.py: renders the three unseen-class deployability figures
embedded in the manuscript (Figs. 3, 4, 5):

    paper/fig3b_calib_dca_unseen.png   calibration + decision curve
    paper/fig4b_covid_unseen.png       COVID-era transportability, trained vs unseen
    paper/fig5b_fairness_unseen.png    subgroup AUROC + calibration gap

Reads only committed CSVs written by src/07b_unseen_deployability.py (and
data/p1_covid_temporal.csv for the trained-substance reference series), so the
figures regenerate without refitting any model. The cohort AUROC used as the
reference line in Fig. 5b comes from data/p1_unseen_summary.csv.

Usage:  python src/09b_paper_figures_unseen.py
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ---- style ------------------------------------------------------------------
# Font ladder / open frame / embedded fonts, matching the figure-style ladder
# (base 8, secondary 7, tick 6). Set explicitly so the script is standalone.
FOCAL, MUTED, ALARM, META_GREY = "#1b4f8f", "#8a8f98", "#b3452c", "#888888"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 16,
    "axes.labelsize": 16, "axes.titlesize": 16, "legend.fontsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.labelweight": "normal", "lines.linewidth": 2.2,
    "patch.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
})


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
    texts = [(t, _quad(t, r)) for t in fig.findobj(mpl.text.Text)
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


def panel_letter(ax, letter, dx=-0.18, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=18, va="bottom", ha="left")


def goodness_arrow(ax, text="higher = better", x=0.02, ha="left"):
    ax.text(x, 0.98, "\u2191 " + text, transform=ax.transAxes,
            fontsize=14, color=META_GREY, ha=ha, va="top")


AGE_LAB = {1: "12\u201314", 2: "15\u201317", 3: "18\u201320", 4: "21\u201324", 5: "25\u201329",
           6: "30\u201334", 7: "35\u201339", 8: "40\u201344", 9: "45\u201349", 10: "50\u201354",
           11: "55\u201364", 12: "65+"}
# RACE codes per src/07_fairness_audit.py (TEDS-D RACE); 3/13/20/21 absent from the
# unseen-class strata (below the n>=200 reporting floor) so they are not mapped here.
# RACE codes and their meanings are taken verbatim from RACE_MAP in
# src/07_fairness_audit.py (TEDS-D RACE). Code 5 is WHITE; code 7 is "Other
# single race". Codes 3/13/20/21 are absent from the unseen-class strata (below
# the n>=200 reporting floor), so only the codes that appear are shortened here.
RACE_MAP = {1: "Alaska Native", 2: "Am Indian", 3: "Asian/PI", 4: "Black", 5: "White",
            6: "Asian", 7: "Other single", 8: "Two+ races", 9: "NHOPI",
            13: "multi", 20: "Hispanic-only(NR)", 21: "Other"}
RACE_SHORT = {1: "Alaska Native", 2: "Am. Indian", 3: "Asian/PI", 4: "Black",
              5: "White", 6: "Asian", 7: "Other single race", 8: "Two or more races",
              9: "Native Hawaiian/PI"}
# Guard against silent relabelling: each shortened label is pinned to the audit's
# label for the same code. Code 5 must stay WHITE.
_RACE_EQUIV = {1: "Alaska Native", 2: "Am Indian", 3: "Asian/PI", 4: "Black",
               5: "White", 6: "Asian", 7: "Other single", 8: "Two+ races",
               9: "NHOPI"}
assert {k: RACE_MAP[k] for k in RACE_SHORT} == _RACE_EQUIV, \
    "RACE_SHORT/RACE_MAP drifted from src/07_fairness_audit.py"
assert RACE_SHORT[5] == "White" and RACE_MAP[5] == "White", "code 5 must be White"
SEX_LAB = {1: "Male", 2: "Female"}


# ------------------------------------------------------------------ Fig 3b
def fig_calibration_dca():
    cal = pd.read_csv("data/p1_calibration_unseen_deciles.csv")
    dca = pd.read_csv("data/p1_dca_realdata_unseen.csv")
    d = dca[dca.threshold <= 0.80]

    fig, (a1, b1) = plt.subplots(1, 2, figsize=(7.0, 3.20))
    a1.plot([0, 1], [0, 1], color=META_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    a1.plot(cal.mean_pred, cal.obs_rate, "-o", color=FOCAL, lw=1.6, ms=4.5, zorder=3)
    a1.text(0.60, 0.28, "perfect\ncalibration", color=META_GREY, ha="left", va="top", fontsize=12)
    a1.set_xlabel("Mean predicted probability", labelpad=9)
    a1.set_ylabel("Observed dropout rate", labelpad=8)
    a1.set_title(f"Calibrated within {cal.abs_dev.max():.3f}", loc="left")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.set_aspect("equal")
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    b1.axhline(0, color=META_GREY, lw=0.9, ls=":", zorder=1)
    b1.plot(d.threshold, d.nb_treat_all, color="#8a8f98", lw=1.3, ls="--", zorder=2)
    b1.plot(d.threshold, d.nb_model, color=FOCAL, lw=2.0, zorder=3)
    b1.set_xlim(0, 0.80); b1.set_ylim(-0.25, 0.46)
    b1.set_xlabel("Decision threshold", labelpad=6)
    b1.set_ylabel("Net benefit", labelpad=4)
    b1.set_title("Net benefit positive to 0.80", loc="left")
    b1.text(0.53, 0.155, "model", color=FOCAL, fontsize=14, fontweight="bold", ha="left", va="bottom")
    b1.text(0.575, -0.215, "treat all", color="#6d727a", fontsize=14, ha="left", va="bottom")
    b1.text(0.02, -0.235, "treat none", color=META_GREY, fontsize=12, ha="left", va="bottom")
    goodness_arrow(b1, x=0.98, ha="right")
    panel_letter(a1, "a"); panel_letter(b1, "b")
    fig.tight_layout(w_pad=1.8)
    fig.savefig("paper/fig3b_calib_dca_unseen.png", dpi=300, bbox_inches="tight")
    worst = cal.loc[cal.abs_dev.idxmax()]
    print(f"fig3b: calib mean|dev|={cal.abs_dev.mean():.4f} max={cal.abs_dev.max():.4f} "
          f"(decile {int(worst.decile)}, mean_pred {worst.mean_pred:.4f} vs obs {worst.obs_rate:.4f})")
    print("fig3b deciles  mean_pred:", [round(v, 4) for v in cal.mean_pred])
    print("fig3b deciles  obs_rate :", [round(v, 4) for v in cal.obs_rate])
    print(f"fig3b DCA: nb_model at 0.0/0.4/0.8 = "
          f"{[round(float(d[d.threshold == t].nb_model.iloc[0]), 4) for t in (0.0, 0.4, 0.8)]}")
    return overlap_count(fig, "fig3b")


# ------------------------------------------------------------------ Fig 4b
def fig_covid():
    big = pd.read_csv("data/p1_covid_temporal.csv")
    uns = pd.read_csv("data/p1_covid_temporal_unseen.csv")
    big_auc, big_lo, big_hi = big.auc.values, big.ci_lo.values, big.ci_hi.values
    uns_auc = list(uns.AUROC)
    drops_big = [big_auc[0] - big_auc[1], big_auc[1] - big_auc[2]]
    drops_uns = [uns_auc[0] - uns_auc[1], uns_auc[1] - uns_auc[2]]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.70))
    x = np.arange(3)
    axA.fill_between([0.5, 2.4], 0.69, 0.81, color="#f0ecdf", zorder=0)
    # Left-aligned just inside the shaded band (band spans data x 0.5..2.4 = 216.9 px, the
    # label is 170.5 px) so the whole label sits inside the era it names. Centring it at
    # x=1.45 fits the band but runs into the "unseen classes" label, which starts at 503 px;
    # anchoring at x=0.55 ends the label at ~494 px, clear of it and inside the band.
    axA.text(0.55, 0.6925, "COVID era", color="#8a7f5c", fontsize=12, ha="left", va="bottom")
    axA.errorbar(x, big_auc, yerr=[big_auc - big_lo, big_hi - big_auc],
                 fmt="-o", color=MUTED, lw=1.4, ms=4.5, capsize=2.5, zorder=3)
    axA.plot(x, uns_auc, "-o", color=FOCAL, lw=1.8, ms=5, zorder=4)
    axA.text(2.12, uns_auc[2] - 0.001, "unseen\nclasses", color=FOCAL, fontsize=13,
             fontweight="bold", ha="left", va="center")
    axA.text(2.12, big_auc[2] + 0.003, "trained\nsubstances", color="#6d727a",
             fontsize=13, ha="left", va="center")
    axA.set_xticks(x)
    axA.set_xticklabels(["2019", "2020", "2022"])
    axA.set_xlim(-0.35, 3.05); axA.set_ylim(0.69, 0.81)
    axA.set_ylabel("AUROC")
    axA.set_title("Unseen classes decay more", loc="left")
    goodness_arrow(axA)

    w = 0.34; xb = np.arange(2)
    axB.axhline(0, color=META_GREY, lw=0.9, zorder=1)
    axB.bar(xb - w / 2, drops_big, width=w, color=MUTED, zorder=3)
    axB.bar(xb + w / 2, drops_uns, width=w, color=FOCAL, zorder=3)
    # A near-zero drop (the 2020->2022 trained-substance step is 0.0008) renders as
    # a sub-pixel bar that reads as an empty slot. Give it a visible stub at the
    # baseline so the value is present as a mark, not only as its label.
    stub = 0.00045
    for xi, v in zip(xb - w / 2, drops_big):
        if 0 < v < stub * 2:
            axB.bar([xi], [stub], width=w, color=MUTED, zorder=3)
    # No per-bar value labels here. This panel has only four bars in a half-column
    # canvas, and one of them (trained substances, 2020->2022) is 0.001 tall: any
    # label above it lands on the 30x taller unseen bar beside it, and moving the
    # pair onto a shared row above the taller bar collides them with each other.
    # The y-axis ticks resolve every bar to the precision the claim needs, and the
    # exact values appear in the body text, so a legend carries the series identity.
    axB.legend(handles=[Patch(facecolor="#b9bcc2", label="trained subst."),
                        Patch(facecolor=FOCAL, label="unseen classes")],
               loc="upper right", frameon=False, fontsize=12,
               handlelength=1.0, handletextpad=0.5, labelspacing=0.25,
               borderpad=0.0, borderaxespad=0.2)
    axB.set_xticks(xb)
    axB.set_xticklabels(["onset", "post-acute"])
    axB.set_ylim(0, 0.088); axB.set_xlim(-0.62, 1.62)
    axB.set_ylabel("AUROC lost")
    assert drops_uns[0] > drops_uns[1] and drops_big[0] > drops_big[1], \
        "panel-b title asserts onset > post-acute for both series"
    axB.set_title("Onset drop is larger", loc="left")
    # No direction cue in this panel. The legend holds the upper right, the bars fill
    # the lower half, and every remaining corner is too tight; lengthening the axis
    # label to carry the cue overflowed the half-column canvas. The panel title
    # ("Onset drop is larger") already states the direction the reader needs.
    panel_letter(axA, "a"); panel_letter(axB, "b")
    fig.tight_layout(w_pad=3.0)
    fig.savefig("paper/fig4b_covid_unseen.png", dpi=300, bbox_inches="tight")
    print(f"fig4b: trained={[round(float(v), 4) for v in big_auc]} "
          f"drops={[round(float(v), 4) for v in drops_big]}")
    print(f"fig4b: unseen ={[round(float(v), 4) for v in uns_auc]} "
          f"drops={[round(float(v), 4) for v in drops_uns]}")
    print(f"fig4b: bar labels drawn = "
          f"trained {[f'{v:.3f}' for v in drops_big]} unseen {[f'{v:.3f}' for v in drops_uns]}")
    return overlap_count(fig, "fig4b")


# ------------------------------------------------------------------ Fig 5b
def fig_fairness(overall):
    f = pd.read_csv("data/p1_fairness_unseen.csv")

    # Category arrives either as the raw TEDS integer code or as the audit's own
    # label text, depending on which revision of 07b wrote the file. Normalise to
    # the integer code first, then apply the display labels, so code 5 resolves to
    # White by either route.
    AGE_CODE = {v: k for k, v in AGE_LAB.items()}
    AGE_CODE.update({v.replace("\u2013", "-"): k for k, v in AGE_LAB.items()})
    RACE_CODE = {v: k for k, v in RACE_MAP.items()}
    SEX_CODE = {v: k for k, v in SEX_LAB.items()}

    def to_code(strat, cat):
        s = str(cat).strip()
        if s.lstrip("-").isdigit():
            return int(s)
        table = {"AGE": AGE_CODE, "RACE": RACE_CODE, "GENDER": SEX_CODE}[strat]
        if s not in table:
            raise KeyError(f"unrecognised {strat} category {cat!r}")
        return table[s]

    f["code"] = [to_code(s, c) for s, c in zip(f.Stratification, f.Category)]
    f["label"] = [AGE_LAB[c] if s == "AGE"
                  else (RACE_SHORT[c] if s == "RACE" else SEX_LAB[c])
                  for s, c in zip(f.Stratification, f.code)]
    age = f[f.Stratification == "AGE"].sort_values("code")
    race = f[f.Stratification == "RACE"].sort_values("AUROC")
    sex = f[f.Stratification == "GENDER"]
    order = list(age.label) + list(race.label) + list(sex.label)
    auc_all = list(age.AUROC) + list(race.AUROC) + list(sex.AUROC)
    ce_all = list(age.calibration_error) + list(race.calibration_error) + list(sex.calibration_error)
    cols = [FOCAL] * len(age) + [MUTED] * len(race) + ["#3f7cb5"] * len(sex)
    bcols = [ALARM if v > 0.05 else MUTED for v in ce_all]
    i65 = order.index("65+"); ce65 = ce_all[i65]
    n65 = int(age[age.code == 12].n.iloc[0])

    fig = plt.figure(figsize=(7.0, 3.00))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.34)
    axT = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1], sharex=axT)

    axT.axhline(overall, color=META_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axT.scatter(range(len(auc_all)), auc_all, s=22, c=cols, zorder=3)
    axT.set_ylabel("AUROC"); axT.set_ylim(0.66, 0.845); axT.set_xlim(-0.7, len(order) - 0.3)
    # Title is checked against the values it asserts (figure-style S1.4): the five
    # lowest-AUROC strata must be the two youngest age bands and the three smallest
    # race groups, otherwise the claim is rewritten rather than shipped.
    lowest5 = {order[i] for i in np.argsort(auc_all)[:5]}
    assert lowest5 == {"12\u201314", "15\u201317", "Alaska Native",
                       "Native Hawaiian/PI", "Am. Indian"}, f"title contradicted: {lowest5}"
    axT.set_title("Stable but for youngest, smallest groups", loc="left")
    axT.text(-0.62, overall + 0.003, f"cohort {overall:.3f}", color=META_GREY,
             fontsize=11, ha="left", va="bottom")
    axT.text(len(age) / 2 - 0.5, 0.836, "age band", color=FOCAL, fontsize=13,
             ha="center", va="top", fontweight="bold")
    axT.text(len(age) + len(race) / 2 - 0.5, 0.836, "race/ethnicity", color="#6d727a",
             fontsize=13, ha="center", va="top")
    axT.text(len(age) + len(race) + 0.5, 0.836, "sex", color="#3f7cb5", fontsize=13,
             ha="center", va="top", fontweight="bold")
    axT.tick_params(labelbottom=False); axT.set_yticks([0.70, 0.75, 0.80])

    axB.bar(range(len(ce_all)), ce_all, color=bcols, width=0.7, zorder=3)
    axB.axhline(0.05, color=META_GREY, lw=0.9, ls=":", zorder=1)
    axB.set_xticks(range(len(order))); axB.set_xticklabels(order, rotation=45, ha="right")
    axB.set_ylabel("|pred \u2212 obs|"); axB.set_ylim(0, 0.172)
    axB.set_yticks([0, 0.05, 0.10, 0.15])
    assert max(c for i, c in enumerate(ce_all) if i != i65) < 0.05, \
        "title asserts 65+ is the only band above 0.05"
    axB.set_title("Calibration holds except 65+", loc="left")
    axB.annotate(f"65+ : {ce65:.3f} (n={n65:,}, over-predicts)", xy=(i65 + 0.4, ce65),
                 xytext=(i65 + 1.2, ce65 + 0.021), fontsize=13, color=ALARM,
                 va="top", ha="left",
                 arrowprops=dict(arrowstyle="-", color=ALARM, lw=0.8, shrinkA=0, shrinkB=2))
    axB.text(len(ce_all) - 0.4, 0.053, "0.05", color=META_GREY, fontsize=12, ha="right", va="bottom")
    panel_letter(axT, "a"); panel_letter(axB, "b")
    fig.savefig("paper/fig5b_fairness_unseen.png", dpi=300, bbox_inches="tight")
    print(f"fig5b: cohort reference AUROC={overall:.4f}; strata AUROC "
          f"{min(auc_all):.4f}-{max(auc_all):.4f}; calib max={max(ce_all):.4f}")
    print(f"fig5b: 65+ gap={ce65:.4f} (n={n65:,}); next largest="
          f"{max(c for i, c in enumerate(ce_all) if i != i65):.4f}")
    for lab, a_, c_ in zip(order, auc_all, ce_all):
        print(f"   {lab:<20s} AUROC={a_:.4f}  |pred-obs|={c_:.4f}")
    return overlap_count(fig, "fig5b")


if __name__ == "__main__":
    OVERALL = float(pd.read_csv("data/p1_unseen_summary.csv").AUROC.iloc[0])
    n_ov = fig_calibration_dca() + fig_covid() + fig_fairness(OVERALL)
    print(f"total text overlaps across fig3b/fig4b/fig5b = {n_ov}")
    if n_ov:
        raise SystemExit("text overlaps detected")
