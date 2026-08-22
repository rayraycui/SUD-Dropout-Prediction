#!/usr/bin/env python3
"""
07c_age65_anomaly.py: diagnose the 65+ calibration gap reported in the subgroup audit.

Training assembly: the canonical joint-stratified holdout (JSH) defined in
src/jsh_fit.py. This script carries no sampling scheme and no seed of its own,
it scores the same fitted model as 07b, so the band it decomposes is exactly the
band 07b reports.

The unseen-class fairness audit (07b) finds one stratum that fails a deployability
criterion: the 65+ age band, where predicted dropout risk exceeds observed by 0.140
(n=5,061, observed 0.165 against a cohort rate of 0.404). This script establishes that
the gap is NOT an age effect but a localised reporting artifact, by decomposing the band
by state and by substance.

Result (see docs/RESULTS.md and data/p1_age65_anomaly_diagnosis.csv):

  - 64% of the entire 65+ unseen band is a single state x substance cell
    (STFIPS 36 x Other_tranquilizers, n=3,234 = 1.5% of the unseen cohort).
  - That cell's OBSERVED dropout is 0.067, against 0.277 for big-6 patients of the same
    age in the same state and 0.404 cohort-wide. The label, not the prediction, is the
    outlier: the model predicts 0.291 there, which the setting mix largely justifies
    (big-6 65+ dropout rates from the same state, applied to the cell's own setting mix,
    give ~0.29).
  - The pattern is absent from the training data, so it was not learnable: big-6 STFIPS 36
    65+ dropout is 0.277 overall and 0.478 in ambulatory settings, where 99.8% of the cell
    sits.
  - Excluding that one cell, the 65+ band gap is -0.010, a slight UNDER-prediction rather
    than the +0.140 over-prediction of the published band, and no age band exceeds 0.05.
    Cohort AUROC is essentially unchanged.

The likely mechanism (a program or state-level discharge-coding convention) cannot be
confirmed from the public-use file, which carries no program identifier. The manuscript
therefore reports the localisation, which is measurable, and not a mechanism.

Usage:  python src/07c_age65_anomaly.py
Inputs: data/teds_d_analysis_2015_2022.parquet
Output: data/p1_age65_anomaly_diagnosis.csv
Runtime: ~5 min (dominated by the shared JSH fit, identical to 07b)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "48")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jsh_fit import fit_jsh, score

DATA = "data/teds_d_analysis_2015_2022.parquet"
OUT = "data"

AGE65 = 12          # TEDS AGE code for the 65+ band
STATE = 36          # STFIPS of the state that dominates the band
SUBSTANCE = "Other_tranquilizers"
AMBULATORY = [7, 8]  # SERVICES codes: ambulatory intensive / non-intensive


def row(label, s):
    return dict(stratum=label, n=len(s), pred=round(s.p.mean(), 4),
                obs=round(s.y.mean(), 4), gap=round(s.p.mean() - s.y.mean(), 4))


def main():
    df = pd.read_parquet(DATA)
    clf, _holdout, uns, _cap, _train = fit_jsh(DATA)

    big6 = df[df.is_big6]
    uns = uns.copy()
    uns["p"] = score(clf, uns)
    print(f"unseen cohort n={len(uns)} AUROC={roc_auc_score(uns.y, uns.p):.4f}")

    band = uns[uns.AGE == AGE65]
    cell = band[(band.STFIPS == STATE) & (band.subname == SUBSTANCE)]
    rest = band.drop(cell.index)

    rows = [row("65+ band (as published)", band),
            row(f"65+ | state {STATE}", band[band.STFIPS == STATE]),
            row("65+ | all other states", band[band.STFIPS != STATE])]
    for sn in band.subname.value_counts().head(4).index:
        rows.append(row(f"65+ | {sn}", band[band.subname == sn]))
    rows += [row(f"65+ | state {STATE} x {SUBSTANCE}", cell),
             row("65+ band excluding that one cell", rest)]
    diag = pd.DataFrame(rows)
    diag.to_csv(f"{OUT}/p1_age65_anomaly_diagnosis.csv", index=False)
    print(diag.to_string(index=False))

    # was the pattern learnable from the training substances?
    b6band = big6[(big6.STFIPS == STATE) & (big6.AGE == AGE65)]
    b6amb = b6band[b6band.SERVICES.isin(AMBULATORY)]
    print(f"\nbig-6 state {STATE} 65+          : n={len(b6band):6d} dropout={b6band.y.mean():.4f}")
    print(f"big-6 state {STATE} 65+ ambulatory: n={len(b6amb):6d} dropout={b6amb.y.mean():.4f}")
    print(f"cell ambulatory share            : {cell.SERVICES.isin(AMBULATORY).mean()*100:.1f}%")

    # does the cell's setting mix justify the model's prediction?
    rates = b6band.groupby("SERVICES").y.mean()
    exp = cell.SERVICES.map(rates).mean()
    print(f"big-6 65+ setting rates at cell's setting mix = {exp:.4f} "
          f"(model predicted {cell.p.mean():.4f}, observed {cell.y.mean():.4f})")

    # sensitivity of the headline numbers
    ex = uns.drop(cell.index)
    band_gap = band.p.mean() - band.y.mean()
    print(f"\ncohort AUROC full={roc_auc_score(uns.y, uns.p):.4f} "
          f"excl_cell={roc_auc_score(ex.y, ex.p):.4f}")
    gaps = (ex.groupby("AGE")
              .apply(lambda g: g.p.mean() - g.y.mean(), include_groups=False))
    print(f"max |age-band gap| excluding cell = {gaps.abs().max():.4f} "
          f"(65+ band as published = {band_gap:+.4f}; "
          f"65+ excluding the cell = {rest.p.mean() - rest.y.mean():+.4f})")


if __name__ == "__main__":
    main()
