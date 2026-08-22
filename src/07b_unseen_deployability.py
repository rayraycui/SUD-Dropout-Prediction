#!/usr/bin/env python3
"""
07b_unseen_deployability.py: deployability of the pooled big-6 model on the twelve
unseen (held-out) substance classes.

Training assembly: the canonical joint-stratified holdout (JSH) defined in
src/jsh_fit.py. The cross-sectional arms below carry no sampling scheme and no
seed of their own; they score the shared fit without refitting, so the numbers
here come from the same model as the coverage, calibration and fairness arms.

  0. cohort-level discrimination summary           -> p1_unseen_summary.csv
  1. discrimination + calibration (decile curve)   -> p1_calibration_unseen_deciles.csv
  2. clinical utility (decision curve)             -> p1_dca_realdata_unseen.csv
  3. subgroup fairness (race / sex / age)          -> p1_fairness_unseen.csv
  4. COVID-era temporal transportability           -> p1_covid_temporal_unseen.csv

DELIBERATE EXCEPTION: covid_temporal() below is OUTSIDE JSH by decision, and is
not to be "fixed" into it. It is an intentional within-unseen replication: it
trains a separate model on the UNSEEN-CLASS 2015-2018 episodes only (n=125,912,
seed 42) and tests it on unseen-class episodes in 2019 / 2020 / 2022. Its purpose
is to ask whether the unseen cohort's own temporal decay is steeper than the
trained substances', which requires a model fitted inside that cohort; folding it
into JSH would replace the question rather than answer it. The manuscript
discloses that this one arm uses a separate within-unseen fit. Every other arm in
this file uses the shared JSH fit.

Figures 3b / 4b / 5b in the paper are rendered from these CSVs by
src/09b_paper_figures_unseen.py.

Usage:  python src/07b_unseen_deployability.py
Inputs: data/teds_d_analysis_2015_2022.parquet
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "48")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

from jsh_fit import fit_jsh, score, FEATS

DATA = "data/teds_d_analysis_2015_2022.parquet"
OUT = "data"

# --- within-unseen replication only (see DELIBERATE EXCEPTION above) ----------
REPLICATION_SEED = 42
REPLICATION_GBT = dict(max_iter=200, learning_rate=0.1, max_bins=255,
                       random_state=REPLICATION_SEED)


def calibration_deciles(y, p, n_bins=10):
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"decile": q, "pred": p, "obs": y})
             .groupby("decile")
             .agg(mean_pred=("pred", "mean"), obs_rate=("obs", "mean"), n=("obs", "size"))
             .reset_index())
    cal["abs_dev"] = (cal.mean_pred - cal.obs_rate).abs()
    return cal


def decision_curve(y, p, thresholds=np.arange(0, 1.01, 0.01)):
    """Net benefit of the model, treat-all, and treat-none at each threshold."""
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        flag = p >= t
        tp = np.sum(flag & (y == 1)) / n
        fp = np.sum(flag & (y == 0)) / n
        w = t / (1 - t) if t < 1 else np.inf
        nb_model = tp - fp * w if np.isfinite(w) else 0.0
        nb_all = prev - (1 - prev) * w if np.isfinite(w) else 0.0
        rows.append(dict(threshold=round(float(t), 2), nb_model=nb_model,
                         nb_treat_all=nb_all, nb_treat_none=0))
    return pd.DataFrame(rows)


def fairness_strata(df, y, p, dims=("RACE", "GENDER", "AGE"), min_n=200):
    rows = []
    for dim in dims:
        for cat, sub in df.groupby(dim).groups.items():
            m = df[dim].to_numpy() == cat
            if m.sum() < min_n or len(np.unique(y[m])) < 2:
                continue
            rows.append(dict(Stratification=dim, Category=cat, n=int(m.sum()),
                             dropout_rate=float(y[m].mean()),
                             AUROC=float(roc_auc_score(y[m], p[m])),
                             calibration_error=float(abs(p[m].mean() - y[m].mean()))))
    return pd.DataFrame(rows)


def covid_temporal(unseen):
    """Within-unseen replication (OUTSIDE JSH, retained by decision).

    Trains on 2015-2018 UNSEEN-CLASS episodes only and tests per era
    (2019/2020/2022). This is deliberately a separate fit with its own seed
    (42): the question is whether the unseen cohort decays faster than the
    trained substances do, which cannot be asked of a model trained on big-6
    episodes. Do not repoint this at the shared JSH fit.
    """
    tr = unseen[unseen.YEAR <= 2018]
    clf = HistGradientBoostingClassifier(**REPLICATION_GBT)
    clf.fit(tr[FEATS].to_numpy(dtype=np.float32), tr.y.to_numpy())
    rows = []
    for yr in (2019, 2020, 2022):
        te = unseen[unseen.YEAR == yr]
        pr = clf.predict_proba(te[FEATS].to_numpy(dtype=np.float32))[:, 1]
        rows.append(dict(Year=yr, n=len(te), dropout_rate=float(te.y.mean()),
                         AUROC=float(roc_auc_score(te.y.to_numpy(), pr))))
    return pd.DataFrame(rows), len(tr)


def main():
    clf, _holdout, unseen, _cap, _train = fit_jsh(DATA)
    unseen = unseen.copy()
    y = unseen.y.to_numpy()
    p = score(clf, unseen)

    auroc = float(roc_auc_score(y, p))
    summary = pd.DataFrame([dict(n=len(y), AUROC=auroc,
                                 Brier=float(brier_score_loss(y, p)),
                                 base_rate=float(y.mean()))])
    summary.to_csv(f"{OUT}/p1_unseen_summary.csv", index=False)
    print(f"unseen n={len(y)} AUROC={auroc:.4f} "
          f"Brier={summary.Brier.iloc[0]:.4f} base={summary.base_rate.iloc[0]:.4f}")

    cal = calibration_deciles(y, p)
    cal.to_csv(f"{OUT}/p1_calibration_unseen_deciles.csv", index=False)
    print(f"calibration mean|dev|={cal.abs_dev.mean():.4f} max={cal.abs_dev.max():.4f}")

    dca = decision_curve(y, p)
    dca.to_csv(f"{OUT}/p1_dca_realdata_unseen.csv", index=False)

    fair = fairness_strata(unseen, y, p)
    fair.to_csv(f"{OUT}/p1_fairness_unseen.csv", index=False)
    print(f"fairness strata={len(fair)} AUROC {fair.AUROC.min():.4f}-{fair.AUROC.max():.4f} "
          f"calib max={fair.calibration_error.max():.4f}")

    cov, n_tr = covid_temporal(unseen)
    cov.to_csv(f"{OUT}/p1_covid_temporal_unseen.csv", index=False)
    print(f"covid train n={n_tr} (within-unseen replication, seed {REPLICATION_SEED}, outside JSH)")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
