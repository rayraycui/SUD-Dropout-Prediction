#!/usr/bin/env python3
"""
07a_fairness_score.py: score the trained-substance holdout under the canonical
fit and save per-episode predictions for the subgroup fairness audit (07).
Produces data/fairness_preds.parquet.

Training assembly: the canonical joint-stratified holdout (JSH) defined in
src/jsh_fit.py. This script carries no sampling scheme and no seed of its own.

Provenance: the original fairness_preds.parquet was written by an interactive
cell that left no code lineage. This script reconstructs that cell on the shared
JSH fit (train n=999,999, test n=1,950,373, overall AUROC 0.7914).

Usage:  python src/07a_fairness_score.py
Inputs: data/teds_d_analysis_2015_2022.parquet
Output: data/fairness_preds.parquet
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "48")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.metrics import roc_auc_score

from jsh_fit import fit_jsh, score

DATA = "data/teds_d_analysis_2015_2022.parquet"

clf, holdout, _unseen, cap, _train = fit_jsh(DATA)
te = holdout.copy()
te["p"] = score(clf, te)
print(f"trained pooled GBT n={len(cap):,}; fairness test set n={len(te):,}")
print("overall test AUROC:", round(roc_auc_score(te.y, te.p), 4))
te[["y", "p", "RACE", "GENDER", "AGE", "STFIPS", "subname"]].to_parquet("data/fairness_preds.parquet")
print("saved data/fairness_preds.parquet")
