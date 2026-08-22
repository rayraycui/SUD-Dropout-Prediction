#!/usr/bin/env python3
"""
jsh_fit.py, the canonical "joint-stratified holdout" (JSH) assembly.

One fit, reused by every deployability arm. The big-6 episodes are split 70/30
stratified jointly on (subname x y); the training side is then subsampled to
1,000,000 episodes on the same joint key, preserving each substance x label
cell's share exactly. The twelve unseen substance classes are never trained on,
so the full unseen cohort is available as a test set.

Features enter as raw TEDS integer codes: no scaling, no imputation, no
one-hot, no rebalancing. HistGradientBoostingClassifier handles the NaNs in the
optional-item columns natively.

Usage:
    from jsh_fit import fit_jsh, FEATS, SEED
    clf, holdout, unseen, cap, train = fit_jsh("data/teds_d_analysis_2015_2022.parquet")

Returns:
    clf: fitted HistGradientBoostingClassifier
    holdout: 1,950,373-row big-6 test frame (30% of big-6)
    unseen: 215,983-row frame of the twelve non-big-6 classes
    cap: 999,999-row training subsample actually fitted
    train: 4,550,870-row big-6 train side before subsampling
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

FEATS = ["SUB2", "SUB3", "ROUTE1", "FREQ1", "FRSTUSE1", "SERVICES", "DSMCRIT", "NOPRIOR",
         "PSOURCE", "AGE", "GENDER", "RACE", "EDUC", "EMPLOY", "STFIPS"]
SEED = 0
TEST_SIZE = 0.30
TRAIN_CAP = 1_000_000
GBT = dict(max_iter=200, learning_rate=0.1, max_bins=255, random_state=SEED)

N_TRAIN = 4_550_870
N_HOLDOUT = 1_950_373
N_CAP = 999_999


def fit_jsh(parquet_path, verbose=True):
    """Fit the canonical JSH model. Returns (clf, holdout_big6, unseen, cap, train)."""
    df = pd.read_parquet(parquet_path)

    big6 = df[df.is_big6]
    unseen = df[~df.is_big6].copy()

    key = big6.subname.astype(str) + "_" + big6.y.astype(str)
    tr, te = train_test_split(big6, test_size=TEST_SIZE, random_state=SEED, stratify=key)

    k2 = tr.subname.astype(str) + "_" + tr.y.astype(str)
    cap = tr.groupby(k2, group_keys=False).apply(
        lambda g: g.sample(int(round(TRAIN_CAP * len(g) / len(tr))), random_state=SEED),
        include_groups=True)

    assert len(tr) == N_TRAIN, f"train side {len(tr)} != {N_TRAIN}"
    assert len(te) == N_HOLDOUT, f"holdout {len(te)} != {N_HOLDOUT}"
    assert len(cap) == N_CAP, f"train subsample {len(cap)} != {N_CAP}"

    clf = HistGradientBoostingClassifier(**GBT)
    clf.fit(cap[FEATS].to_numpy(np.float32), cap.y.to_numpy())

    if verbose:
        print(f"JSH fit: train={len(tr):,} -> cap={len(cap):,} | "
              f"holdout={len(te):,} | unseen={len(unseen):,}")
    return clf, te, unseen, cap, tr


def score(clf, frame):
    """Positive-class probabilities for a frame, using the canonical feature order."""
    return clf.predict_proba(frame[FEATS].to_numpy(np.float32))[:, 1]


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/teds_d_analysis_2015_2022.parquet"
    fit_jsh(p)
