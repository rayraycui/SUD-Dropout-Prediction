#!/usr/bin/env python3
"""
jsh_year_restricted.py, the year-restricted variant of the canonical
joint-stratified holdout (JSH).

Identical to src/jsh_fit.py in every respect that matters: same 15 features,
same joint (subname x y) stratification for the 1,000,000-row cap, same
HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_bins=255,
random_state=0), with exactly one difference: the big-6 training pool is
filtered by YEAR *before* the cap is applied.

Why there is no 70/30 holdout here. In the cross-sectional arms the test set is
drawn from the same years as the training pool, so JSH holds out 30% to keep the
two disjoint. In the temporal arms the test set is a *different year* (05: the
unseen classes in 2022; 06: big-6 episodes in 2019/2020/2022), and the training
pool is filtered to exclude those years, so train and test are already disjoint
by construction. Splitting again would only discard training data for no gain,
so the whole year-restricted pool is capped and used. Verified: this
specification reproduces the committed data/p1_covid_temporal.csv
(0.7863 / 0.7689 / 0.7681) and all twelve rows of
data/p1_coverage_temporal2022.csv exactly.

Usage:
    from jsh_year_restricted import fit_jsh_years, score
    clf, cap, pool = fit_jsh_years(df, df.YEAR < 2022)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from jsh_fit import FEATS, GBT, SEED, TRAIN_CAP, score  # noqa: F401  (score re-exported)


def cap_joint(pool, cap=TRAIN_CAP, seed=SEED):
    """Subsample `pool` to ~`cap` rows, preserving the joint (subname x y) mix.

    Same key and same rounding as src/jsh_fit.py, so the composition of the
    training set is the canonical one restricted to the retained years.
    """
    key = pool.subname.astype(str) + "_" + pool.y.astype(str)
    return pool.groupby(key, group_keys=False).apply(
        lambda g: g.sample(int(round(cap * len(g) / len(pool))), random_state=seed),
        include_groups=True)


def fit_jsh_years(df, year_mask, verbose=True, label=""):
    """Fit the canonical estimator on big-6 episodes selected by `year_mask`.

    Parameters
    ----------
    df : the full analytic table (must carry is_big6, subname, y, YEAR, FEATS)
    year_mask : boolean Series aligned to df, e.g. ``df.YEAR < 2022``
    label : text appended to the progress line, for the caller's log

    Returns
    -------
    (clf, cap, pool), the fitted model, the capped training frame, and the
    year-restricted pool it was drawn from.
    """
    pool = df[df.is_big6 & year_mask]
    cap = cap_joint(pool)
    clf = HistGradientBoostingClassifier(**GBT)
    clf.fit(cap[FEATS].to_numpy(np.float32), cap.y.to_numpy())
    if verbose:
        print(f"year-restricted JSH {label}: pool={len(pool):,} -> train={len(cap):,}",
              flush=True)
    return clf, cap, pool
