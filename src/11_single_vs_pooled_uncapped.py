"""
Single-substance vs pooled models across all 17 substances, uncapped.

Design (supersedes the capped joint-stratified holdout used in earlier versions):
  1. Split every one of the 17 substances 70/30, stratified on outcome (seed 0).
  2. Fit one model per substance on its own 70% (no subsampling).
  3. Pool the 70% training halves of the six highest-volume substances, fit ONE model on
     all 4,550,867 rows, and score it on all 17 held-out test sets.

Both model families use the same specification as src/02_baselines_single_pooled.py.

Writes:
  data/p1_plan_eval_17substance.csv                 gradient boosting
  data/p1_plan_eval_17substance_logistic.csv        logistic regression
  data/p1_plan_eval_17substance_gbt_vs_logistic.csv head-to-head

Runtime: about 20 minutes on 48 CPU cores (the uncapped pooled fits dominate).
"""
import os
os.environ["OMP_NUM_THREADS"] = "48"
import numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RNG = 0
SEED_MODEL = 42
BOOT_N = 2000
FEAT15 = ["SUB2", "SUB3", "ROUTE1", "FREQ1", "FRSTUSE1", "SERVICES", "DSMCRIT",
          "NOPRIOR", "PSOURCE", "AGE", "GENDER", "RACE", "EDUC", "EMPLOY", "STFIPS"]

df = pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
sizes = df.groupby("subname").size().sort_values(ascending=False)
substances = list(sizes.index)
big6 = list(df.loc[df.is_big6, "subname"].unique())
small = [s for s in substances if s not in big6]


def gbt():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                          max_bins=255, random_state=SEED_MODEL)


def logit():
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore",
                                                  min_frequency=50), FEAT15)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(penalty="l2", C=1.0, solver="saga",
                                                max_iter=200, n_jobs=48))])


def Xg(idx):
    return df.loc[idx, FEAT15]


def Xl(idx):
    return df.loc[idx, FEAT15].astype("Int64").astype(str)


def boot_delta(y, pa, pb, seed=0):
    """95% percentile interval on the paired AUROC difference (pooled minus single)."""
    r = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    n, out = len(y), np.empty(BOOT_N)
    for i in range(BOOT_N):
        ix = r.integers(0, n, n)
        if len(np.unique(y[ix])) < 2:
            out[i] = np.nan
            continue
        out[i] = roc_auc_score(y[ix], pa[ix]) - roc_auc_score(y[ix], pb[ix])
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


# ---- splits: every substance 70/30, stratified on outcome -------------------
train_idx, test_idx = {}, {}
for s in substances:
    d = df[df.subname == s]
    a, b = train_test_split(d.index.values, test_size=0.30,
                            random_state=RNG, stratify=d["y"].values)
    train_idx[s], test_idx[s] = a, b
pool_idx = np.concatenate([train_idx[s] for s in big6])
print(f"pooled training set (uncapped): {len(pool_idx):,}")

rows = {}
for fam, mk, prep in [("gbt", gbt, Xg), ("lr", logit, Xl)]:
    t0 = time.time()
    pooled = mk().fit(prep(pool_idx), df.loc[pool_idx, "y"].values)
    print(f"{fam}: pooled fit {time.time() - t0:.0f}s")

    for s in substances:
        te, y = test_idx[s], df.loc[test_idx[s], "y"].values
        p_pool = pooled.predict_proba(prep(te))[:, 1]
        single = mk().fit(prep(train_idx[s]), df.loc[train_idx[s], "y"].values)
        p_single = single.predict_proba(prep(te))[:, 1]

        rec = {"substance": s,
               "group": "BIG6" if s in big6 else "small",
               "train_n": len(train_idx[s]), "test_n": len(te),
               f"{fam}_single": roc_auc_score(y, p_single),
               f"{fam}_pooled": roc_auc_score(y, p_pool)}
        rec[f"{fam}_delta"] = rec[f"{fam}_pooled"] - rec[f"{fam}_single"]
        if s in small:                      # paired interval only where n allows
            lo, hi = boot_delta(y, p_pool, p_single)
            rec[f"{fam}_ci_lo"], rec[f"{fam}_ci_hi"] = lo, hi
            rec[f"{fam}_verdict"] = ("pooled better" if lo > 0 else
                                     "single better" if hi < 0 else "ns")
        rows.setdefault(s, {}).update(rec)
        print(f"  {fam} {s:26s} done ({time.time() - t0:.0f}s)")

res = pd.DataFrame([rows[s] for s in substances])
res["gbt_minus_lr_single"] = res.gbt_single - res.lr_single
res["gbt_minus_lr_pooled"] = res.gbt_pooled - res.lr_pooled

# per-family tables, then the head-to-head
(res.rename(columns={"gbt_single": "single_auroc", "gbt_pooled": "pooled_auroc_uncapped",
                     "gbt_delta": "pooled_minus_single", "gbt_ci_lo": "boot_ci_lo",
                     "gbt_ci_hi": "boot_ci_hi", "gbt_verdict": "verdict"})
   [["substance", "group", "train_n", "test_n", "single_auroc", "pooled_auroc_uncapped",
     "pooled_minus_single", "boot_ci_lo", "boot_ci_hi", "verdict"]]
   .to_csv("data/p1_plan_eval_17substance.csv", index=False))

(res.rename(columns={"lr_single": "single", "lr_pooled": "pooled_uncapped",
                     "lr_delta": "pooled_minus_single", "lr_ci_lo": "lo",
                     "lr_ci_hi": "hi", "lr_verdict": "sig"})
   [["substance", "group", "train_n", "test_n", "single", "pooled_uncapped",
     "pooled_minus_single", "lo", "hi", "sig"]]
   .to_csv("data/p1_plan_eval_17substance_logistic.csv", index=False))

res[["substance", "group", "train_n", "test_n",
     "gbt_single", "gbt_pooled", "gbt_delta", "gbt_verdict",
     "lr_single", "lr_pooled", "lr_delta", "lr_verdict",
     "gbt_minus_lr_single", "gbt_minus_lr_pooled"]].to_csv(
    "data/p1_plan_eval_17substance_gbt_vs_logistic.csv", index=False)

wt = res.test_n / res.test_n.sum()
print("SAVED", len(res), "substances")
print(f"episode-weighted GBT: pooled {(res.gbt_pooled * wt).sum():.4f} "
      f"single {(res.gbt_single * wt).sum():.4f}")
