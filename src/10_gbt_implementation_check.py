"""
Boosting-library check: scikit-learn HistGradientBoostingClassifier vs XGBoost.

Justifies the choice of HistGradientBoostingClassifier as the reported GBT by running
both libraries at matched settings on the two largest substances.

Writes: data/p1_gbt_implementation_check.csv
"""
import time

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEAT15 = ["SUB2", "SUB3", "ROUTE1", "FREQ1", "FRSTUSE1", "SERVICES", "DSMCRIT",
          "NOPRIOR", "PSOURCE", "AGE", "GENDER", "RACE", "EDUC", "EMPLOY", "STFIPS"]
RNG = 0
CAP = 1_000_000
SUBSTANCES = ["Alcohol", "Cannabis"]

d = pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
big6 = d[d.is_big6]


def cap(df, n=CAP):
    if len(df) <= n:
        return df
    return df.groupby("y", group_keys=False).apply(
        lambda g: g.sample(int(round(n * len(g) / len(df))), random_state=RNG))


rows = []
for sub in SUBSTANCES:
    ds = big6[big6.subname == sub]
    tr, te = train_test_split(ds, test_size=0.30, random_state=RNG, stratify=ds.y)
    trc = cap(tr)
    Xtr, ytr = trc[FEAT15].values, trc.y.values
    Xte, yte = te[FEAT15].values, te.y.values

    t0 = time.time()
    h = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RNG)
    h.fit(Xtr, ytr)
    hist_s = time.time() - t0
    hist_auc = roc_auc_score(yte, h.predict_proba(Xte)[:, 1])

    t0 = time.time()
    x = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, tree_method="hist",
                          max_cat_to_onehot=1, random_state=RNG, n_jobs=48,
                          eval_metric="logloss")
    x.fit(Xtr, ytr)
    xgb_s = time.time() - t0
    xgb_auc = roc_auc_score(yte, x.predict_proba(Xte)[:, 1])

    rows.append(dict(substance=sub, n_train=len(trc), n_test=len(te),
                     hist_auc=hist_auc, xgb_auc=xgb_auc, delta=xgb_auc - hist_auc,
                     hist_s=hist_s, xgb_s=xgb_s, hist_trees=h.n_iter_))
    print(f"{sub}: hist {hist_auc:.4f} ({hist_s:.0f}s, {h.n_iter_} trees) | "
          f"xgb {xgb_auc:.4f} ({xgb_s:.0f}s) | delta {xgb_auc - hist_auc:+.4f}")

out = pd.DataFrame(rows)
out.to_csv("data/p1_gbt_implementation_check.csv", index=False)
print(f"\nwrote data/p1_gbt_implementation_check.csv ({len(out)} rows)")
