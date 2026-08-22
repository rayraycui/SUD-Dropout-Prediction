#!/usr/bin/env python3
"""Two pooled cohorts of 1.5M each, scored on identical per-substance test sets:
  A. proportional: substance mix matches the training pool (stratified by substance)
  B. balanced: 250,000 episodes from each of the six substances
Tests whether the -0.008 pooling cost is driven by the alcohol-heavy mixture.

Output schema (data/p1_pooled_cohorts.csv), matching the committed file
column-for-column:

  substance            the big-6 substance, in BIG6 order
  auc_prop_full        AUROC of the proportional cohort on that substance's test set
  auc_bal_full         AUROC of the balanced cohort on the same test set
  single               AUROC of the substance-specific (single-substance) GBT on the
                       same substance, joined from data/p1_phase1_single_pooled.csv
                       (family=GBT, regime=single). This is NOT computed here: 02c
                       fits only the two pooled cohorts, so the single-substance
                       baseline is read from the file that does compute it (02).
  gap_prop             auc_prop_full - single   (pooling cost, proportional mix)
  gap_bal              auc_bal_full  - single   (pooling cost, balanced mix)
  pool_train_share_%   the substance's share of the proportional training pool, %
  delta                auc_bal_full - auc_prop_full, from the unrounded AUROCs
  d_lo, d_hi           95% paired-bootstrap CI on delta (same test rows)

The paired-bootstrap significance flag the earlier draft of this script emitted
("sig") is not part of the committed schema and is not written; d_lo/d_hi carry
the same information.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PQ = "data/teds_d_analysis_2015_2022.parquet"
SINGLE_REF = "data/p1_phase1_single_pooled.csv"
FEAT = ["SUB2", "SUB3", "ROUTE1", "FREQ1", "FRSTUSE1", "SERVICES", "DSMCRIT", "NOPRIOR",
        "PSOURCE", "AGE", "GENDER", "RACE", "EDUC", "EMPLOY", "STFIPS"]
BIG6 = ["Alcohol", "Heroin", "Cannabis", "Methamphetamine", "Other_opioids", "Cocaine"]
RNG, N_TOT, BOOT, BOOT_CAP = 0, 1_500_000, 300, 100_000
PER = N_TOT // len(BIG6)

d = pd.read_parquet(PQ)
big6 = d[d.subname.isin(BIG6)]
tr, te = {}, {}
for s in BIG6:
    a, b = train_test_split(big6[big6.subname == s], test_size=0.30,
                            random_state=RNG, stratify=big6[big6.subname == s].y)
    tr[s], te[s] = a, b
pool = pd.concat([tr[s] for s in BIG6])

# --- Cohort A: proportional (stratified by substance, mix = pool mix) ---
fracs = {s: len(tr[s]) / len(pool) for s in BIG6}
A = pd.concat([tr[s].sample(int(round(fracs[s] * N_TOT)), random_state=RNG) for s in BIG6])
# --- Cohort B: balanced (equal n per substance) ---
B = pd.concat([tr[s].sample(PER, random_state=RNG) for s in BIG6])


def fit(train_df):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RNG)
    m.fit(train_df[FEAT].values, train_df.y.values)
    return m


out = {}
for tag, coh in [("proportional", A), ("balanced", B)]:
    t0 = time.time(); m = fit(coh); secs = time.time() - t0
    comp = coh.subname.value_counts().to_dict()
    out[tag] = {"n": int(len(coh)), "dropout": float(coh.y.mean()), "fit_s": round(secs, 1),
                "composition": {k: int(v) for k, v in comp.items()}, "auc": {}}
    for s in BIG6:
        p = m.predict_proba(te[s][FEAT].values)[:, 1]
        np.save(f"coh_{tag}_{s}.npy", p)
        out[tag]["auc"][s] = float(roc_auc_score(te[s].y.values, p))
    print(tag, "fit", round(secs), "s |",
          {k: round(v, 4) for k, v in out[tag]["auc"].items()}, flush=True)

# single-substance GBT baseline, computed by 02 and read here (see docstring)
_sp = pd.read_csv(SINGLE_REF)
single_auc = (_sp[(_sp.family == "GBT") & (_sp.regime == "single")]
              .set_index("substance").auc.to_dict())
missing = [s for s in BIG6 if s not in single_auc]
if missing:
    raise SystemExit(f"{SINGLE_REF} has no GBT/single row for {missing}; run src/02 first")

# paired bootstrap on the per-substance difference (balanced - proportional), same test rows
rng = np.random.default_rng(RNG); rows = []
for s in BIG6:
    y = te[s].y.values
    pa = np.load(f"coh_proportional_{s}.npy"); pb = np.load(f"coh_balanced_{s}.npy")
    if len(y) > BOOT_CAP:
        idx0 = rng.choice(len(y), BOOT_CAP, replace=False)
        y, pa, pb = y[idx0], pa[idx0], pb[idx0]
    ds = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        ds.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # AUROCs rounded to 4 dp are the reported values; the two gap columns are
    # differences of the reported values, while delta keeps full precision.
    a_prop = round(out["proportional"]["auc"][s], 4)
    a_bal = round(out["balanced"]["auc"][s], 4)
    rows.append({"substance": s,
                 "auc_prop_full": a_prop,
                 "auc_bal_full": a_bal,
                 "single": single_auc[s],
                 "gap_prop": round(a_prop - single_auc[s], 4),
                 "gap_bal": round(a_bal - single_auc[s], 4),
                 "pool_train_share_%": round(100 * fracs[s], 1),
                 "delta": round(out["balanced"]["auc"][s] - out["proportional"]["auc"][s], 4),
                 "d_lo": round(lo, 4), "d_hi": round(hi, 4)})
res = pd.DataFrame(rows)[["substance", "auc_prop_full", "auc_bal_full", "single",
                          "gap_prop", "gap_bal", "pool_train_share_%",
                          "delta", "d_lo", "d_hi"]]
res.to_csv("data/p1_pooled_cohorts.csv", index=False)
json.dump(out, open("p1_pooled_cohorts_meta.json", "w"), indent=1)
print(res.to_string(index=False))
print("mean delta:", round(res.delta.mean(), 4),
      "| cells with CI excluding zero:", int(((res.d_lo > 0) | (res.d_hi < 0)).sum()))
print("DONE")
