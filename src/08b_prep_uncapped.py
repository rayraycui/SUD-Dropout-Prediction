"""Build the uncapped-arm comparison: identical test rows for TabPFN (GPU, n_train=100k)
and GBT (CPU, full training pool). Writes one npz per cell for the GPU worker, and
trains/scores the full-data GBT side locally."""
import os, json, time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PARQUET = "data/teds_d_analysis_2015_2022.parquet"
FEAT = ["SUB2","SUB3","ROUTE1","FREQ1","FRSTUSE1","SERVICES","DSMCRIT","NOPRIOR",
        "PSOURCE","AGE","GENDER","RACE","EDUC","EMPLOY","STFIPS"]
RNG = 0
N_TABPFN = 100_000      # TabPFN training context (GPU envelope)
GBT_CAP  = 1_000_000    # full-data GBT cap, same convention as the paper baseline
N_TEST   = 2_000
CELLDIR  = "cells"; os.makedirs(CELLDIR, exist_ok=True)

BIG6 = ["Alcohol","Heroin","Cannabis","Methamphetamine","Other_opioids","Cocaine"]
SMALL = ["Other_amphetamines","Other_hallucinogens","Other_stimulants",
         "Inhalants","OTC","Barbiturates"]

d = pd.read_parquet(PARQUET)
big6 = d[d.subname.isin(BIG6)]

# per-substance 70/30 split (deterministic)
tr, te = {}, {}
for s in BIG6:
    sub = big6[big6.subname == s]
    a, b = train_test_split(sub, test_size=0.30, random_state=RNG, stratify=sub.y)
    tr[s], te[s] = a, b

def cap(df, n):
    return df.sample(min(n, len(df)), random_state=RNG)

cells, gbt_rows = [], []
def run_cell(name, substance, regime, train_pool, test_df):
    Xte, yte = test_df[FEAT].values, test_df.y.values
    # --- TabPFN inputs (100k context) ---
    tp = cap(train_pool, N_TABPFN)
    np.savez_compressed(f"{CELLDIR}/{name}.npz",
                        Xtr=tp[FEAT].values.astype(np.float32), ytr=tp.y.values.astype(np.int8),
                        Xte=Xte.astype(np.float32), yte=yte.astype(np.int8))
    # --- GBT on the FULL pool (uncapped arm), same test rows ---
    gp = cap(train_pool, GBT_CAP)
    t0 = time.time()
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RNG)
    m.fit(gp[FEAT].values, gp.y.values)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    np.save(f"{CELLDIR}/{name}__gbtprob.npy", p)
    np.save(f"{CELLDIR}/{name}__ytest.npy", yte)
    gbt_rows.append({"cell": name, "substance": substance, "regime": regime,
                     "n_gbt_train": len(gp), "n_tabpfn_train": len(tp), "n_test": len(test_df),
                     "gbt_auc": round(auc, 4), "gbt_secs": round(time.time()-t0, 1)})
    cells.append({"cell": name, "substance": substance, "regime": regime})
    print(f"{name:38s} gbt_train={len(gp):>9,} test={len(test_df):>6,} auc={auc:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

for s in BIG6:
    tes = cap(te[s], N_TEST)
    run_cell(f"{s}__single",  s, "single", tr[s], tes)
    run_cell(f"{s}__pooled",  s, "pooled", pd.concat([tr[o] for o in BIG6]), tes)

pool_all = pd.concat([tr[o] for o in BIG6])
for c in SMALL:
    sub = d[d.subname == c]
    if len(sub) < 500 or sub.y.nunique() < 2:
        print("skip small class", c, len(sub)); continue
    run_cell(f"{c}__coverage", c, "coverage", pool_all, cap(sub, N_TEST))

pd.DataFrame(gbt_rows).to_csv("data/uncapped_gbt.csv", index=False)
json.dump(cells, open("cells.json", "w"), indent=1)
print("CELLS:", len(cells), "-> uncapped_gbt.csv")
