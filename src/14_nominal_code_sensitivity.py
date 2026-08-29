"""
Nominal-code sensitivity check for the boosted models.

HistGradientBoostingClassifier bins integer features and splits on bin thresholds,
so an integer-coded nominal field such as STFIPS (51 codes) or RACE (9 codes) is
split as if its codes were ordered. Trees can approximate an unordered partition
with enough splits, but the fit is NOT invariant to how the codes are numbered.

Two checks:
  1. perm1-3  relabel each nominal column by a random bijection of its own codes.
              If binning ignored code order, AUROC would be unchanged. It is not.
  2. cat      declare the ten nominal fields via categorical_features, which uses
              true unordered splits.

Writes: cat_sensitivity.csv (90 rows = 5 configurations x 17 substances)
Runtime: about 21 minutes on 48 CPU cores.
"""

import os
os.environ["OMP_NUM_THREADS"]="48"
import numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

NOMINAL = ['SUB2', 'SUB3', 'ROUTE1', 'SERVICES', 'DSMCRIT', 'PSOURCE', 'GENDER', 'RACE', 'EMPLOY', 'STFIPS']
ORDINAL = ['FREQ1', 'FRSTUSE1', 'NOPRIOR', 'AGE', 'EDUC']
FEAT15  = NOMINAL + ORDINAL
RNG, SEED = 0, 42
df = pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
subs = list(df.groupby("subname").size().sort_values(ascending=False).index)
big6 = list(df.loc[df.is_big6,"subname"].unique())

tr, te = {}, {}
for s in subs:
    d = df[df.subname==s]
    a,b = train_test_split(d.index.values, test_size=0.30, random_state=RNG, stratify=d["y"].values)
    tr[s], te[s] = a, b
pool = np.concatenate([tr[s] for s in big6])
y = lambda i: df.loc[i,"y"].values

def permuted(seed):
    """Relabel each nominal column by a random bijection of its own codes."""
    g = df[FEAT15].copy()
    r = np.random.default_rng(seed)
    for c in NOMINAL:
        u = np.sort(g[c].dropna().unique())
        g[c] = g[c].map(dict(zip(u, r.permutation(u))))
    return g

CONFIGS = [("base", None, False)] + [(f"perm{k}", k, False) for k in (1,2,3)] + [("cat", None, True)]
out = []
for tag, pseed, native in CONFIGS:
    G = df[FEAT15] if pseed is None else permuted(pseed)
    kw = dict(max_iter=200, learning_rate=0.1, max_bins=255, random_state=SEED)
    if native:
        kw["categorical_features"] = [FEAT15.index(c) for c in NOMINAL]
    t0 = time.time()
    P = H(**kw).fit(G.loc[pool], y(pool))
    for s in subs:
        m = H(**kw).fit(G.loc[tr[s]], y(tr[s]))
        yt = y(te[s])
        out.append(dict(cfg=tag, substance=s, group="BIG6" if s in big6 else "small",
                        train_n=len(tr[s]), test_n=len(te[s]),
                        single=roc_auc_score(yt, m.predict_proba(G.loc[te[s]])[:,1]),
                        pooled=roc_auc_score(yt, P.predict_proba(G.loc[te[s]])[:,1])))
    print(tag, "done", round(time.time()-t0), "s", flush=True)
pd.DataFrame(out).to_csv("data/p1_categorical_sensitivity.csv", index=False)
print("WROTE", len(out))
