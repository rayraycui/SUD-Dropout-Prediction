"""
Early-stopping sensitivity check for the single vs pooled comparison.

HistGradientBoostingClassifier defaults to early_stopping='auto', which enables
internal early stopping only above 10,000 training episodes. That threshold falls
inside the range of substance training sizes studied here (435 to 1,743,828), so the
fitting procedure itself changes across the size range whose effect we report. This script refits all 18 boosted models three ways -- library default, forced off, and
forced on -- so the reported pattern can be checked against a uniform procedure.

Writes: es_sensitivity.csv (54 rows = 3 configurations x 17 substances)
Runtime: about 13 minutes on 48 CPU cores.
"""

import os
os.environ["OMP_NUM_THREADS"]="48"

import numpy as np, pandas as pd, time, json
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RNG, SEED = 0, 42
FEAT15 = ["SUB2","SUB3","ROUTE1","FREQ1","FRSTUSE1","SERVICES","DSMCRIT","NOPRIOR",
          "PSOURCE","AGE","GENDER","RACE","EDUC","EMPLOY","STFIPS"]

NOMINAL = ["SUB2","SUB3","ROUTE1","SERVICES","DSMCRIT",
           "PSOURCE","GENDER","RACE","EMPLOY","STFIPS"]

df = pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
subs = list(df.groupby("subname").size().sort_values(ascending=False).index)
big6 = list(df.loc[df.is_big6,"subname"].unique())

tr, te = {}, {}
for s in subs:
    d = df[df.subname==s]
    a,b = train_test_split(d.index.values, test_size=0.30, random_state=RNG, stratify=d["y"].values)
    tr[s], te[s] = a, b

pool = np.concatenate([tr[s] for s in big6])
X = lambda i: df.loc[i, FEAT15]
y = lambda i: df.loc[i,"y"].values

out = []
for tag, kw in [("auto", {}), ("off", {"early_stopping": False}), ("on", {"early_stopping": True})]:
    t0=time.time()

    P = H(max_iter=200, learning_rate=0.1, max_bins=255, random_state=SEED,
          categorical_features=[FEAT15.index(c) for c in NOMINAL],
          **kw).fit(X(pool), y(pool))

    pit = int(P.n_iter_)

    for s in subs:
        m = H(max_iter=200, learning_rate=0.1, max_bins=255, random_state=SEED,
              categorical_features=[FEAT15.index(c) for c in NOMINAL],
              **kw).fit(X(tr[s]), y(tr[s]))

        yt = y(te[s])

        out.append(dict(cfg=tag, substance=s, group="BIG6" if s in big6 else "small",
                        train_n=len(tr[s]), test_n=len(te[s]),
                        single=roc_auc_score(yt, m.predict_proba(X(te[s]))[:,1]),
                        pooled=roc_auc_score(yt, P.predict_proba(X(te[s]))[:,1]),
                        n_iter_single=int(m.n_iter_), n_iter_pooled=pit,
                        es_single=bool(m.do_early_stopping_), es_pooled=bool(P.do_early_stopping_)))

    print(tag, "done", round(time.time()-t0), "s", flush=True)

pd.DataFrame(out).to_csv("data/p1_early_stopping_sensitivity.csv", index=False)
print("WROTE", len(out))