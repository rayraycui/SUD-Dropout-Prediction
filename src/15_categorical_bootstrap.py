"""
Bootstrap verdicts for the eleven small substances under native categorical splits.

Recomputes the paired pooled-minus-single bootstrap intervals with the ten nominal
fields declared categorical, so the significance verdicts in the manuscript can be
compared against the ordered-code fits actually reported.

Writes: cat_bootstrap.csv
Runtime: about 6 minutes on 48 CPU cores.
"""

import os
os.environ["OMP_NUM_THREADS"]="48"
import numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
NOMINAL=['SUB2', 'SUB3', 'ROUTE1', 'SERVICES', 'DSMCRIT', 'PSOURCE', 'GENDER', 'RACE', 'EMPLOY', 'STFIPS']; ORDINAL=['FREQ1', 'FRSTUSE1', 'NOPRIOR', 'AGE', 'EDUC']; FEAT15=NOMINAL+ORDINAL
RNG,SEED,BOOT_N=0,42,2000
df=pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
subs=list(df.groupby("subname").size().sort_values(ascending=False).index)
big6=list(df.loc[df.is_big6,"subname"].unique())
small=[s for s in subs if s not in big6]
tr,te={},{}
for s in subs:
    d=df[df.subname==s]
    a,b=train_test_split(d.index.values,test_size=0.30,random_state=RNG,stratify=d["y"].values)
    tr[s],te[s]=a,b
pool=np.concatenate([tr[s] for s in big6])
y=lambda i: df.loc[i,"y"].values
G=df[FEAT15]
kw=dict(max_iter=200,learning_rate=0.1,max_bins=255,random_state=SEED,
        categorical_features=[FEAT15.index(c) for c in NOMINAL])
def boot(yv,pa,pb,seed=0):
    r=np.random.default_rng(seed); n=len(yv); o=np.empty(BOOT_N)
    for i in range(BOOT_N):
        ix=r.integers(0,n,n)
        if len(np.unique(yv[ix]))<2: o[i]=np.nan; continue
        o[i]=roc_auc_score(yv[ix],pa[ix])-roc_auc_score(yv[ix],pb[ix])
    return tuple(np.nanpercentile(o,[2.5,97.5]))
P=H(**kw).fit(G.loc[pool],y(pool))
rows=[]
for s in small:
    m=H(**kw).fit(G.loc[tr[s]],y(tr[s]))
    yt=y(te[s]); pp=P.predict_proba(G.loc[te[s]])[:,1]; ps=m.predict_proba(G.loc[te[s]])[:,1]
    lo,hi=boot(yt,pp,ps)
    rows.append(dict(substance=s,train_n=len(tr[s]),test_n=len(te[s]),
        single=roc_auc_score(yt,ps),pooled=roc_auc_score(yt,pp),
        delta=roc_auc_score(yt,pp)-roc_auc_score(yt,ps),ci_lo=lo,ci_hi=hi,
        verdict="pooled better" if lo>0 else "single better" if hi<0 else "ns"))
    print(s,"done",flush=True)
pd.DataFrame(rows).to_csv("data/p1_categorical_verdicts.csv",index=False)
print("WROTE",len(rows))
