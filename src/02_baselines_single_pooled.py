import pandas as pd, numpy as np, os, time
os.environ["OMP_NUM_THREADS"]="48"
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

d = pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
feat15 = ["SUB2","SUB3","ROUTE1","FREQ1","FRSTUSE1","SERVICES","DSMCRIT","NOPRIOR",
          "PSOURCE","AGE","GENDER","RACE","EDUC","EMPLOY","STFIPS"]
big6 = d[d.is_big6].copy()
subs = sorted(big6.subname.unique())
RNG=0; TRAIN_CAP=1_000_000; BOOT_N=300; TEST_BOOT_CAP=100_000

splits={}
for s in subs:
    ds=big6[big6.subname==s]
    tr,te=train_test_split(ds,test_size=0.30,random_state=RNG,stratify=ds.y)
    splits[s]=(tr,te)

def cap(df,n=TRAIN_CAP):
    return df if len(df)<=n else df.groupby("y",group_keys=False).apply(
        lambda g: g.sample(int(round(n*len(g)/len(df))),random_state=RNG))
def gbt(): return HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,random_state=RNG)
def logit_pipe():
    pre=ColumnTransformer([("oh",OneHotEncoder(handle_unknown="ignore",min_frequency=50),feat15)])
    return Pipeline([("pre",pre),("clf",LogisticRegression(penalty="l2",C=1.0,solver="saga",max_iter=200,n_jobs=48))])
def Lprep(f): return f[feat15].astype("Int64").astype(str)
def boot_ci(yt,ys,seed=0):
    r=np.random.default_rng(seed)
    if len(yt)>TEST_BOOT_CAP:
        idx=r.choice(len(yt),TEST_BOOT_CAP,replace=False); yt,ys=yt[idx],ys[idx]
    n=len(yt); a=[]
    for _ in range(BOOT_N):
        b=r.integers(0,n,n); yb=yt[b]
        if yb.min()!=yb.max(): a.append(roc_auc_score(yb,ys[b]))
    return round(float(np.percentile(a,2.5)),4), round(float(np.percentile(a,97.5)),4)

pooled_train = cap(pd.concat([splits[o][0] for o in subs]))
print("pooled train (capped):", f"{len(pooled_train):,}")
rows=[]; t0=time.time()
for fam,mk,prep in [("GBT",gbt,lambda f:f[feat15].values),
                    ("Logistic",logit_pipe,Lprep)]:
    # pooled: fit ONCE
    mp=mk(); trp=pooled_train; mp.fit(prep(trp),trp.y.values)
    for s in subs:
        tr_s,te_s=splits[s]; te_y=te_s.y.values
        sc=mp.predict_proba(prep(te_s))[:,1]; lo,hi=boot_ci(te_y,sc,RNG)
        rows.append({"family":fam,"substance":s,"regime":"pooled","n_test":len(te_s),
                     "auc":round(roc_auc_score(te_y,sc),4),"ci_lo":lo,"ci_hi":hi})
    # single: per substance
    for s in subs:
        tr_s,te_s=splits[s]; te_y=te_s.y.values
        for regime,train_df in [("single",cap(tr_s))]:
            m=mk(); m.fit(prep(train_df),train_df.y.values)
            sc=m.predict_proba(prep(te_s))[:,1]; lo,hi=boot_ci(te_y,sc,RNG)
            rows.append({"family":fam,"substance":s,"regime":regime,"n_test":len(te_s),
                         "auc":round(roc_auc_score(te_y,sc),4),"ci_lo":lo,"ci_hi":hi})
        print(f"{fam:8s} {s:16s} done ({time.time()-t0:.0f}s)")
res=pd.DataFrame(rows); res.to_csv("data/p1_phase1_single_pooled.csv",index=False)
print("SAVED", len(res),"rows |", round(time.time()-t0),"s")