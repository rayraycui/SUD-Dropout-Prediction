#!/usr/bin/env python3
"""
TabPFN-3 vs GBT head-to-head at matched 10k-train / 2k-test per cell (bootstrap 300),
across single / pooled regimes (big-6) plus coverage of small classes.
Resumable: reloads data/p1_tabpfn_vs_gbt.csv and skips completed (substance,regime,family)
cells. Requires the TabPFN-3 checkpoint at data/tabpfn_v3_default.ckpt and the
`tabpfn` package (see requirements-tabpfn.txt). CPU-only; set OMP threads as needed.
"""
import time, os, numpy as np, pandas as pd, torch
torch.set_num_threads(48)
from tabpfn import TabPFNClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CKPT="data/tabpfn_v3_default.ckpt"
RNG=0; TRAIN_CAP=10000; TEST_CAP=2000; BOOT_N=300
feat=["SUB2","SUB3","ROUTE1","FREQ1","FRSTUSE1","SERVICES","DSMCRIT","NOPRIOR",
      "PSOURCE","AGE","GENDER","RACE","EDUC","EMPLOY","STFIPS"]
big6=["Alcohol","Heroin","Cannabis","Methamphetamine","Other_opioids","Cocaine"]

d=pd.read_parquet("data/teds_d_analysis_2015_2022.parquet")
B=d[d.is_big6].copy(); S=d[~d.is_big6].copy()

def cap(df,n,seed=RNG):
    if len(df)<=n: return df
    return df.groupby("y",group_keys=False).apply(lambda g:g.sample(max(1,int(round(n*len(g)/len(df)))),random_state=seed),include_groups=True)

def fit_tabpfn(Xtr,ytr):
    m=TabPFNClassifier(model_path=CKPT,device="cpu",ignore_pretraining_limits=True,random_state=RNG)
    m.fit(Xtr,ytr); return m
def fit_gbt(Xtr,ytr):
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,random_state=RNG)
    m.fit(Xtr,ytr); return m
def boot_ci(y,p,seed=RNG):
    r=np.random.default_rng(seed); n=len(y); a=[]
    for _ in range(BOOT_N):
        b=r.integers(0,n,n); yb=y[b]
        if yb.min()!=yb.max(): a.append(roc_auc_score(yb,p[b]))
    return round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)

OUT="data/p1_tabpfn_vs_gbt.csv"
# resume: load already-computed cells, skip them
done=set(); rows=[]
if os.path.exists(OUT):
    prev=pd.read_csv(OUT)
    rows=prev.to_dict("records")
    done={(r["substance"],r["regime"],r["family"]) for r in rows}
    print(f"resuming: {len(done)} cells already done",flush=True)
def save(): pd.DataFrame(rows).to_csv(OUT,index=False)

# per-substance train/test splits (fixed test set shared across regimes & models)
splits={}
for s in big6:
    sd=B[B.subname==s]
    tr,te=train_test_split(sd,test_size=0.30,random_state=RNG,stratify=sd.y)
    splits[s]=(tr, cap(te,TEST_CAP))

# Build a global leakage-free pool: every substance's 30% test rows removed once.
test_idx=set()
for s in big6:
    test_idx.update(splits[s][1].index.tolist())
B_notest=B[~B.index.isin(test_idx)]  # no held-out test row of ANY substance

for s in big6:
    tr_s,te=splits[s]; yte=te.y.values; Xte=te[feat].values
    single_tr=cap(tr_s,TRAIN_CAP)
    pooled_tr=cap(B_notest, TRAIN_CAP)          # pooled = all big6 train rows (no test leakage)
    for regime,trdf in [("single",single_tr),("pooled",pooled_tr)]:
        Xtr=trdf[feat].values; ytr=trdf.y.values
        for fam,fitter in [("TabPFN",fit_tabpfn),("GBT",fit_gbt)]:
            if (s,regime,fam) in done: continue
            t=time.time(); mdl=fitter(Xtr,ytr)
            p=mdl.predict_proba(Xte)[:,1]; el=time.time()-t
            auc=roc_auc_score(yte,p); lo,hi=boot_ci(yte,p)
            rows.append(dict(substance=s,regime=regime,family=fam,n_train=len(trdf),
                             n_test=len(te),auc=round(auc,4),ci_lo=lo,ci_hi=hi,secs=round(el,1)))
            print(f"{s:16} {regime:7} {fam:7} ntr={len(trdf):6} auc={auc:.4f} [{lo},{hi}] {el:.0f}s",flush=True)
            save()

# coverage: pooled big6 (10k, no test leakage) predicts each small class (2k test), both models
pooled_all=cap(B_notest,TRAIN_CAP)
cov_todo=[s for s in S.subname.unique()
          if not (S[S.subname==s].y.nunique()<2 or len(S[S.subname==s])<50)]
need_cov=any((f"COV:{s}","coverage",fam) not in done for s in cov_todo for fam in ("TabPFN","GBT"))
if need_cov:
    mt=fit_tabpfn(pooled_all[feat].values,pooled_all.y.values)
    mg=fit_gbt(pooled_all[feat].values,pooled_all.y.values)
for s in S.subname.unique():
    sd=S[S.subname==s]
    if sd.y.nunique()<2 or len(sd)<50: continue
    te=cap(sd,TEST_CAP); yte=te.y.values; Xte=te[feat].values
    for fam,mdl in [("TabPFN",mt),("GBT",mg)]:
        if (f"COV:{s}","coverage",fam) in done: continue
        t=time.time(); p=mdl.predict_proba(Xte)[:,1]; el=time.time()-t
        auc=roc_auc_score(yte,p); lo,hi=boot_ci(yte,p)
        rows.append(dict(substance=f"COV:{s}",regime="coverage",family=fam,n_train=len(pooled_all),
                         n_test=len(te),auc=round(auc,4),ci_lo=lo,ci_hi=hi,secs=round(el,1)))
        print(f"COV {s:16} {fam:7} auc={auc:.4f} [{lo},{hi}] {el:.0f}s",flush=True)
        save()
print("DONE rows=",len(rows),flush=True)
