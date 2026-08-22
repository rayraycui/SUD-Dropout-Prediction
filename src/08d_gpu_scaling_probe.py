"""Run ONE TabPFN-3 GPU timing config per invocation, append to CSV, exit.
Designed to fit inside the 60s call_command window; resumable via the done-set."""
import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PARQUET = "data/teds_d_analysis_2015_2022.parquet"
CKPT = "models/tabpfn-v3-classifier-v3_default.ckpt"
FEAT = ["SUB2","SUB3","ROUTE1","FREQ1","FRSTUSE1","SERVICES","DSMCRIT","NOPRIOR",
        "PSOURCE","AGE","GENDER","RACE","EDUC","EMPLOY","STFIPS"]
OUT = "gpu_probe.csv"
RNG = 0

CONFIGS = [(10_000, 2_000), (25_000, 2_000), (50_000, 2_000), (100_000, 2_000),
           (200_000, 2_000), (50_000, 20_000), (100_000, 20_000)]

idx = int(sys.argv[1])
n_train, n_test = CONFIGS[idx]
key = f"{n_train}_{n_test}"

done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT).key.astype(str))
if key in done:
    print(f"SKIP {key} (already done)"); sys.exit(0)

d = pd.read_parquet(PARQUET)
alc = d[d.subname == "Alcohol"]
tr_all, te_all = train_test_split(alc, test_size=0.30, random_state=RNG, stratify=alc.y)
trs = tr_all.sample(min(n_train, len(tr_all)), random_state=RNG)
te  = te_all.sample(min(n_test, len(te_all)), random_state=RNG)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
clf = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=2,
                       ignore_pretraining_limits=True)
clf.fit(trs[FEAT].values, trs.y.values)
tfit = time.time() - t0
t0 = time.time()
p = clf.predict_proba(te[FEAT].values)[:, 1]
tpred = time.time() - t0
auc = roc_auc_score(te.y.values, p)
vram = torch.cuda.max_memory_allocated() / 2**30

row = {"key": key, "n_train": len(trs), "n_test": len(te), "fit_s": round(tfit,1),
       "pred_s": round(tpred,1), "total_s": round(tfit+tpred,1),
       "auc": round(auc,4), "peak_vram_GiB": round(vram,2)}
pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
print("OK", json.dumps(row), flush=True)
