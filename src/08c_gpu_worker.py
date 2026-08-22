"""Score ONE prepared cell with TabPFN-3 on GPU, append to CSV, exit.
Resumable: skips cells already in the output. Usage: gpu_worker.py <cell_name>"""
import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score

CKPT = "models/tabpfn-v3-classifier-v3_default.ckpt"
OUT = "uncapped_tabpfn.csv"
name = sys.argv[1]

if os.path.exists(OUT) and name in set(pd.read_csv(OUT).cell.astype(str)):
    print(f"SKIP {name}"); sys.exit(0)

z = np.load(f"cells/{name}.npz")
Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
clf = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=2,
                       ignore_pretraining_limits=True)
clf.fit(Xtr, ytr)
tfit = time.time() - t0
t0 = time.time()
p = clf.predict_proba(Xte)[:, 1]
tpred = time.time() - t0
auc = roc_auc_score(yte, p)
np.save(f"cells/{name}__tabpfnprob.npy", p)

row = {"cell": name, "n_train": len(Xtr), "n_test": len(Xte),
       "tabpfn_auc": round(auc, 4), "fit_s": round(tfit, 1), "pred_s": round(tpred, 1),
       "peak_vram_GiB": round(torch.cuda.max_memory_allocated()/2**30, 2)}
pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
print("OK", json.dumps(row), flush=True)
