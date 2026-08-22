## Supplementary scripts and data added for the GPU (maximum-training-size) protocol

These four scripts and their outputs support results reported in the manuscript that the
main `run_all.sh` pipeline does not regenerate, because they require a CUDA GPU or are
sensitivity analyses rather than headline results.

| File | What it does | Requires |
|---|---|---|
| `src/08b_prep_uncapped.py` | Fits full-data GBT per substance/regime (cap 1,000,000 rows) and writes the paired per-comparison train/test splits used by the GPU worker | CPU only |
| `src/08c_gpu_worker.py` | Fits TabPFN-3 at 100,000 rows per comparison on CUDA; resumable, appends one row per completed comparison | NVIDIA GPU, `requirements-tabpfn.txt` |
| `src/08d_gpu_scaling_probe.py` | Measures the TabPFN-3 train-size / runtime / VRAM envelope; produces `data/gpu_probe.csv` | NVIDIA GPU |
| `src/02c_pooled_cohorts.py` | Sensitivity check: balanced vs proportional substance mixtures in the pooled training set | CPU only |

Data files:

- `data/gpu_probe.csv`: measured envelope (10k/25k/50k/100k training rows): wall time,
  AUROC, and peak VRAM. This is the source for the manuscript's timing claim.
- `data/p1_pooled_cohorts.csv`: balanced-vs-proportional pooling comparison.
- `docs/FEATURE_EXPANSION.md`: the 15 vs 27 feature comparison behind the locked panel.

**Checkpoint note.** `08c` and `08d` expect the TabPFN-3 checkpoint at
`data/tabpfn_v3_default.ckpt`. It is not redistributed here; obtain it from
the upstream TabPFN release and place it at that path.

## Boosting-library check (HistGradientBoosting vs XGBoost)

`data/p1_gbt_implementation_check.csv` records a matched comparison run to justify the
choice of scikit-learn's `HistGradientBoostingClassifier` as the reported GBT. Both
libraries were given 200 boosting rounds, learning rate 0.1, histogram tree method, and the
identical train/test split for the two largest substances.

| Substance | HistGBT AUROC | XGBoost AUROC | Delta | HistGBT fit (s) | XGBoost fit (s) |
|---|---|---|---|---|---|
| Alcohol | 0.7892 | 0.7904 | +0.0012 | 143 | 21 |
| Cannabis | 0.7632 | 0.7642 | +0.0010 | 61 | 15 |

XGBoost is marginally more accurate and several times faster to fit, but the AUROC
differences are an order of magnitude below every effect the paper reports (compare the
+0.0084 single-vs-pooled gap and the -0.0174 COVID onset drop) and fall inside the bootstrap
intervals. The reported results therefore stay with scikit-learn, keeping the pipeline in a
single dependency. Reproduce with `src/10_gbt_implementation_check.py`.
