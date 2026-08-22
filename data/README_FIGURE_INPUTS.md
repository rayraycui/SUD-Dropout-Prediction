# Input files for `09_paper_figures.py`

Everything `src/09_paper_figures.py` needs to render the six manuscript figures
(`paper/fig1_coverage.png` ... `paper/fig6_tabpfn.png`) at IEEE printed width.

Verified two ways. (1) In a clean directory containing ONLY these files, the script
exits 0 and its six PNGs are md5-identical to the repo's `paper/fig*.png`. (2) The six
images embedded in `paper_v1.pdf` were extracted and compared pixel-by-pixel against those
PNGs: maximum per-pixel difference 0 on all six, i.e. the PDF carries exactly what this
script produces.

## Layout

Unzip so that `data/` and `src/` are siblings, then run from that parent directory:

```
.
├── data/    <- the 13 files below
├── src/09_paper_figures.py
└── paper/   <- created by the script if absent; figures are written here
```

```bash
mkdir -p paper
python src/09_paper_figures.py
```

Requirements: `pandas`, `numpy`, `matplotlib`, `pyarrow`, `scikit-learn`
(the repo's `requirements.txt` covers these). No TabPFN, no GPU, no raw TEDS-D
download: every model has already been fitted and scored; these are its outputs.

## Files

| File | Rows | Size | Feeds | Produced by |
|------|------|------|-------|-------------|
| `p1_coverage_ci.csv` | 12 | 599 B | Fig. 1 | `src/04_coverage_unseen.py` |
| `p1_phase1_single_pooled.csv` | 24 | 1,286 B | Fig. 2 | `src/02_baselines_single_pooled.py` |
| `fairness_preds.parquet` | 1,950,373 | 31,195,715 B | Fig. 3 (calibration), Fig. 5 (reference line) | `src/07a_fairness_score.py` |
| `p1_dca_realdata.csv` | 16 | 545 B | Fig. 3 (decision curve) | `src/03_calibration_dca.py` |
| `p1_covid_temporal.csv` | 3 | 198 B | Fig. 4 | `src/06_covid_temporal.py` |
| `p1_fairness_subgroups.csv` | 22 | 1,187 B | Fig. 5 (panels) | `src/07_fairness_audit.py` |
| `p1_fairness_states.csv` | 48 | 1,127 B | (supp) trained-substance fairness | `src/07_fairness_audit.py` |
| `p1_unseen_summary.csv` | 1 |: | Fig. 5b (cohort AUROC reference line) | `src/07b_unseen_deployability.py` |
| `p1_calibration_unseen_deciles.csv` | 10 |: | Fig. 3b (panel a) | `src/07b_unseen_deployability.py` |
| `p1_dca_realdata_unseen.csv` | 101 |: | Fig. 3b (panel b) | `src/07b_unseen_deployability.py` |
| `p1_fairness_unseen.csv` | 22 |: | Fig. 5b (both panels) | `src/07b_unseen_deployability.py` |
| `p1_covid_temporal_unseen.csv` | 3 |: | Fig. 4b (unseen series) | `src/07b_unseen_deployability.py` |
| `p1_tabpfn_uncapped.csv` | 18 | 2,477 B | Fig. 6 | committed intermediate; assembled from `src/08b_prep_uncapped.py` + `src/08c_gpu_worker.py` (merge step not in repo) |

`fairness_preds.parquet` is 30 MB and dominates the bundle; the other twelve total
under 15 KB.

## Columns

**`p1_coverage_ci.csv`**: one row per small substance class held out of the pooled
big-6 training set. `substance`, `n` (episodes in that class), `dropout` (base rate),
`auc`, `ci_lo`, `ci_hi` (bootstrap 95% CI).

**`p1_phase1_single_pooled.csv`**: `family` (GBT / logistic), `substance`,
`regime` (single / pooled), `n_test`, `auc`, `ci_lo`, `ci_hi`. Fig. 2 uses the
GBT rows; the logistic rows appear in the paper's Table I.

**`fairness_preds.parquet`**: the pooled-GBT held-out predictions, one row per test
episode. `y` (1 = dropout), `p` (predicted probability), plus the grouping variables
`RACE`, `GENDER`, `AGE`, `STFIPS`, `subname`. Note the subgroup dimension is named
`GENDER`, not `SEX`. Fig. 3's calibration curve is computed from `y`/`p` directly
(decile bins), which is why the script re-derives and prints AUROC 0.7913 and
Brier 0.1792 on every run: it is a self-check that the file is the right one.

**`p1_dca_realdata.csv`**: decision-curve analysis on the same predictions.
`threshold`, `nb_model`, `nb_treat_all`, `nb_treat_none`, `advantage_vs_best_default`.

**`p1_covid_temporal.csv`**: `test_era` (2019 pre-COVID held-out / 2020 / 2022), `n`,
`dropout`, `auc`, `ci_lo`, `ci_hi`. Training years are 2015-2018 only, so 2019 is a
clean held-out year.

**`p1_fairness_subgroups.csv`**: `dim` (RACE / GENDER / AGE), `group`, `n`, `dropout`,
`auc`, `ci_lo`, `ci_hi`, `calib_gap` (mean predicted minus mean observed). Max
|calib_gap| is 0.0212 (Asian, n=13,751); every group with n>50,000 is within 0.005.

**`p1_fairness_states.csv`**: `state`, `n`, `auc`, `calib_gap` for 48 states. Fig. 5
uses only `auc.min()` and `auc.max()` for its title.

**`p1_tabpfn_uncapped.csv`**: the two-arm TabPFN-3 benchmark, 24 cells.
`cell`, `substance`, `regime` (single / pooled / coverage), `n_gbt_train`,
`n_tabpfn_train`, `n_test`, `gbt_auc`, `gbt_secs`, `tabpfn_auc`, `n_tabpfn_ctx`,
`pred_s`, `delta` (TabPFN minus GBT on the same test rows), `d_lo`, `d_hi` (paired
bootstrap 95% CI on the difference), `sig`, and `delta_matched` (the same cell's value
from the matched 10,000-row arm). Fig. 6 plots `delta_matched` as the light point and
`delta` with its CI as the dark point, so each arrow shows one cell moving from the
matched budget to best-available data.

## Checksums (md5)

```
5a04dde3d362f702d986264915f9e9ea  fairness_preds.parquet
d55124576474db49e8ad908b71eebb7b  p1_coverage_ci.csv
488897fc0fa3e8024e9c6776bdecf347  p1_covid_temporal.csv
4658402fe027c9eaca8116fa4a7ebf5b  p1_dca_realdata.csv
74dd65852089f4af74c1db831472f440  p1_fairness_states.csv
169b9d7dcaa126e354082c9df576ec34  p1_fairness_subgroups.csv
5d13e99e4166e581c4df735783e4b4b5  p1_phase1_single_pooled.csv
3f537121138088d3b2647c6604ac17a5  p1_tabpfn_uncapped.csv
5c9c4e2b3249b9607585f4f921ab5939  p1_unseen_summary.csv
f170c3a2e197796e097d0ef54c8b9575  p1_calibration_unseen_deciles.csv
89bf385b5382a8ea9675ad7211a4c992  p1_dca_realdata_unseen.csv
560e42a8e22446c2c06c062c9676adf3  p1_fairness_unseen.csv
db7a55ae049f380b827f34774d9f9311  p1_covid_temporal_unseen.csv
```
