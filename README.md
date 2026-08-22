# Zero-Shot Coverage of Unseen Substance Classes for SUD Treatment-Dropout Prediction on 6.7 Million Treatment Episodes

Reproduction code, data-processing pipeline, and figures for the manuscript
*"Zero-Shot Coverage of Unseen Substance Classes for SUD Treatment-Dropout Prediction on 6.7 Million Treatment Episodes
on 6.7 Million National Treatment Episodes."*

A single pooled gradient-boosted-tree model, trained only on the six highest-volume
substances in the U.S. Treatment Episode Data Set (TEDS-D), predicts treatment dropout
for twelve **held-out substance classes it never saw** at mean AUROC 0.777. The repo
reproduces every result and figure in the paper from public data on CPU-only hardware.

## Key findings
1. **Coverage (primary):** one pooled model covers 12 unseen small substance classes
   (mean AUROC 0.777, range 0.745–0.806), no per-substance training required.
2. **Pooling costs no accuracy:** pooling does **not** raise accuracy (Δ = −0.008 against
   substance-specific training), so the value is coverage, not a leaderboard gain.
3. **Deployable on the unseen classes:** calibration transfers (mean decile gap 0.015, max
   0.046), net benefit stays positive to threshold 0.80, and discrimination is stable across
   22 strata (0.677–0.798), with one exception, a +0.140 calibration gap in the 65+ band
   that traces to a single state × substance reporting cell.
4. **Foundation-model benchmark:** TabPFN-3 beats GBT in all **18** matched-training-size
   comparisons (mean +0.018 AUROC), but at each model's maximum training size the two are
   indistinguishable (8 of 18, mean −0.0001), a sample-efficiency effect, not a higher
   accuracy ceiling.

## Training assembly

Every cross-sectional result comes from **one** fit, the **joint-stratified holdout (JSH)**
in `src/jsh_fit.py`: the 6,501,243 big-6 episodes are split 70/30 stratified jointly on
(primary substance × outcome) → 4,550,870 train / 1,950,373 holdout; the train side is
subsampled to 1,000,000 on the same joint key; one `HistGradientBoostingClassifier` is fitted
there and scored **without refitting** on both the big-6 holdout and all 215,983 unseen-class
episodes. The two temporal arms are year-restricted refits of the identical specification
(`src/jsh_year_restricted.py`). Four arms sit outside JSH by design and are labelled as such.
`docs/PROVENANCE.md` maps every number in the paper to its fit and data file.

## Repository layout
```
.
├── README.md                 # this file
├── LICENSE                   # MIT (code)
├── requirements.txt          # core Python dependencies
├── requirements-tabpfn.txt   # extra deps for the optional TabPFN benchmark
├── Makefile                  # `make all` runs the full pipeline
├── run_all.sh                # same pipeline as a plain shell script
├── src/                      # all analysis + figure code (numbered by run order)
│   ├── jsh_fit.py                     # CANONICAL training assembly (joint-stratified holdout); imported, not run
│   ├── jsh_year_restricted.py         # year-restricted JSH variant used by the two temporal arms
│   ├── 00_download_teds.py            # fetch TEDS-D public-use files -> data/raw/
│   ├── 01_build_analysis_table.py     # harmonize 7 years -> data/teds_d_analysis_2015_2022.parquet
│   ├── 02_baselines_single_pooled.py  # -> data/p1_phase1_single_pooled.csv
│   ├── 02b_plot_regimes.py            # -> figures/p1_phase1_regimes.png
│   ├── 03_calibration_dca.py          # -> figures/p1_calibration_dca.png
│   ├── 04_coverage_unseen.py          # -> figures/p1_coverage_ci.png (+ data/p1_coverage_ci.csv)
│   ├── 05_coverage_temporal.py        # -> figures/p1_coverage_temporal.png
│   ├── 06_covid_temporal.py           # -> figures/p1_covid_temporal.png
│   ├── 07a_fairness_score.py          # -> data/fairness_preds.parquet
│   ├── 07_fairness_audit.py           # -> figures/p1_fairness.png
│   ├── 07b_unseen_deployability.py    # -> unseen-class calibration/DCA/fairness/COVID CSVs
│   ├── 07c_age65_anomaly.py           # -> data/p1_age65_anomaly_diagnosis.csv (65+ gap cause)
│   ├── 08a_tabpfn_vs_gbt_compute.py   # optional; -> data/p1_tabpfn_vs_gbt.csv
│   ├── 08_tabpfn_vs_gbt_plot.py       # -> figures/p1_tabpfn_vs_gbt.png
│   ├── 08b_prep_uncapped.py           # GPU: full-data GBT + per-comparison splits
│   ├── 08c_gpu_worker.py              # GPU: resumable TabPFN-3 comparisons (CUDA)
│   ├── 08d_gpu_scaling_probe.py       # GPU: train-size/VRAM envelope -> data/gpu_probe.csv
│   ├── 02c_pooled_cohorts.py          # sensitivity: balanced vs proportional pooling
│   ├── 10_gbt_implementation_check.py # XGBoost vs HistGBT agreement -> data/p1_gbt_implementation_check.csv
│   ├── 09_paper_figures.py            # -> paper/fig1..fig6 at IEEE print width
│   └── 09b_paper_figures_unseen.py    # -> paper/fig3b, fig4b, fig5b (unseen-class deployability)
├── data/                     # analysis table + per-figure result CSVs (see data/README.md)
│   └── raw/                  # TEDS-D public-use CSVs land here (not committed)
├── figures/                  # publication figures (PNG, 300 dpi)
├── paper/                    # LaTeX source (see Overleaf package)
└── docs/                     # data dictionary, results write-up, figure->code map
```

## Quick start
```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. get the data (public, ~2 GB of CSVs)
python src/00_download_teds.py          # writes data/raw/*.csv

# 3. run everything
make all                                # or: bash run_all.sh
```
The committed `data/teds_d_analysis_2015_2022.parquet` lets you skip steps 0–1 and
reproduce all figures directly (steps 02–08 read it). To rebuild the table from scratch,
run steps 00 then 01.

## Data source and access
TEDS-D public-use files are distributed by SAMHSA:
<https://www.samhsa.gov/data/data-we-collect/teds/datafiles>. They are de-identified and
public; no application is required. **2021 is intentionally excluded**: its public-use
file ships text labels instead of the numeric codes used in every other year.
`src/00_download_teds.py` documents the exact URLs; refresh them from the page above if
SAMHSA moves a file.

## Compute
Everything runs on a single CPU machine (developed on 48 cores, no GPU). The full
pipeline (excluding downloads) takes on the order of a few hours, dominated by the
gradient-boosting fits and bootstrap CIs. The TabPFN benchmark (step 08a) is optional and
slow on CPU (in-context inference cost scales with train×test); it is resumable.

## Reproducibility notes
- Random seed is fixed (`RNG=0`) throughout; training is capped at a stratified
  1,000,000 rows where noted. The one exception is the within-unseen COVID arm in
  `src/07b_unseen_deployability.py`, which uses seed 42 by design.
- **Bootstrap CIs use 300 resamples** in every arm (`src/04`, `src/05`, `src/06`,
  `src/07_fairness_audit.py`). Test-set caps differ per arm: 100,000 rows in `src/04`,
  `src/05` and `src/07_fairness_audit.py`, 150,000 in `src/06`. Point AUROCs are unaffected
  by the resample count; only CI bounds are.
- `src/jsh_fit.py` and `src/jsh_year_restricted.py` are **imported modules, not pipeline
  steps**, they define the canonical training assembly that the numbered scripts call, so
  they do not appear as stages in `run_all.sh` or the `Makefile`. Running `python
  src/jsh_fit.py` directly performs the fit and prints the cohort sizes as a self-check.
- **Known gap:** `src/02b_plot_regimes.py` is not present in this repository. Both
  `run_all.sh` and the `Makefile` reference it in a commented-out line, so the pipeline
  runs end to end without it. The figure it produced (`figures/p1_phase1_regimes.png`)
  is committed, and the manuscript does not use it.
- Each figure's underlying numbers are saved as a CSV in `data/` (see
  `docs/FIGURES.md` for the figure→code→data mapping).
- Model: scikit-learn `HistGradientBoostingClassifier`; TabPFN-3 via the `tabpfn` package.

## Citation
If you use this code or the derived analysis table, please cite the manuscript (see
`paper/`) and the TEDS-D data source (SAMHSA). A `CITATION.cff` template is included.

## License
Code is released under the MIT License (`LICENSE`). TEDS-D data is governed by SAMHSA's
public-use data terms and is **not** redistributed in this repository.
