# Provenance of every number in `paper/paper_v1.tex`

This file supersedes the earlier eight-scheme provenance table. It maps every value
reported in manuscript v1 to the fit that produced it and the committed data file it is
read from, so the "one canonical assembly" claim is auditable line by line.

Verified on the compiled `paper/paper_v1.pdf` (5 pages, 612 x 792 pt,
`paper_v1.tex` sha256 `a3b4eeee6713f3ed8d913776133ea7d7046020308bb7bb0ad30d7f739c3bebbd`).

## The three statements this table supports

**(a) All cross-sectional results derive from ONE canonical JSH fit.** The
joint-stratified holdout (JSH), defined in `src/jsh_fit.py`: the 6,501,243 big-6
episodes are split 70/30 stratified jointly on (primary substance x outcome), giving
4,550,870 train / 1,950,373 holdout; the training side is subsampled to 1,000,000
(999,999 actual) on the same joint key; one `HistGradientBoostingClassifier`
(`max_iter=200`, `learning_rate=0.1`, `max_bins=255`, `random_state=0`) is fitted on that
cohort and scored **without refitting** on both the big-6 holdout and all 215,983
unseen-class episodes. Coverage, calibration, decision-curve, fairness and the 65+
diagnosis are all read off that single fit.

Independently re-fitted during this audit: train=4,550,870, cap=999,999,
holdout=1,950,373, unseen=215,983; holdout AUROC 0.791438, Brier 0.179173; unseen AUROC
0.777701, Brier 0.185933; all twelve per-substance coverage AUROCs reproduce the
committed `data/p1_coverage_ci.csv` to 4 decimal places (max absolute difference
0.000000).

**(b) The two temporal arms are year-restricted refits of the identical specification.**
`src/jsh_year_restricted.py` filters the big-6 pool by `YEAR` *before* applying the same
1,000,000-row joint-stratified cap, with the same features, same hyperparameters and same
seed. There is no 70/30 holdout in these arms because the test set is a different year
and is therefore already disjoint from the training pool. `YEAR < 2022` feeds the
temporal-coverage arm (`src/05`), `YEAR <= 2018` the COVID arm (`src/06`).

**(c) Documented exceptions that sit outside JSH.** Four, all deliberate:

1. **Unseen COVID arm** (`src/07b_unseen_deployability.py`, `covid_temporal()`), an
   intentional *within-unseen* replication that trains a separate model on 125,912
   unseen-class 2015-2018 episodes (seed 42) and tests on unseen-class episodes in
   2019/2020/2022. Folding it into JSH would replace the question rather than answer it:
   the arm asks whether the unseen cohort's own temporal decay is steeper than the trained
   substances', which requires a model fitted inside that cohort. Kept by user decision
   and disclosed in the manuscript.
2. **Pooling-cost baselines** (`src/02_baselines_single_pooled.py`), the single-vs-pooled
   contrast uses six per-substance 70/30 splits and a pooled model trained on the union of
   those six 70% halves. This is what makes 0.776 (pooling cost) a different quantity from
   0.777 (canonical JSH pooled mean); see the note below.
3. **TabPFN protocols** (`src/08a`, `src/08b`, `src/08c`, `src/08d`): matched 10,000/2,000
   train/test budget and maximum-training-size (TabPFN-3 at 100,000 rows vs GBT at up to
   10^6), each with its own splits by design.
4. **Library check** (`src/10_gbt_implementation_check.py`): XGBoost vs HistGBT at matched
   settings on the two largest substances, and the 27-feature expansion sensitivity
   (`docs/FEATURE_EXPANSION.md`, 2020 held out).

## Two pooled means that legitimately coexist

`0.776` and `0.777` are different quantities and neither is an error:

- **0.776** is the *pooling-cost* value from `src/02_baselines_single_pooled.py`: the mean
  over six substances of a model trained on the union of the six per-substance 70% halves
  (0.776083). It exists to be differenced against the single-substance mean 0.785, giving
  the -0.008 pooling penalty.
- **0.777** is the *canonical JSH pooled mean* (0.776923), the reference line in Fig. 1,
  from `data/p1_phase1_pooled_jsh.csv`.

Both appear in the manuscript, each in its own context; conflating them would be the error.

## Bootstrap resample convention

All confidence intervals in the committed CSVs use **300 resamples**. This was verified
empirically during the audit by re-running each arm at 300, 400 and 500 resamples and
comparing to the committed bounds: the temporal-coverage arm matched at 300 in 12 of 12
rows and 0 of 12 at 500, and the COVID arm matched at 300 in 3 of 3 rows and 0 of 3 at 400.
The committed `src/05` and `src/04` declared `BOOT_N = 500` and `src/06` declared
`BOOT_N = 400`, which did not reproduce the shipped CIs. **The scripts have been
standardised on `BOOT_N = 300`** to match the data. Point AUROCs are unaffected by this;
only CI bounds were ever in question, and the shipped bounds were already the 300-resample
values.

Test-set caps differ per arm and are unchanged: 100,000 in `src/04`, `src/05` and
`src/07_fairness_audit.py`, 150,000 in `src/06`.

## The mapping

| Manuscript claim | Value | Producing script | Training assembly | Data file |
|---|---|---|---|---|
| Analytic sample N | 6,717,226 | `src/01_build_analysis_table.py` | n/a (cohort construction) | `data/teds_d_analysis_2015_2022.parquet` |
| Overall dropout rate | 0.398 | `src/01_build_analysis_table.py` | n/a (cohort construction) | `data/teds_d_analysis_2015_2022.parquet` |
| Dropout of resolved episodes | 39.8% | `src/01_build_analysis_table.py` | n/a (cohort construction) | `data/teds_d_analysis_2015_2022.parquet` |
| Big-6 pool N | 6,501,243 | `src/jsh_fit.py` | JSH (canonical) | `data/teds_d_analysis_2015_2022.parquet` |
| Unseen-class N | 215,983 | `src/jsh_fit.py` | JSH (canonical) | `data/teds_d_analysis_2015_2022.parquet` |
| JSH train side | 4,550,870 | `src/jsh_fit.py` | JSH (canonical) | `data/teds_d_analysis_2015_2022.parquet` |
| JSH big-6 holdout | 1,950,373 | `src/jsh_fit.py` | JSH (canonical) | `data/teds_d_analysis_2015_2022.parquet` |
| JSH training subsample | 1,000,000 (999,999 actual) | `src/jsh_fit.py` | JSH (canonical) | `data/teds_d_analysis_2015_2022.parquet` |
| Table II: all 51 cohort groups (153 numeric cells) | see table | derived from the analysis parquet (no script emits the LaTeX; the table body is hand-typed and every cell re-verified against the parquet) | n/a (cohort construction) | `data/p1_cohort_table.csv, data/teds_d_class_sizes_2015_2022.csv` |
| Small classes share of cohort | 3.2% | `src/01_build_analysis_table.py` | n/a | `data/teds_d_class_sizes_2015_2022.csv` |
| Cocaine / benzodiazepines size ratio | 4.9x | `src/01_build_analysis_table.py` | n/a | `data/teds_d_class_sizes_2015_2022.csv` |
| Substance base-rate range | 0.170-0.531 | `src/01_build_analysis_table.py` | n/a | `data/teds_d_class_sizes_2015_2022.csv` |
| Unseen mean AUROC (headline) | 0.777 | `src/04_coverage_unseen.py` | JSH (canonical) | `data/p1_coverage_ci.csv` |
| Unseen AUROC range | 0.745-0.806 | `src/04_coverage_unseen.py` | JSH (canonical) | `data/p1_coverage_ci.csv` |
| Benzodiazepines / barbiturates / OTC AUROC | 0.745 / 0.748 / 0.751 | `src/04_coverage_unseen.py` | JSH (canonical) | `data/p1_coverage_ci.csv` |
| Fig. 1 big-6 reference line | 0.777 (0.776923) | `src/09_paper_figures.py` | JSH (canonical) | `data/p1_phase1_pooled_jsh.csv` |
| Six classes above / six below reference | 6 / 6 | `src/04_coverage_unseen.py` | JSH (canonical) | `data/p1_coverage_ci.csv + data/p1_phase1_pooled_jsh.csv` |
| Cohort-level unseen AUROC | 0.778 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_unseen_summary.csv` |
| Cohort-level unseen Brier | 0.186 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_unseen_summary.csv` |
| Unseen base rate | 0.404 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_unseen_summary.csv` |
| Big-6 holdout AUROC | 0.791 | `src/03_calibration_dca.py` | JSH (canonical) | `data/fairness_preds.parquet` |
| Big-6 holdout Brier | 0.179 | `src/03_calibration_dca.py` | JSH (canonical) | `data/fairness_preds.parquet` |
| Big-6 holdout base rate | 0.397 | `src/03_calibration_dca.py` | JSH (canonical) | `data/fairness_preds.parquet` |
| Mean absolute decile calibration gap | 0.015 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_calibration_unseen_deciles.csv` |
| Max decile calibration gap | 0.046 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_calibration_unseen_deciles.csv` |
| Net benefit at threshold 0.20 | 0.275 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_dca_realdata_unseen.csv` |
| Net benefit at threshold 0.50 | 0.126 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_dca_realdata_unseen.csv` |
| Net benefit positive through 0.80 | 0.020 at 0.80 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_dca_realdata_unseen.csv` |
| Treat-all crosses zero | 0.41 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_dca_realdata_unseen.csv` |
| GBT single-substance mean AUROC | 0.785 | `src/02_baselines_single_pooled.py` | per-substance 70/30 (pooling-cost arm) | `data/p1_phase1_single_pooled.csv` |
| GBT pooled mean AUROC (pooling cost) | 0.776 | `src/02_baselines_single_pooled.py` | union of six per-substance 70% halves | `data/p1_phase1_single_pooled.csv` |
| Pooled minus single | -0.008 | `src/02_baselines_single_pooled.py` | pooling-cost arm | `data/p1_phase1_single_pooled.csv` |
| Per-substance pooling loss range | 0.004 (Alcohol) - 0.012 (Other opioids) | `src/02_baselines_single_pooled.py` | pooling-cost arm | `data/p1_phase1_single_pooled.csv` |
| Logistic single / pooled mean | 0.753 / 0.742 | `src/02_baselines_single_pooled.py` | pooling-cost arm | `data/p1_phase1_single_pooled.csv` |
| GBT over logistic | 0.031 - 0.034 | `src/02_baselines_single_pooled.py` | pooling-cost arm | `data/p1_phase1_single_pooled.csv` |
| Balanced-vs-proportional pooling max shift | 0.004 (mean +0.001) | `src/02c_pooled_cohorts.py` | balanced-cohort sensitivity | `data/p1_pooled_cohorts.csv` |
| Temporal coverage mean AUROC (2022) | 0.785 | `src/05_coverage_temporal.py` | year-restricted JSH (YEAR<2022) | `data/p1_coverage_temporal2022.csv` |
| Trained-substance AUROC 2019/2020/2022 | 0.786 / 0.769 / 0.768 | `src/06_covid_temporal.py` | year-restricted JSH (YEAR<=2018) | `data/p1_covid_temporal.csv` |
| Trained-substance onset step / later decay | 0.017 / 0.001 | `src/06_covid_temporal.py` | year-restricted JSH (YEAR<=2018) | `data/p1_covid_temporal.csv` |
| Unseen AUROC 2019/2020/2022 | 0.789 / 0.738 / 0.708 | `src/07b_unseen_deployability.py` | EXCEPTION: within-unseen refit, 2015-2018, n=125,912, seed 42 | `data/p1_covid_temporal_unseen.csv` |
| Unseen onset step / later decay | 0.051 / 0.030 | `src/07b_unseen_deployability.py` | EXCEPTION: within-unseen refit | `data/p1_covid_temporal_unseen.csv` |
| Unseen test n by era | 35,424 / 26,861 / 27,786 | `src/07b_unseen_deployability.py` | EXCEPTION: within-unseen refit | `data/p1_covid_temporal_unseen.csv` |
| Combined shift cost over 7 years | 0.081 | `src/07b_unseen_deployability.py` | EXCEPTION: within-unseen refit | `data/p1_covid_temporal_unseen.csv` |
| Pre-2020 / 2020 / 2022 base rates | ~0.39 / 0.427 / 0.426 | `src/06_covid_temporal.py` | n/a (test-set rates) | `data/p1_covid_temporal.csv` |
| Trained-substance calibration gap ceiling | +/-0.022 | `src/07_fairness_audit.py` | JSH (canonical) | `data/p1_fairness_subgroups.csv` |
| Trained-substance age gradient | 0.728 (12-14) - 0.837 (65+) | `src/07_fairness_audit.py` | JSH (canonical) | `data/p1_fairness_subgroups.csv` |
| Unseen strata count | 22 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| Unseen AUROC span across strata | 0.677-0.798 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| Unseen AUROC sd across strata | 0.032 (sample sd) | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| Male / female AUROC (unseen) | 0.775 / 0.780 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| Youngest / oldest band AUROC (unseen) | 0.707 (12-14) / 0.791 (65+) | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| Smallest stratum (Alaska Native) | n=260 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| 21 of 22 strata calibration bound | 0.044 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| 65+ calibration gap | 0.140 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| 65+ band n and observed rate | n=5,061, obs 0.165 | `src/07c_age65_anomaly.py` | JSH (canonical) | `data/p1_age65_anomaly_diagnosis.csv` |
| State-36 x Other_tranquilizers cell | n=3,234, obs 0.067 | `src/07c_age65_anomaly.py` | JSH (canonical) | `data/p1_age65_anomaly_diagnosis.csv` |
| Same-state big-6 65+ observed rate | 0.277 | `src/07c_age65_anomaly.py` | JSH (canonical) | `data/p1_age65_anomaly_diagnosis.csv` |
| 65+ excluding that cell | under-predicts by 0.010 | `src/07c_age65_anomaly.py` | JSH (canonical) | `data/p1_age65_anomaly_diagnosis.csv` |
| Eight strata n>20,000: calibration / AUROC | <=0.016 / 0.772-0.798 | `src/07b_unseen_deployability.py` | JSH (canonical) | `data/p1_fairness_unseen.csv` |
| Matched-budget TabPFN wins | 18 of 18, mean +0.018 | `src/08a_tabpfn_vs_gbt_compute.py` | EXCEPTION: matched 10k/2k protocol | `data/p1_tabpfn_vs_gbt.csv` |
| Maximum-training-size TabPFN wins | 8 of 18, mean -0.0001 | `src/08b_prep_uncapped.py + src/08c_gpu_worker.py` | EXCEPTION: maximum-training-size protocol | `data/p1_tabpfn_uncapped.csv` |
| CIs including zero / significant | 15 of 18 / 3 | `src/08c_gpu_worker.py` | EXCEPTION: maximum-training-size protocol | `data/p1_tabpfn_uncapped.csv` |
| Largest GBT fit time | 346 s on 48 CPU cores | `src/08b_prep_uncapped.py` | EXCEPTION: maximum-training-size protocol | `data/p1_tabpfn_uncapped.csv` |
| TabPFN 100,000-row pass | 38.9 s (~39 s), 2.07 GiB VRAM | `src/08d_gpu_scaling_probe.py` | EXCEPTION: GPU scaling probe | `data/gpu_probe.csv` |
| CPU-vs-GPU speedup at 10k | 60x (187.5 s vs 3.1 s) | `src/08d_gpu_scaling_probe.py` | EXCEPTION: GPU scaling probe | `data/gpu_probe.csv + data/p1_tabpfn_vs_gbt.csv` |
| Shared ~0.78 AUROC ceiling | ~0.78 | `src/08c_gpu_worker.py` | EXCEPTION: maximum-training-size protocol | `data/p1_tabpfn_uncapped.csv` |
| XGBoost vs HistGBT agreement | within 0.002 | `src/10_gbt_implementation_check.py` | EXCEPTION: library check (1M Alcohol / 588,619 Cannabis) | `data/p1_gbt_implementation_check.csv` |
| Feature-expansion sensitivity | +0.005 (0.820 to 0.825) | `(2020 single-year check)` | EXCEPTION: 27-feature sensitivity, 2020 held out | `docs/FEATURE_EXPANSION.md` |
