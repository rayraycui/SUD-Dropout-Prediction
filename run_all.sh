#!/usr/bin/env bash
# Full reproduction pipeline. Assumes deps installed (pip install -r requirements.txt).
#
# TRAINING ASSEMBLY. Every cross-sectional stage below scores ONE canonical fit, the
# joint-stratified holdout (JSH) defined in src/jsh_fit.py. That file and its year-restricted
# variant src/jsh_year_restricted.py are IMPORTED MODULES, not pipeline stages -- the numbered
# scripts call them. They are listed here so the src tree is fully accounted for:
#
#   src/jsh_fit.py             canonical JSH: big-6 70/30 joint-stratified split
#                              (4,550,870 train / 1,950,373 holdout), train side capped at
#                              1,000,000 on the same joint key, one GBT fitted and scored
#                              without refitting on the holdout and all 215,983 unseen episodes
#   src/jsh_year_restricted.py identical specification with the big-6 pool filtered by YEAR
#                              before the cap; used by stages 05 (YEAR<2022) and 06 (YEAR<=2018)
#
# Run either directly to perform the fit and print the cohort sizes as a self-check:
#   python src/jsh_fit.py
#
# BOOTSTRAP. All confidence intervals use 300 resamples (BOOT_N = 300 in stages 04, 05, 06
# and n_boot=300 in 07_fairness_audit.py). Test-set caps differ per arm by design:
# 100,000 rows in 04/05/07_fairness_audit, 150,000 in 06.
set -euo pipefail
PY=${PYTHON:-python}

# Step 0-1: build the analysis table from raw TEDS-D files.
# Skip if data/teds_d_analysis_2015_2022.parquet is already present (it is committed).
if [ ! -f data/teds_d_analysis_2015_2022.parquet ]; then
  echo ">> downloading TEDS-D and building analysis table"
  $PY src/00_download_teds.py
  $PY src/01_build_analysis_table.py
fi

echo ">> baselines (single/pooled)";  $PY src/02_baselines_single_pooled.py
# NOTE: src/02b_plot_regimes.py is referenced historically but is NOT present in this repo.
# The figure it produced (figures/p1_phase1_regimes.png) is committed; the manuscript does
# not use it. Uncomment only if you restore the script.
# echo ">> regimes figure";              $PY src/02b_plot_regimes.py
echo ">> pooled-cohort sensitivity";    $PY src/02c_pooled_cohorts.py
echo ">> calibration + DCA";            $PY src/03_calibration_dca.py
echo ">> coverage of unseen substances"; $PY src/04_coverage_unseen.py
echo ">> temporal coverage (2022)";     $PY src/05_coverage_temporal.py
echo ">> COVID-aware transportability"; $PY src/06_covid_temporal.py
echo ">> fairness scoring";             $PY src/07a_fairness_score.py
echo ">> fairness audit figure";        $PY src/07_fairness_audit.py
echo ">> unseen-class deployability";   $PY src/07b_unseen_deployability.py
echo ">> 65+ calibration-gap diagnosis"; $PY src/07c_age65_anomaly.py
echo ">> GBT library check (XGBoost vs HistGBT)"; $PY src/10_gbt_implementation_check.py
echo ">> TabPFN-vs-GBT figure";         $PY src/08_tabpfn_vs_gbt_plot.py
echo ">> manuscript figures (print width)"; $PY src/09_paper_figures.py
echo ">> manuscript figures (unseen-class panels)"; $PY src/09b_paper_figures_unseen.py

echo ">> done. Review figures in figures/; manuscript figures in paper/."
echo ">> (Optional) TabPFN-3 compute benchmark: pip install -r requirements-tabpfn.txt"
echo ">>           then: $PY src/08a_tabpfn_vs_gbt_compute.py  (slow on CPU, resumable)"
echo ">> (Optional, needs CUDA) maximum-training-size protocol + GPU probe:"
echo ">>           $PY src/08b_prep_uncapped.py && $PY src/08c_gpu_worker.py && $PY src/08d_gpu_scaling_probe.py"
