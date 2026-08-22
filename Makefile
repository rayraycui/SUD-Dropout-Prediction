# Reproduction pipeline. `make all` runs everything from the committed analysis table.
# To rebuild the table from raw TEDS-D files, run `make data` first (needs downloads).
PY=python

.PHONY: all data figures clean
all: figures

# --- rebuild analysis table from raw (optional; committed table lets you skip this) ---
data: data/teds_d_analysis_2015_2022.parquet
data/raw/.downloaded:
	$(PY) src/00_download_teds.py && touch data/raw/.downloaded
data/teds_d_analysis_2015_2022.parquet: data/raw/.downloaded
	$(PY) src/01_build_analysis_table.py

# --- analyses + figures (read the committed parquet) ---
figures:
	$(PY) src/02_baselines_single_pooled.py
# NOTE: src/02b_plot_regimes.py is referenced historically but is NOT present in this repo.
#	$(PY) src/02b_plot_regimes.py
	$(PY) src/02c_pooled_cohorts.py
	$(PY) src/03_calibration_dca.py
	$(PY) src/04_coverage_unseen.py
	$(PY) src/05_coverage_temporal.py
	$(PY) src/06_covid_temporal.py
	$(PY) src/07a_fairness_score.py
	$(PY) src/07_fairness_audit.py
	$(PY) src/07b_unseen_deployability.py
	$(PY) src/07c_age65_anomaly.py
	$(PY) src/10_gbt_implementation_check.py
	$(PY) src/08_tabpfn_vs_gbt_plot.py
	$(PY) src/09_paper_figures.py
	$(PY) src/09b_paper_figures_unseen.py
	@echo "Figures written to figures/. (TabPFN compute step 08a is optional; see README.)"

clean:
	rm -f figures/*.png
