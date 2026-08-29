# Reproduction pipeline for the figures, tables and results behind the manuscript.
# The manuscript sources themselves ship separately, in the Overleaf package.
# The committed analysis table lets you start at `results` without re-downloading.

PY ?= python

.PHONY: all results figures tables check sensitivity data clean

all: results figures tables check

data:                       ## fetch TEDS-D and rebuild the analysis table
	$(PY) src/00_download_teds.py
	$(PY) src/01_build_analysis_table.py

results:                    ## main comparison: Table III inputs
	$(PY) src/11_single_vs_pooled_uncapped.py

figures: results            ## Figs. 1, 2 and 3 -> figures/
	$(PY) src/12_single_vs_pooled_figures.py
	$(PY) src/18_overview_figure.py

tables:                     ## Table II and Table III bodies as LaTeX -> tables/
	$(PY) src/16_build_tables.py

check: tables               ## assert every data file agrees with the results
	$(PY) src/17_check_consistency.py

sensitivity:                ## the three reported sensitivity analyses
	$(PY) src/13_early_stopping_sensitivity.py
	$(PY) src/14_nominal_code_sensitivity.py
	$(PY) src/15_categorical_bootstrap.py

clean:
	rm -rf src/__pycache__
