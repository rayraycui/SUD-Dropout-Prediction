#!/usr/bin/env bash
# Reproduce every number, figure and table behind the manuscript.
# The manuscript sources ship separately, in the Overleaf package.
# Starts from the committed analysis table; pass --from-source to rebuild from raw TEDS-D.
set -euo pipefail
PY="${PY:-python}"

if [[ "${1:-}" == "--from-source" ]]; then
  echo "== 00: download TEDS-D public-use files into data/raw/"
  $PY src/00_download_teds.py
  echo "== 01: build the analysis table"
  $PY src/01_build_analysis_table.py
else
  echo "== using committed data/teds_d_analysis_2015_2022.parquet"
  echo "   (pass --from-source to rebuild it from the raw SAMHSA files)"
fi

echo "== 11: substance-specific vs pooled, uncapped (the main comparison)"
$PY src/11_single_vs_pooled_uncapped.py

echo "== 12: Figs. 2 and 3"
$PY src/12_single_vs_pooled_figures.py

echo "== 18: Fig. 1, the study-overview schematic"
$PY src/18_overview_figure.py

echo "== 16: Table II and Table III bodies"
$PY src/16_build_tables.py

echo "== 17: cross-file consistency check"
$PY src/17_check_consistency.py

echo "== 13-15: reported sensitivity analyses"
$PY src/13_early_stopping_sensitivity.py
$PY src/14_nominal_code_sensitivity.py
$PY src/15_categorical_bootstrap.py

echo "== done. Figures in figures/, table bodies in tables/, result CSVs in data/."
