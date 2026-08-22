# Data directory

## Committed
- `teds_d_analysis_2015_2022.parquet`: harmonized analysis table, 6,717,226 episodes,
  15 intake features + label `y` (1=dropout) + `subname`/`is_big6`/`YEAR`. Built by
  `src/01_build_analysis_table.py` from the raw TEDS-D files.
- `fairness_preds.parquet`: held-out test predictions (y, p, RACE, GENDER, AGE, STFIPS,
  subname) from the pooled GBT; input to the fairness audit. Produced by
  `src/07a_fairness_score.py`.
- Per-figure result CSVs: `p1_coverage_ci.csv`, `p1_coverage_temporal2022.csv`,
  `p1_phase1_single_pooled.csv`, `p1_dca_realdata.csv`, `p1_covid_temporal.csv`,
  `p1_fairness_subgroups.csv`, `p1_fairness_states.csv`, `p1_tabpfn_vs_gbt.csv`.

## Not committed (fetch yourself)
- `raw/`: TEDS-D public-use CSVs (2015-2020, 2022). Download with
  `python src/00_download_teds.py`. ~2 GB. Public, de-identified; SAMHSA terms apply.
- `tabpfn_v3_default.ckpt`: TabPFN-3 checkpoint for the optional benchmark (step 08a).

The analysis table is the single source all analyses read; you do not need `raw/` unless
rebuilding the table or running `src/05` (which re-reads raw for the 2022 temporal test).
