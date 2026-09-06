# One Pooled Model or Seventeen Substance-Specific Models?

Reproduction code and data for the manuscript

*Predicting SUD treatment dropout on 6.68 million national treatment episodes.*

The question is not how accurate dropout prediction can be, but **how many models a

treatment system should run**. A national reporting system records 17 primary substances

of very uneven size. Fitting one model per substance means 17 models to tune, validate,

monitor and re-fit, and the rarest substances still get little data of their own. This repository

contains everything needed to reproduce that comparison.


## Getting the files

The analysis table (`data/teds_d_analysis_2015_2022.parquet`, ~39 MB) is 93% of the

archive, so the repository is distributed two ways:

- **`teds_dropout_repo.zip`** (~33 MB) contains everything, including the analysis table.

- **`teds_dropout_repo_code_only.zip`** (~2 MB) contains everything except the analysis

  table. Use this if the larger archive fails to download or unzip, then either drop

  `teds_d_analysis_2015_2022.parquet` into `data/` or rebuild it with

  `python src/00_download_teds.py && python src/01_build_analysis_table.py`.

The model-fitting, sensitivity, cohort/table, overview and consistency steps require the

analysis table. Step 12 (Figs. 2 and 3) can be regenerated from the committed result CSVs

alone.

## Headline results

| | Boosting | Logistic |
|---|---:|---:|
| Substance-specific (17 models) | 0.7759 | 0.7494 |
| Pooled (1 model) | **0.7862** | 0.7439 |

Unweighted mean AUROC across the 17 held-out substance-specific test sets, giving each substance equal weight.

1. **One model instead of 17 improves mean performance across substances.** Pooled boosting reaches

   0.7862 against 0.7759 for the 17 substance-specific models, an exact

   unweighted mean improvement of 0.0104 AUROC.

2. **Weighting by test-set size reverses the comparison.** Episode-weighted, pooled boosting reaches

   0.7857 versus 0.7903 for the substance-specific models, an exact difference of -0.0046 AUROC.

   This occurs because the six highest-volume substances, which account for 97.3% of episodes,

   all favor substance-specific training; pooling reduces AUROC by 0.006 on average across those six.

3. **Pooled performance is more uniform.** It spans 0.752-0.815

   (SD 0.017) against 0.713-0.839 (SD 0.032) for

   substance-specific models, so the substance served worst is served better.

4. **The gain concentrates where data are thin.** The four substances with fewer than

   5,000 training episodes gain 0.034 AUROC on average. The lowest-volume, other

   tranquilizers (435 training / 187 test episodes), goes from 0.713 alone to 0.815 pooled.

   Barbiturates is the exception, losing 0.037.

   Across substances, the pooling gain correlates with training size at

   r = -0.616 (log10 episodes).

5. **The learner matters more than the strategy.** Boosting beats logistic regression in

   33 of 34 substance-by-strategy comparisons, with mean AUROC gaps of

   0.026 under substance-specific training and 0.042 under pooled training.

## Data

TEDS-D public-use files, discharge years 2015-2020 and 2022 (**2021 excluded**: its

public-use CSV ships text labels rather than the numeric codes used in every other year).

- **Cohort:** 6,679,648 episodes across 17 primary substances, dropout rate 0.397.

- **Excluded:** TEDS-D SUB1 code 19, *other drugs* (37,578 episodes), a heterogeneous

  residual category rather than a specific substance class.

- **Pooled training set:** the six highest-volume substances, 4,550,867 episodes,

  uncapped (6,501,243 episodes total, 97.3% of the cohort).

  The eleven lower-volume substances (178,405 episodes) contribute none.

- **Target:** binary, dropout (`REASON == 2`, left against professional advice) versus

  treatment completed (`REASON == 1`). Other discharge reasons are excluded.

- **Features:** the 15 intake-time predictors in Table I. The primary substance itself is

  never a feature, so one fitted model applies to any substance unmodified.

SAMHSA distributes the raw files; this repository does **not** redistribute them.

`src/00_download_teds.py` fetches them and `data/raw/MANIFEST.csv` pins the exact bytes

(URL, size and SHA-256 per file) used here.

## Reproduce

```bash
pip install -r requirements.txt

python src/00_download_teds.py         # fetch TEDS-D into data/raw/ (not committed)
python src/01_build_analysis_table.py  # -> data/teds_d_analysis_2015_2022.parquet

python src/11_single_vs_pooled_uncapped.py    # the main comparison (Table III inputs)
python src/12_single_vs_pooled_figures.py     # -> figures/fig1, figures/fig2 (Figs. 2 and 3)
python src/18_overview_figure.py               # -> figures/fig0_overview (Fig. 1)
python src/16_build_tables.py                  # -> tables/table2_body.tex, table3_body.tex
python src/17_check_consistency.py             # cross-file consistency checks

python src/13_early_stopping_sensitivity.py   # reported early-stopping sensitivity
python src/14_nominal_code_sensitivity.py     # additional nominal-code robustness check
python src/15_categorical_bootstrap.py        # categorical-bootstrap cross-check
```

`PY=python3 ./run_all.sh` runs the full sequence; `make all` runs the same pipeline if using

the included Makefile. Step 13 is the reported early-stopping sensitivity analysis; steps

14-15 are additional robustness checks and are independent of each other.

The committed parquet lets every downstream step run without re-downloading: start at

step 11 if you only want the results, or at step 00 to rebuild from source.

### Runtime

Step 11 fits 36 models with no subsampling (the pooled boosted fit uses

4,550,867 episodes) and computes bootstrap intervals, so it is the expensive

main-analysis step: roughly 8-10 minutes on the 48-core development machine in the latest

run. Steps 13-15 refit related designs and can each take several minutes. Steps 12, 16 and

18 are much faster.

## What produces what

| Manuscript element / analysis | Script | Data |
|---|---|---|
| Table I (predictors) | - | narrative, `docs/teds_d_data_dictionary.md` |
| Table II (cohort) | `src/16_build_tables.py` | `data/teds_d_analysis_2015_2022.parquet` |
| Table III (AUROC) | `src/16_build_tables.py` | `data/p1_plan_eval_17substance_gbt_vs_logistic.csv` |
| Fig. 1 (study overview) | `src/18_overview_figure.py` | parquet + `data/p1_plan_eval_17substance_gbt_vs_logistic.csv` |
| Fig. 2 (single vs pooled) | `src/12_single_vs_pooled_figures.py` | same CSV |
| Fig. 3 (pooling difference) | `src/12_single_vs_pooled_figures.py` | same CSV |
| Early-stopping sensitivity | `src/13_early_stopping_sensitivity.py` | `data/p1_early_stopping_*.csv` |
| Nominal-code robustness check | `src/14_nominal_code_sensitivity.py` | `data/p1_categorical_sensitivity.csv` |
| Categorical-bootstrap cross-check | `src/15_categorical_bootstrap.py` | `data/p1_categorical_verdicts.csv` |
| Feature-set sensitivity (15 vs 27) | - | `docs/FEATURE_EXPANSION.md` |
| Cross-file consistency check | `src/17_check_consistency.py` | all of `data/`, `tables/`, `figures/fig0_overview.svg` |

Figure numbering in the manuscript: Fig. 1 is the study-overview schematic

(`figures/fig0_overview.pdf`), Fig. 2 is `figures/fig1_single_vs_pooled.png` and Fig. 3 is

`figures/fig2_pooling_delta.png`. The filenames predate the overview figure, so they run

one behind the figure numbers. Step 18 always regenerates the SVG; refreshing the PDF also

requires `rsvg-convert`, CairoSVG or Inkscape.

`src/16_build_tables.py` regenerates the Table II and Table III bodies as LaTeX, plus

`data/teds_d_class_sizes_2015_2022.csv`. `src/17_check_consistency.py` asserts that the

committed result files, analysis table, generated table bodies and overview values agree,

and exits nonzero if they do not. If `PAPER_TEX` points to the manuscript source, it also

checks manuscript values; otherwise the manuscript check is skipped. Run it after any

change to `data/`.

## Models

Identical settings for every substance and both training strategies, so any difference

between strategies reflects the training data rather than tuning:

- **Boosting:** `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,

  max_bins=255, random_state=42)`, scikit-learn 1.9.0. The 10 nominal predictors

  (`SUB2`, `SUB3`, `ROUTE1`, `SERVICES`, `DSMCRIT`, `PSOURCE`, `GENDER`, `RACE`,

  `EMPLOY`, `STFIPS`) are declared categorical; the 5 ordinal predictors (`FREQ1`,

  `FRSTUSE1`, `NOPRIOR`, `AGE`, `EDUC`) retain their natural numeric ordering.

- **Logistic:** `LogisticRegression(penalty='l2', C=1.0, solver='saga', max_iter=200)`

  behind `OneHotEncoder(handle_unknown='ignore', min_frequency=50)`.

- **Splits:** 70/30 per substance, stratified on the outcome, seed 0. No subsampling on

  either strategy. The 30% halves are never used for fitting, so the comparison is paired

  substance by substance.

- **Intervals:** 2,000 paired bootstrap resamples on the shared test set, computed for the

  eleven lower-volume substances. A difference counts as conclusive only when its interval

  excludes zero; the six highest-volume comparisons are reported without bootstrap intervals.

## Reported sensitivity analyses

- **Early stopping.** scikit-learn enables internal early stopping only above 10,000

  training episodes, a threshold inside the size range studied here. All 18

  boosted models were refitted with it forced off and forced on. No substance changed the

  sign of its pooled-versus-specific difference under either uniform setting; disabling

  early stopping changed the differences by approximately 0.005 AUROC or less.

- **Nominal-code ordering.** As an additional robustness check, treating the nominal fields

  as ordinary integer codes rather than declared categorical features reduced episode-

  weighted AUROC by 0.004 for substance-specific models and 0.006 for the pooled model.

  The overall size-dependent pattern persisted, although several lower-volume point

  estimates and bootstrap verdicts changed.

- **Categorical-bootstrap cross-check.** An independent recomputation for the eleven

  lower-volume substances under the primary categorical specification reproduces the main

  GBT deltas and confidence intervals: pooling is conclusive for 5 substances,

  substance-specific training for 1, and 5 are inconclusive.


## Layout

```
src/       numbered pipeline, run in order
data/      committed analysis table and result CSVs; raw/ holds the manifest only
figures/   the three manuscript figures, regenerated by src/12 and src/18
tables/    Table II and Table III bodies as LaTeX, regenerated by src/16
assets/    the overview figure's SVG layout template, read by src/18
docs/      data dictionary, TEDS-D codebook, feature-set sensitivity note
```

The manuscript sources (`paper.tex`, `paper.pdf`, `IEEEtran.cls`) are not in this

repository; they ship in the Overleaf package alongside copies of the three figures.

This repository holds everything needed to regenerate those figures, the table bodies

and every number in them from the data.

## Limitations

The unit is a treatment episode, not a person, and the outcome is an administrative

discharge code rather than a clinical assessment of recovery. Bootstrap intervals for the

lowest-volume substances are wide (half-widths up to approximately 0.061 AUROC), so

per-substance verdicts there are indicative rather than settled. The comparison is

retrospective and within one national data source.

## License

Code is released under the MIT License (`LICENSE`). TEDS-D data is governed by SAMHSA's

public-use data terms and is not redistributed here.
