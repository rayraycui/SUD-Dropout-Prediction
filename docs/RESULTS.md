# P1: Pan-Substance Treatment-Dropout Prediction on TEDS-D: Results

**Project:** Pan-substance treatment-dropout prediction, evaluated with
single-model **zero-shot coverage** of unseen substance classes and a TabPFN-3 vs
gradient-boosting benchmark.

> **Terminology note.** Coverage here is an *evaluation protocol*: train the pooled
> model on the six high-volume substances, then apply it directly to a held-out small
> class with **no adaptation, no fine-tuning, no learned alignment**. It is zero-shot
> direct generalization, **not** a transfer-learning *method*. Because pooling does not
> beat substance-specific training (§2, Δ = −0.008), there is no cross-substance gain for
> a stronger adaptation method to capture; a single importance-weighted / few-shot control
> is noted as optional future work (§10), not run here.
**Data:** SAMHSA Treatment Episode Data Set, Discharges (TEDS-D), public-use files.
**Target venue framing:** IEEE BIBM 2026 UGHS (feasibility-scoped, open data, CPU-only).

> **Training assembly (read this first).** Every cross-sectional number below comes from
> **one** fit: the **joint-stratified holdout (JSH)**, defined in `src/jsh_fit.py`. The
> 6,501,243 big-6 episodes are split 70/30 stratified *jointly* on (primary substance ×
> outcome) → 4,550,870 train / 1,950,373 holdout; the train side is subsampled to
> 1,000,000 (999,999 actual) on the same joint key; a single
> `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_bins=255,
> random_state=0)` is fitted there and scored **without refitting** on both the big-6
> holdout and all 215,983 unseen-class episodes. Seed 0 throughout.
>
> The two temporal arms are **year-restricted JSH** (`src/jsh_year_restricted.py`), the
> identical specification with the big-6 pool filtered by `YEAR` before the cap
> (`YEAR < 2022` for temporal coverage, `YEAR <= 2018` for COVID). They carry no 70/30
> holdout because the test year is already disjoint from the training pool.
>
> Four arms sit **outside** JSH by design and are labelled as exceptions where they appear:
> the unseen COVID arm (§6b), the single-vs-pooled baselines (§2), the TabPFN protocols
> (§7), and the XGBoost library check. `docs/PROVENANCE.md` maps every reported number to
> its fit and data file.
>
> **Bootstrap convention.** All committed CIs use **300 resamples** (verified empirically:
> the temporal-coverage arm reproduces at 300 in 12/12 rows and 0/12 at 500; the COVID arm
> 3/3 at 300 and 0/3 at 400). `src/04`, `src/05` and `src/06` previously declared 500/500/400
> and have been standardised on `BOOT_N = 300` to match the shipped data. Point AUROCs are
> unaffected; only CI bounds were ever at issue. Test-set caps remain per-arm: 100,000 in
> `src/04`/`src/05`/`src/07_fairness_audit.py`, 150,000 in `src/06`.

> **Scope caveat (applies throughout).** This is a research-grade discovery/benchmark
> study on de-identified administrative episode records. The unit is a treatment
> *episode*, not a person; the label is an *administrative* discharge reason, not a
> clinical outcome. Nothing here is a diagnostic instrument or a deployed individual
> predictor. As a per-individual risk score the model has essentially no use; its value
> is population/program-level (caseload triage, resource targeting) and *scientific*
> (evidence that dropout risk is substance-general). Person-level deployment would
> require prospective, linked, IRB/DUA-governed validation.

---

## 1. Data and task

- **Years:** 2015–2020 and 2022 (7 years). **2021 excluded**, its public-use CSV ships
  text labels rather than the numeric codes used in every other year; a hand-built
  crosswalk was judged to carry unnecessary miscoding risk.
- **Analytic sample:** **6,717,226 episodes** after restricting to a resolvable outcome
  and a mappable primary substance. Overall dropout rate **0.398**.
- **Target:** binary, dropout (`REASON == 2`, "left against professional advice") vs
  treatment completed (`REASON == 1`). Episodes with other discharge reasons excluded.
- **Features:** **15 intake-time variables** (SUB2, SUB3, ROUTE1, FREQ1, FRSTUSE1,
  SERVICES, DSMCRIT, NOPRIOR, PSOURCE, AGE, GENDER, RACE, EDUC, EMPLOY, STFIPS). All are
  known at admission. `SUB1` (primary substance) is the coverage grouping variable and is
  **never a feature**. Leakage exclusions: REASON, SUB1_D, LOS, and all discharge (`_D`)
  fields.
- **Substance split (locked on the full pool):**
  - **Big-6** (self-trainable): Alcohol (2.49M), Heroin (1.56M), Cannabis (841k),
    Methamphetamine (776k), Other opioids (457k), Cocaine (376k).
  - **12 small classes** (coverage targets, too small to train a substance-specific model, exact
    pipeline labels): Benzodiazepines, Other_amphetamines, OTC, PCP, Non_rx_methadone,
    Other_hallucinogens, Barbiturates, Other_stimulants, Other_tranquilizers,
    Other_sedatives_hypnotics, Inhalants, Other_sedatives. Smallest big-6 (Cocaine 376k)
    = 4.9× largest small class (benzodiazepines 76k).

**Feature-set decision.** Expanding 15→27 pre-admission features improved AUROC by only
+0.005 (pooled) / +0.006 (mean per-substance), within bootstrap CI width. The 15-feature
set was kept as primary for parsimony and lower missingness.
(`docs/FEATURE_EXPANSION.md`)

---

## 2. Baseline models: single / pooled

Gradient-boosted trees (HistGBM) and logistic regression, each in two regimes, 1M-row
stratified training cap, bootstrap 95% CIs. (`p1_phase1_single_pooled.csv`,
`p1_phase1_regimes.png`)

| Regime | GBT mean AUROC | Logistic mean AUROC |
|---|---|---|
| single (train & test same substance) | **0.785** | 0.753 |
| pooled (train all, test one) | 0.776 | 0.742 |

- **GBT beats logistic by ~0.033** (range 0.028–0.043 across the twelve substance x regime cells): nonlinear
  interactions matter.
- **Pooling does not improve accuracy** (pooled − single = **−0.008** for GBT). Training
  on all substances buys nothing over a substance-specific single-substance model on data-rich
  substances. This is expected and important: the pan-substance value is *not* an
  accuracy gain.

---

## 3. Calibration and clinical utility (real data)

Pooled GBT scored on **1,950,373** held-out episodes. (`p1_calibration_dca.png`,
`p1_dca_realdata.csv`)

- **AUROC 0.791, Brier 0.179**, base rate 0.397.
- **Well-calibrated**: predicted-vs-observed dropout tracks the diagonal across all 10
  quantile bins (e.g. pred 0.05→obs 0.04 … pred 0.84→obs 0.86).
- **Decision-curve analysis (real data, replacing an earlier simulated illustration):**
  the model's net benefit exceeds both "treat all" and "treat none" across thresholds
  0.05–0.80. Advantage is largest in the 0.30–0.50 band:

  | Threshold | NB(model) | NB(treat-all) | Advantage |
  |---|---|---|---|
  | 0.30 | 0.216 | 0.139 | +0.077 |
  | 0.35 | 0.191 | 0.073 | +0.118 |
  | 0.40 | 0.168 | −0.004 | +0.168 |
  | 0.50 | 0.128 | −0.205 | +0.128 |

  At a 0.40 risk threshold the model captures meaningful net benefit where treating
  everyone would be actively harmful: this is the population/program-level utility.

---

## 4. Coverage: single model for unseen substances

The pooled big-6 model predicts dropout for the 12 held-out small classes it was never
trained on, the deployment argument for "one model covers everything."
(`p1_coverage_ci.png`, `p1_coverage_ci.csv`)

- **Full 7-year pool:** mean AUROC **0.777** (unweighted, 0.776925), range 0.745–0.806.
  Not uniform: 6 classes sit above the big-6 JSH reference (0.777, exactly 0.776923) and 6
  below (benzodiazepines 0.745, barbiturates 0.748, OTC 0.751 are the lowest), but every
  class is usefully predictable with no substance-specific training.
- **Temporal coverage (train <2022 → test small classes in 2022):** mean AUROC
  **0.785** (year-restricted JSH). 4 of 12 classes have CIs non-overlapping with the same-pool estimate
  (3 higher, 1 lower): aggregate coverage is stable, per-class differences exist.
  (`p1_coverage_temporal.png`, `p1_coverage_temporal2022.csv`)

This turns "covers unseen substances" from assertion into a measured result.

---

## 5. COVID-aware temporal transportability

The dropout base rate jumps from ~0.39 (pre-2020) to 0.427 at COVID onset and stays
elevated (0.426 in 2022). Trained on **2015–2018 only** (2019 held out clean to avoid
train/test overlap), tested per era: (`p1_covid_temporal.png`, `p1_covid_temporal.csv`)

| Test era | AUROC (95% CI) | Dropout rate |
|---|---|---|
| 2019 (pre-COVID, held-out) | 0.786 [0.783, 0.788] | 0.405 |
| 2020 (COVID) | 0.769 [0.764, 0.769] | 0.427 |
| 2022 (post-COVID) | 0.768 [0.765, 0.769] | 0.426 |

The pre-COVID model transports across the pandemic with a **0.017 one-time step down and
no further erosion** (0.001 from 2020 to 2022). This properly isolates the COVID
distribution shift from ordinary model drift, which a naive 2020→2022 comparison conflated.
Training assembly: year-restricted JSH, `YEAR <= 2018`, via `src/jsh_year_restricted.py`.

---

## 6. Subgroup fairness (trained substances)

Pooled GBT scored on 1.95M held-out episodes, broken down by race, sex, age, state.
(`p1_fairness.png`, `p1_fairness_subgroups.csv`, `p1_fairness_states.csv`)

- **Calibration is uniform across all demographic subgroups**: calibration gap (mean
  predicted − mean observed) within ±0.022 everywhere (max 0.0212, Asian, n=13,751) and
within ±0.005 for every group with n>50,000. No
  systematic over- or under-prediction of dropout by any race, sex, or age. This is the
  most important fairness property and it holds.
- **Sex:** essentially equal (M 0.789 / F 0.792).
- **Age:** monotonic gradient, harder for the youngest (12–14: 0.728), easier for the
  oldest (65+: 0.837).
- **Race:** most groups 0.74–0.85; the low AUROC outliers are the smallest-n groups
  (Alaska Native n=4,755 at 0.739, NHOPI n=9,411 at 0.739).
- **State:** widest spread (0.607–0.947, median 0.719, calibration gap up to ±0.10),
  expected, since STFIPS is a feature and state treatment systems and reporting practices
  differ substantially.

---

## 6b. Deployability on the twelve unseen classes

Sections 5 and 6 above measure calibration, utility, temporal robustness and fairness on the
**trained** substances. Because the paper's headline claim is coverage of substances the model
never saw, those four properties are re-measured on the unseen-class cohort
(n=215,983, dropout 0.404). One pooled big-6 GBT fit (1,000,000-row stratified sample) supplies
every number below; `src/07b_unseen_deployability.py` writes all four result CSVs from that
single fit. Figures: `paper/fig3b_calib_dca_unseen.png`, `paper/fig4b_covid_unseen.png`,
`paper/fig5b_fairness_unseen.png`.

**Discrimination and calibration.** AUROC 0.778, Brier 0.186 (trained-substance reference:
0.791 / 0.179). Across risk deciles the mean absolute gap between predicted and observed
dropout is **0.015** (0.0147), never exceeding **0.046** (0.0460, decile 8); the scores are
usable as probabilities, not only as a ranking, on classes never trained on.

> These two values changed when the canonical JSH assembly was adopted (from 0.013 and
> 0.026). Both trace to the smaller training pool: 1,000,000 drawn from the 4,550,870 JSH
> train side rather than from the full 6,501,243 big-6, and the degradation is real, not a
> rounding artifact. Discrimination is unaffected (AUROC reproduces exactly); calibration is
> the more sample-sensitive property. The conclusion is unchanged: "never exceeds 0.046"
> still supports usable-as-a-probability rather than merely as a ranking.

**Clinical utility.** Net benefit stays above both "treat all" and "treat none" through
threshold 0.80 (0.275 at 0.20, 0.126 at 0.50, 0.020 at 0.80); "treat all" turns negative at
0.41. The margin widens above threshold 0.15, which is the regime a capacity-limited program
actually operates in.

**COVID-era transportability** (train on 2015–2018 unseen-class episodes only, n=125,912):

> **Documented exception to JSH: intentional, kept by user decision, disclosed in the
> paper.** This one arm is a *within-unseen* replication: it trains a separate model on the
> unseen-class 2015–2018 episodes (n=125,912, **seed 42**) and tests on unseen-class
> episodes in 2019/2020/2022. It is not a JSH scoring pass and is not to be "fixed" into
> one. The question it asks (is the unseen cohort's *own* temporal decay steeper than the
> trained substances'?) requires a model fitted inside that cohort; folding it into JSH
> would replace the question rather than answer it. Every other arm in
> `src/07b_unseen_deployability.py` scores the shared JSH fit without refitting.

| Test era | AUROC | n | Dropout rate |
|---|---|---|---|
| 2019 (pre-COVID, held-out) | 0.789 | 35,424 | 0.446 |
| 2020 (COVID) | 0.738 | 26,861 | 0.417 |
| 2022 (post-COVID) | 0.708 | 27,786 | 0.394 |

Unlike the trained substances (0.017 then 0.001), the unseen classes lose **0.051** at the
pandemic boundary and a further **0.030** by 2022: combined substance and temporal shift costs
about 0.081 AUROC over seven years, whereas either alone costs under 0.020. Coverage does not
come with temporal stability for free.

**Subgroup fairness** (22 strata with n≥200, across race, sex, age):

- **Discrimination** spans 0.677–0.798 (sample sd 0.032); sex is near-equal (M 0.775 /
  F 0.780); age rises from 0.707 at 12–14 to 0.791 at 65+ (the highest AUROC in the set is
  Black, 0.798, not the 65+ band); the lowest values fall on the smallest strata
  (Alaska Native n=260, 0.677).
- **Calibration** holds in 21 of 22 strata (all within 0.044), but the **65+ band is
  over-predicted by 0.140** (n=5,061; observed dropout 0.165 against a cohort rate of 0.404).
  Restricting to the eight strata with n>20,000, calibration error never exceeds 0.016 and
  AUROC stays in 0.772–0.798.
- The 65+ gap is real in the data as distributed, but it is **not an age effect**: it is a
  localised reporting artifact. `src/07c_age65_anomaly.py` decomposes the band
  (`p1_age65_anomaly_diagnosis.csv`):

  | Stratum | n | pred | obs | gap |
  |---|---|---|---|---|
  | 65+ band (as published) | 5,061 | 0.305 | 0.165 | **+0.140** |
  | 65+, STFIPS 36 | 3,961 | 0.288 | 0.100 | +0.188 |
  | 65+, all other states | 1,100 | 0.366 | 0.399 | −0.033 |
  | 65+ × STFIPS 36 × Other_tranquilizers | 3,234 | 0.291 | 0.067 | **+0.225** |
  | **65+ band excluding that one cell** | 1,827 | 0.329 | 0.339 | **−0.010** |

  One state × substance cell (3,234 episodes, **1.5% of the unseen cohort**) is 64% of the
  whole 65+ band. Its *observed* dropout (0.067) is the outlier, not the prediction: big-6
  patients of the same age in the same state drop out at 0.277, and the model's 0.291 is what
  the cell's setting mix implies. The pattern was **not learnable** from training data,
  big-6 STFIPS 36 65+ dropout is 0.277 overall and far higher in ambulatory settings, where
  almost the whole cell sits.
- Excluding the cell, the 65+ band **under-predicts by 0.010** (−0.0104; it previously read
  +0.000 under the pre-JSH assembly) and **no stratum exceeds 0.05** (max 0.044 at 12–14).
  Headline numbers are insensitive to the exclusion.
- The likely mechanism (a program- or state-level discharge-coding convention) **cannot be
  confirmed** from the public-use file, which has no program identifier. The indicated
  deployment check is therefore per-state label screening, not age-stratified recalibration.

---

## 7. Foundation model benchmark: TabPFN-3 vs GBT

TabPFN-3 (v3 checkpoint, CPU) vs GBT at **matched 10,000-row training / 2,000-row test**
per comparison, bootstrap 300, leakage-free. 18 comparisons across single/pooled (big-6) plus
coverage. (`p1_tabpfn_vs_gbt.png`, `p1_tabpfn_vs_gbt.csv`)

- **TabPFN-3 wins every matched comparison (18/18), mean advantage +0.018 AUROC.**
  (`p1_tabpfn_vs_gbt.csv` has 36 *rows*: 18 comparisons × 2 model families. The manuscript
  correctly says 18; an earlier draft of this file said 36/36, which counted rows, not
  comparisons.)
  - Big-6 (12/12): single +0.020, pooled +0.017.
  - Coverage (6/6): GBT 0.729 → TabPFN 0.746 (+0.018).

### 7b. Second protocol: maximum training size for each model (GPU)

The matched-training-size protocol answers "which model is better per training row", not "which model is
better". With a GPU available (NVIDIA RTX 2000 Ada, 16 GB), the benchmark was rerun giving
each architecture the most data it can use: **TabPFN-3 at 100,000 training rows** (its
measured ceiling for a single scoring pass) vs **GBT at up to 1,000,000 rows**, on the same
18 comparisons and the *same held-out test rows*, with a 300-resample bootstrap on the paired
difference. (`p1_tabpfn_uncapped.csv`, `paper/fig6_tabpfn.png`: rendered as a supplementary
figure; the manuscript states these values in text to meet the 5-page limit)

- **The advantage disappears: 8 of 18 comparisons favour TabPFN-3, mean −0.0001 AUROC**, and
  **15 of 18 confidence intervals include zero**.
- Only 3 comparisons are significant, and in **both** directions: Other\_opioids pooled
  +0.0076 [0.0010, 0.0141] and Cocaine single +0.0075 [0.0005, 0.0135] favour TabPFN-3;
  Other\_hallucinogens coverage −0.0087 [−0.0161, −0.0010] favours GBT.
- Per-regime means collapse from the matched-training-size protocol to the maximum-training-size protocol:
  pooled +0.0174 → +0.0016, single +0.0199 → +0.0003, coverage +0.0175 → −0.0022.
- GPU speed: the identical 10k/2k comparison took **187.5 s on CPU vs 3.1 s on GPU (~60x)**; each
  100k-row comparison runs in ~39 s (fit 1.3--1.5 s, predict ~37 s) at 2.07 GiB peak VRAM.

**Interpretation.** TabPFN-3's edge is a **sample-efficiency effect, not a higher accuracy
ceiling**. Both architectures converge near 0.78 AUROC, which locates the performance limit
in the intake variables rather than in the model class. The two protocols together are the
result: TabPFN-3 reaches parity on a tenth of the episodes GBT uses, so it is the model of
choice when labelled data is scarce, while GBT remains the cheaper choice when the full
national file is available.

Note that the earlier statement "full-data GBT exceeds capped TabPFN" was refined by a
per-substance correction. The relevant comparison is per substance, not against the 0.785
cross-substance mean: on the paired Alcohol/single comparison of this benchmark (same 2,000
held-out test rows), GBT at 1,000,000 rows scores 0.7915 and TabPFN-3 at 100,000 rows
scores 0.7866, so TabPFN-3 is 0.0049 behind (95% CI [-0.0129, 0.0038], not significant)
rather than ahead. Evaluated instead on the full 747,356-row test set of the Phase-1
baselines, GBT's Alcohol single-substance AUROC is 0.7892; that number comes from a
differently-scaled evaluation and is not the paired value for this comparison.

**TabFlex** (ICML 2025 poster; linear-attention TabPFN variant) was evaluated as a
faster alternative and **deferred**: by its own paper it *matches* rather than beats
TabPFN on small data, its speedup is a GPU result, and it hardcodes single-threaded CPU
execution. Worth revisiting only with GPU access for full-scale (1M-row) in-context
learning.

---

## 8. Novelty

Literature search across PubMed, OpenAlex, **arXiv, and IEEE Xplore** (the last two a
title scan completed as the final gate). The **base task** (dropout prediction on TEDS)
is **not novel**: prior work exists (e.g. Addict Behav 2025 RF AUC 0.63–0.67; JAMIA
2021 XGBoost 0.89 with external crime-data join; PLoS One 2022 virtual-twins).

The three differentiators, after the full scan:
- **(a) Cross-substance transfer: NOT novel as a bare idea.** Whitworth et al.
  (*Substance Use & Misuse* 2022, PMID 36128946) already did CNN weight-transfer from a
  heroin dataset to an opioid dataset ("across related substances, large→small"). P1's
  defensible contribution is narrower: a **systematic leave-one-substance-out design
  across 6 diverse substances** used as an *evaluation protocol*, plus the **negative
  finding** that pooling does not improve accuracy (Whitworth framed transfer as an
  accuracy gain). Roberts et al. 2022 (COMBINE) is adjacent via leave-**site**-out.
  Cite Whitworth; do not claim transfer as the novel core.
- **(b) Single pooled model covering unseen/data-poor substance *classes*: novel**, no
  arXiv/IEEE/PubMed/OpenAlex precedent found. **This is the recommended headline.**
- **(c) TabPFN-3-vs-GBT benchmark on SUD treatment dropout: novel**, no precedent
  found (TabPFN clinical uses exist elsewhere, none on SUD).

Raw AUROC (~0.78) is on par with the strongest intake-only prior work, the contribution
is the **generalization structure** (coverage of unseen substances), not leaderboard
accuracy. See `docs/NOVELTY.md` for the full scan and citations.

---

## 9. Summary of deliverables

| Artifact | Content |
|---|---|
| `teds_d_analysis_2015_2022.parquet` | Harmonized 6.72M-episode analysis table (checkpoint) |
| `teds_d_data_dictionary.md` | Locked feature/label definitions, big-6/small split |
| `p1_phase1_single_pooled.csv` / `.png` | Baseline GBT+logistic, 3 regimes, CIs |
| `p1_calibration_dca.png` / `p1_dca_realdata.csv` | Calibration + real-data decision curve |
| `p1_coverage_ci.*` / `p1_coverage_temporal*.* ` | Unseen-substance coverage (+ 2022 temporal) |
| `p1_covid_temporal.*` | COVID-aware transportability (leakage-free) |
| `p1_fairness.png` / `p1_fairness_*.csv` | Subgroup discrimination + calibration |
| `p1_tabpfn_vs_gbt.*` | TabPFN-3 vs GBT head-to-head (18/18 matched comparisons) |
| `docs/NOVELTY.md` | Literature-grounded novelty case |

### Headline findings
1. **(Recommended lead)** A single pooled model **covers 12 unseen small substance
   classes** at mean AUROC 0.777 (range 0.745–0.806), with no substance-specific training, the deployment argument,
   measured. No literature precedent found (novel).
2. The model is **well-calibrated, fair in calibration across demographics**, and shows
   real-data net benefit at program-relevant thresholds.
3. It **transports across the COVID discontinuity** with a small, bounded, one-time
   decay.
4. **TabPFN-3 is the stronger low-data learner** (18/18 matched wins, +0.018); **GBT is
   the full-data deployment model**.
5. Pooling does **not** improve accuracy, the pan-substance value is generality and
   coverage, never an accuracy gain.

---

## 10. Optional future work: stronger transfer control (not run)

Coverage as used here is **zero-shot direct generalization** (instance-based naive transfer),
the weakest form of transfer. A reviewer may ask whether a *stronger* adaptation method
would improve cross-substance prediction. The near-certain answer is **no**: because
pooling already fails to beat substance-specific training (Δ = −0.008), there is no
covariate-shift gain left to exploit, but this can be converted from an assumed null
into a measured one with a single cheap control:

- **Importance-weighted GBT** (primary model). Estimate a source→target density ratio
  (train a classifier to separate target-substance rows from pooled-source rows; use its
  scores as `sample_weight`), refit the pooled GBT, compare to naive pooled in the
  coverage setup. This is a standard, citable transfer method
  (classifier-based importance weighting) and works directly through GBT's
  `sample_weight`. Note: GBT has **no** fine-tuning path: a fitted tree ensemble cannot
  be adapted by continued training, so instance reweighting is the correct transfer
  operator for the tree stack, not "fine-tuning."
- **TabPFN-3 few-shot in-context** (foundation-model analog of fine-tuning). Add a handful
  of target-substance rows to the context set and compare to pure zero-shot. No retraining
  needed.

Expected outcome for both: adaptation ≈ zero-shot (no gain), confirming the risk
structure is fully substance-general. One experiment each (~30 min); recommended only if
a reviewer presses the transfer-strength question. Not run in this study.
