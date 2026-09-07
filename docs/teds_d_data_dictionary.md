# TEDS-D Analysis Table: Data Dictionary
## File: teds_d_analysis_2015_2022.parquet

**Source:** SAMHSA Treatment Episode Data Set, Discharges (TEDS-D), public-use files.
Per-year source-file identity (URL, SHA-256, byte size, row count) is pinned in
`data/raw/MANIFEST.csv`; the seven raw files total 10,713,968 data rows before
outcome and substance filtering.
**Years:** 2015, 2016, 2017, 2018, 2019, 2020, 2022 (7 years).
**NOTE: 2021 EXCLUDED:** the 2021 public-use CSV ships every field as TEXT labels
(e.g. REASON="Dropped out of treatment", SUB1="Marijuana/hashish") rather than the
numeric codes used in all other years. Rather than hand-build a text→code crosswalk
(silent-miscoding risk), 2021 is dropped. 
**Analytic sample:** episodes with REASON in {1,2} and a mappable primary substance.
- Rows: 6,679,648 (17 substances; SUB1 code 19, other drugs, excluded)
- Overall dropout rate: 0.397

## Columns
| Column | Role | Definition |
|---|---|---|
| YEAR | metadata | Discharge year (2015–2020, 2022) |
| y | LABEL | 1 = dropout (REASON==2, "left against professional advice"); 0 = completed (REASON==1) |
| SUB1 | grouping | Primary substance numeric code (NOT a feature; defines the substance classes and the coverage split) |
| subname | grouping | Primary substance name (mapped from SUB1) |
| is_big6 | grouping | True for the 6 large trainable classes; False for the 11 small classes |
| SUB2, SUB3 | feature | Secondary / tertiary substance |
| ROUTE1 | feature | Route of administration, primary substance |
| FREQ1 | feature | Frequency of use, primary substance (at admission) |
| FRSTUSE1 | feature | Age at first use, primary substance |
| SERVICES | feature | Treatment service setting at admission (strongest predictor) |
| DSMCRIT | feature | DSM diagnosis criteria |
| NOPRIOR | feature | Number of prior treatment episodes |
| PSOURCE | feature | Referral source |
| AGE, GENDER, RACE, EDUC, EMPLOY | feature | Demographics at admission |
| STFIPS | feature | State (FIPS code) |

**Primary feature set (LOCKED): the 15 intake-time features above.** The tested
12-field expansion improved pooled AUROC by only +0.005 (mean per-substance +0.006)
and is NOT used.

## Leakage exclusions (hard: never used as features)
- REASON (the label), SUB1_D and all *_D discharge fields, LOS (length of stay),
  CASEID/DISYR (identifiers), SUB1 (grouping var), and the 18 substance FLG flags
  (ALCFLG…OTHERFLG, they re-encode SUB1/2/3 and would leak the coverage grouping).

## High-volume / lower-volume split (locked on the full 7-year pool)
The lowest-volume of the six high-volume substances (Cocaine, 375,967) is 4.9× the highest-volume of the eleven lower-volume substances (Benzodiazepines, 76,380). See teds_d_class_sizes_2015_2022.csv for all 17 classes.

## Missing data
TEDS missing sentinel (-9) mapped to NaN in all features; HistGBM handles NaN natively.
