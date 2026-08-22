# TEDS-D Cohort Characteristics (2015-2022, 2021 excluded)

Analytic sample: **6,717,226** discharge episodes with REASON in {1,2} and a mappable primary substance.
Overall dropout rate **0.3976**. Percentages are column percentages of the full cohort;
`Dropout` is the within-group rate of leaving against professional advice.

Label maps for substance, year, race, sex, and age band are the ones used in the committed
analysis code. The remaining TEDS categorical fields are reported by raw numeric code because
no SAMHSA codebook is bundled with the public-use CSVs in this repository; see `p1_cohort_table.csv`.

**Overall**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| All episodes | 6,717,226 | 100.0 | 0.398 |

**Substance group**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| Big-6 (trainable) | 6,501,243 | 96.8 | 0.397 |
| Small (held out) | 215,983 | 3.2 | 0.404 |

**Primary substance**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| Alcohol | 2,491,184 | 37.1 | 0.295 |
| Heroin | 1,560,370 | 23.2 | 0.473 |
| Cannabis | 840,885 | 12.5 | 0.456 |
| Methamphetamine | 776,204 | 11.6 | 0.444 |
| Other opioids | 456,633 | 6.8 | 0.473 |
| Cocaine | 375,967 | 5.6 | 0.442 |
| Benzodiazepines | 76,380 | 1.1 | 0.335 |
| Other amphetamines | 39,015 | 0.6 | 0.421 |
| OTC | 37,578 | 0.6 | 0.531 |
| PCP | 18,098 | 0.3 | 0.401 |
| Non rx methadone | 9,867 | 0.1 | 0.485 |
| Other hallucinogens | 8,096 | 0.1 | 0.399 |
| Barbiturates | 7,695 | 0.1 | 0.430 |
| Other stimulants | 7,624 | 0.1 | 0.402 |
| Other tranquilizers | 5,187 | 0.1 | 0.170 |
| Other sedatives hypnotics | 3,214 | 0.1 | 0.399 |
| Inhalants | 2,607 | 0.0 | 0.427 |
| Other sedatives | 622 | 0.0 | 0.476 |

**Discharge year**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| 2015 | 986,614 | 14.7 | 0.383 |
| 2016 | 1,015,960 | 15.1 | 0.388 |
| 2017 | 1,080,620 | 16.1 | 0.383 |
| 2018 | 1,034,079 | 15.4 | 0.387 |
| 2019 | 1,042,783 | 15.5 | 0.406 |
| 2020 | 782,004 | 11.6 | 0.426 |
| 2022 | 775,166 | 11.5 | 0.425 |

**Sex**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| Male | 4,492,019 | 66.9 | 0.378 |
| Female | 2,222,621 | 33.1 | 0.437 |
| Missing | 2,586 | 0.0 | 0.527 |

**Age band at admission**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| 12-14 | 35,002 | 0.5 | 0.426 |
| 15-17 | 182,273 | 2.7 | 0.432 |
| 18-20 | 215,916 | 3.2 | 0.463 |
| 21-24 | 588,001 | 8.8 | 0.434 |
| 25-29 | 1,119,941 | 16.7 | 0.427 |
| 30-34 | 1,108,153 | 16.5 | 0.417 |
| 35-39 | 903,291 | 13.4 | 0.407 |
| 40-44 | 662,231 | 9.9 | 0.392 |
| 45-49 | 606,764 | 9.0 | 0.370 |
| 50-54 | 577,785 | 8.6 | 0.347 |
| 55-64 | 625,069 | 9.3 | 0.323 |
| 65+ | 92,800 | 1.4 | 0.290 |

**Race/ethnicity**

| Group | n | % | Dropout |
|---|---:|---:|---:|
| Alaska Native | 16,029 | 0.2 | 0.288 |
| American Indian | 162,380 | 2.4 | 0.319 |
| Asian/Pacific Islander | 2,487 | 0.0 | 0.349 |
| Black/African American | 1,190,339 | 17.7 | 0.425 |
| White | 4,374,536 | 65.1 | 0.390 |
| Asian | 47,345 | 0.7 | 0.385 |
| Other single race | 602,916 | 9.0 | 0.448 |
| Two or more races | 147,044 | 2.2 | 0.459 |
| Native Hawaiian/Pacific Islander | 32,251 | 0.5 | 0.394 |
| Missing | 141,899 | 2.1 | 0.222 |

**Other intake fields (raw TEDS codes)**

See `p1_cohort_table.csv` rows for SERVICES, PSOURCE, EDUC, EMPLOY, ROUTE1, FREQ1, FRSTUSE1, NOPRIOR, DSMCRIT.
