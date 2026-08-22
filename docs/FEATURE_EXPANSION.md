# Feature-Set Sensitivity: Does Expansion Improve Prediction?

## Decision: use the parsimonious 15-feature set as primary.

We tested whether expanding the intake-time feature panel beyond the 15 locked
features improves dropout prediction. The candidate expansion added 12 further
pre-admission fields (all validated as admission-record values, not discharge
_D twins): DIVISION, REGION, ALCDRUG, ETHNIC, METHUSE, IDU, PSYPROB, VET,
LIVARAG, ARRESTS, FREQ_ATND_SELF_HELP, MARSTAT: taking the panel from 15 to 27.

**Result (pooled GBT, 2020 held-out test, single stratified split):**
- Pooled AUROC: 0.820 (15 features) -> 0.825 (27 features), delta = +0.005
- Mean per-substance AUROC delta = +0.006 (all six substances positive,
  range +0.004 to +0.009)

**Interpretation.** The expansion produces a *consistent but small* improvement:
every substance improves, so the added fields carry real (non-noise) signal, but
the magnitude (~0.005-0.006 AUROC) is a fraction of a percentage point and does
not change the model's character or any downstream conclusion. It is smaller
than the pooled-vs-substance-specific gap (~0.008) and
comparable to the bootstrap CI half-width.

**Why 15 features is kept as primary:**
1. Parsimony: 15 fields are easier to collect, document, and deploy.
2. Lower missingness: several expansion fields sit at 12-23% missing.
3. Avoids METHUSE, a planned-MAT treatment-plan variable whose pre-admission
   status is defensible but borderline.
4. The gain is within CI width: not a meaningful accuracy difference.

This is reported as a **marginal/negative sensitivity result**: expanding the
feature set does not meaningfully improve prediction, so the simpler model is
preferred. No expanded model is trained in the main results.
