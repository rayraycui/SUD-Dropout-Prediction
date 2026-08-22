# P1 Novelty Assessment: Literature Search (PubMed, gated pre-run check)

> **Status note (added after the novelty assessment was written).** The
> leave-one-substance-out (LOSO) experiment discussed throughout this document was
> subsequently **removed from the manuscript and the reproduction pipeline**. The paper now
> rests on single-model zero-shot coverage of unseen substance classes (ground (b)) and the
> TabPFN-3 vs GBT benchmark (ground (c)); the substance-generality claim that LOSO supported
> is no longer made. This file is retained unedited below as the dated record of the
> literature search as it was conducted, including the LOSO material, because rewriting it
> would misrepresent what was searched and found at the time. Read every LOSO passage below
> as historical context, not as a description of the current study.


**Method.** PubMed via connector; ~15 targeted queries across the three novelty
pillars (TEDS dropout prediction; cross-substance transfer; single-model coverage /
tabular foundation models). ~73 unique articles pulled and abstract-screened. DOIs
below are from PubMed metadata (verified identifiers, not recalled).

## What ALREADY EXISTS (so is NOT novel)

**Predicting SUD treatment completion / dropout from administrative data with ML,
well established, including on TEDS:**
- *Use of machine learning to examine disparities in completion of SUD treatment*,
  PLoS One 2022, **TEDS-D 2017–2019**, virtual-twins (RF + decision tree).
  DOI: 10.1371/journal.pone.0275054. (Disparities/subgroups, not a transfer test.)
- *Machine learning-based outcome prediction … for SUD treatment*: JAMIA 2021,
  XGBoost predicting completion, **AUC 89.3%**, patient-episode discharge data +
  crime data, nested CV. DOI: 10.1093/jamia/ocaa350. (Interaction-effect discovery.)
- *Predictors of treatment attrition … A machine learning approach*: Addict Behav
  2025, RF on **N=29,809**, predicts AMA / involuntary / early discharge,
  **AUC 0.63–0.67, low PPV 0.09–0.34**. DOI: 10.1016/j.addbeh.2025.108265.
  (Closest to P1's core task, but single-cohort, no cross-substance transfer.)
- Numerous MOUD/OUD-specific retention models (e.g. veterans B-MOUD, J Addict Dis
  2024, DOI 10.1080/10550887.2024.2363035; multiple OUD dropout RF/GBM papers).

**Takeaway:** "predict dropout from intake features" by itself is a crowded space.
A paper claiming only that would not be novel.

## What was NOT FOUND (the differentiators that make P1 novel)

1. **Cross-substance / leave-one-substance-out (LOSO) transfer test.**
   Query "transfer learning across substance types treatment outcome prediction"
   → **0 hits**; "cross-substance transfer learning treatment outcome" → **0 hits**.
   No paper trains on some substances and tests dropout prediction on a *held-out*
   substance. This is P1's scientific-generality claim and appears unoccupied.

2. **Single pooled model covering unseen / data-poor substances (coverage test).**
   No retrieved paper demonstrates one model producing valid dropout predictions for
   small substance classes it was not trained on. P1's §3.1 coverage result
   (big-6 → 12 held-out small classes, AUROC 0.745–0.806) has no direct precedent found.

3. **Tabular foundation model (TabPFN) for SUD treatment outcomes.**
   "TabPFN substance use disorder" → **0 hits**. TabPFN is being applied in clinical
   prediction broadly (11 hits: AKI, sepsis, TIPS-HE, EHR adaptation), but not to
   substance-use treatment retention/dropout. A TabPFN-vs-GBT benchmark here is new.

## Adjacent work that sharpens (not blocks) the novelty
- *Generalizability of Treatment Outcome Prediction Across Antidepressant Trials*,
  JAMA Netw Open 2025, DOI 10.1001/jamanetworkopen.2025.1310: cross-**trial**
  transfer, but in depression and across trials, not across **substances**. Confirms
  "transportability of outcome models" is a live question; P1 answers it in a
  different, unaddressed axis (substance).
- *Umbrella review of predictors across mental disorders*: Mol Psychiatry 2023,
  DOI 10.1038/s41380-023-02298-3 (117 reviews, 403 studies, 299,888 individuals),
  transdiagnostic *predictors*, not a transdiagnostic *transfer/coverage model*.

## Verdict
- **Base task (dropout prediction on TEDS): NOT novel**, do not frame the paper this way.
- **P1 as framed IS novel** on three grounds with no direct precedent found:
  (a) LOSO cross-substance transfer as a generality test,
  (b) single-model coverage of unseen/data-poor substances (the deployment argument),
  (c) TabPFN-vs-GBT benchmark for this task.
- **Framing rule (unchanged):** lead with transfer + coverage + foundation-model
  benchmark; never claim "pooling improves accuracy" (data refutes it) and never
  claim a novel base task.

*Caveat: PubMed only. IEEE/arXiv/ACM (CS venues) not searched here, a preprint
doing substance-transfer could exist there. Recommend an arXiv/IEEE check before
submission, but nothing in PubMed blocks the project.*


---
## OpenAlex cross-check (covers CS/IEEE/arXiv venues PubMed misses)
Ran the same three pillars on OpenAlex. Matching is loose full-text (high api_totals
but topically scattered top hits: itself a signal there is no tight cluster on these
exact ideas):
- **Cross-substance / LOSO transfer:** no work found training on some substances and
  testing dropout on a held-out substance (top hits were neuroscience/education
  dropout reviews, not this design).
- **TabPFN + SUD treatment:** only 4 total hits, none on treatment retention/dropout.
- **TEDS + ML dropout:** one additional on-target item surfaced, *"Use of a machine
  learning framework to predict substance use disorder treatment success"* (2017),
  a single-cohort success predictor, again NOT a cross-substance transfer/coverage design.

**Net:** OpenAlex corroborates PubMed. The base task has precedents; the three
differentiators (LOSO transfer, single-model coverage of unseen substances,
TabPFN-vs-GBT benchmark) have no direct precedent found in either index.

**Residual caveat:** neither index is exhaustive for very recent workshop/preprint CS
papers; a focused arXiv + IEEE Xplore title scan is advisable right before submission.
No result found here blocks proceeding with the study as framed.


---
## arXiv + IEEE Xplore title scan (final gate, completed)

**Method.** arXiv API (structured title/abstract queries across the three pillars +
base task) and targeted IEEE Xplore / web searches. This closes the "CS-venue"
caveat left open above. **It changed the picture for pillar (a) and must be recorded.**

### MATERIAL FINDING: cross-substance transfer is PARTIALLY preempted
- **Bailey and DeFulio, *Predicting Substance Use Treatment Failure with Transfer
  Learning*, Substance Use & Misuse 2022, PMID 36128946 / DOI
  10.1080/10826084.2022.2125272.** Trains a **CNN on a heroin treatment dataset,
  then transfers to a smaller opioid dataset**, explicitly testing "if model weights
  transfer **across related substances** and from large to small datasets." Transfer
  model beat a tuned RF and a no-transfer baseline.
  - **This is a genuine cross-substance transfer precedent.** The bare claim
    "cross-substance transfer has never been done" is therefore **FALSE** and must not
    be made.
  - **How P1 still differs (defensible, narrower claims):** (i) it is *one* source→target
    pair, both opioids (closely related), vs P1's **systematic leave-one-substance-out
    across 6 diverse substances** (alcohol, cannabis, cocaine, meth, heroin, other
    opioids); (ii) it is a *modeling* transfer method (CNN weight-transfer/fine-tuning)
    framed as an **accuracy gain**, whereas P1 uses LOSO as an **evaluation protocol**
    and reports the **opposite** headline: pooling does *not* improve accuracy; the
    value is generality + zero-shot coverage; (iii) it does not test a single pooled
    model on entirely unseen substance *classes*.
- **Roberts et al. 2022 (COMBINE trial), via the SUD-ML systematic review
  (Int J Ment Health Addict 2024, DOI 10.1007/s11469-024-01403-z):** uses
  **leave-site-out (LSO)** validation, an adjacent grouped-generalization protocol,
  but the grouping variable is *institution*, not *substance*.
- **IEEE Xplore:** the closest hit is a **state-specific multi-task model for MOUD
  dropout** (shared + state-specific layers, SHAP/LIME), but opioid-only and
  multi-task across *states*, not transfer across *substances*. Does not preempt LOSO.

### CONFIRMED UNPREEMPTED
- **(b) Single pooled model covering unseen/data-poor substance classes**: no arXiv or
  IEEE precedent found. Still novel.
- **(c) TabPFN-vs-GBT on SUD treatment dropout / TEDS**: no precedent. TabPFN clinical
  applications exist (stroke sICH AUC 0.948; causal-inference simulation; mycotoxin;
  RNA-seq; soil mapping) but **none on SUD treatment**. Still novel. A bioRxiv note
  (*Limitations of TabPFN for High-Dimensional RNA-seq*, 2025) independently confirms
  P1's honest caveat: restricted to its ≤500-feature/≤10k-sample envelope, TabPFN's
  advantage over classical baselines using full data disappears.

### REVISED VERDICT (supersedes the pre-run verdict above)
- **Base task: NOT novel** (unchanged).
- **(a) Cross-substance transfer: NOT novel as a bare idea**, Bailey and DeFulio 2022 did
  heroin→opioid transfer. P1's defensible contribution is the **systematic 6-substance
  LOSO design + the negative "pooling doesn't improve accuracy" finding**, not the
  existence of cross-substance transfer. Reframe accordingly and cite Bailey and DeFulio 2022.
  **Terminology (important):** LOSO here is **zero-shot direct generalization**
  (instance-based naive transfer / an evaluation protocol), *not* a transfer-learning
  method: do not describe it as "transfer learning." A stronger transfer control
  (importance-weighted GBT; TabPFN few-shot in-context) is noted as optional future work,
  expected null because pooled ≈ single (Δ −0.008).
- **(b) single-model coverage of unseen substances: novel** (no precedent found).
- **(c) TabPFN-vs-GBT benchmark for this task: novel** (no precedent found).
- **Net for a UGHS/workshop submission:** the project remains a legitimate, publishable
  contribution, but the **headline must shift** from "cross-substance transfer" (partly
  preempted) to **single-model coverage of unseen substances (b)** as the primary
  novelty, with LOSO reframed as a *systematic-generality control that extends*
  Bailey and DeFulio 2022, and the TabPFN benchmark (c) as a secondary contribution. Do not
  claim (a) as the novel core.
