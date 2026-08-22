# UGHS 2026 Submission Compliance

Checked against the instructions at https://bibm2026-hs.github.io/ (retrieved 2026-08-16).

**Every measurement in this file was taken fresh from the compiled `paper/paper_v1.pdf`
during the paper_v1 verification pass: nothing here is carried over or assumed.** The
measuring commands are given at the end so any row can be re-checked in one step.

## Measured properties of `paper/paper_v1.pdf`

| Property | Measured value | How measured |
|---|---|---|
| Pages | **5** | `pdfinfo` |
| Page size | **612 x 792 pt** (US Letter), all 5 pages | `pdfinfo` |
| Producer | xdvipdfmx (0.1): tectonic | `pdfinfo` |
| Embedded font descriptors | **10** | `pdffonts` |
| Font types present | 5 x CID Type 0C, 5 x Type 1C | `pdffonts` |
| **Type 3 fonts** | **0** | `pdffonts` |
| All fonts embedded | yes (every row `emb=yes`) | `pdffonts` |
| Abstract length | **209 words** | token count on the `abstract` environment, macros stripped |
| Figures in the manuscript | **4** (`fig1_coverage`, `fig3b_calib_dca_unseen`, `fig4b_covid_unseen`, `fig5b_fairness_unseen`) | `\includegraphics` scan of `paper_v1.tex` |
| Tables | **2** (Table I predictors, Table II cohort) | `\begin{table}` scan |
| References | **15** `\bibitem`, all 15 cited, no duplicates, no dangling `\cite` | label/cite set comparison |
| Unresolved references | **0** | recompiled log, no "undefined references" |
| Labels | 6 defined, 6 referenced, **0 unused, 0 dangling** | `\label` vs `\ref` set comparison |
| Missing image files | **0** | every `\includegraphics` target exists in `paper/` |
| Final-column slack | **31.4 pt** = **2.63 body lines** (11.95 pt pitch) or 3.50 reference lines (8.97 pt pitch), measured by trailing-\vspace* insertion | see below |

### How the slack was measured

Word bounding boxes were extracted with `pdftotext -bbox`. The text-block bottom is
`yMax = 718.70 pt`, established from pages 3 and 4, where both columns run full-length and
agree exactly. On page 5 the right (final) column's last baseline, the closing line of
reference [15]: sits at `yMax = 693.01 pt`. The difference is **25.69 pt**. Body-line
pitch on that page measures 11.95 pt and reference-line pitch 8.97 pt, giving 2.15 and 2.86
lines respectively.

> **Correction to the previous version of this file.** It reported 29.3 pt / ~2.5 lines
> (and a working note elsewhere said 31.4 pt / 2.62 lines). Neither reproduces. The
> re-measured figure is **25.69 pt / 2.15 body lines**. The margin is real but roughly two
> body lines, not two and a half, so there is less room than previously recorded before the
> paper spills to a sixth page.

## Formatting requirements

| Requirement (as stated on the site) | Status |
|---|---|
| Follow IEEE Computer Society Proceedings Manuscript Formatting Guidelines | Met. `\documentclass[conference]{IEEEtran}`, IEEEtran.cls V1.8b, two-column |
| High school papers: max 5 pages incl. figures, tables, references | **Met, 5 pages measured**, with 31.4 pt (2.63 body lines) of slack in the final column |
| Undergraduate papers: max 6 pages | Met with a page to spare in the undergraduate track |
| Indicate in the author affiliation whether the first author is high school or undergraduate | Addressed implicitly. Per author decision the explicit designation line was removed; the affiliation names the high school, which carries it |
| (not stated on the site; from the IEEE template) US Letter page size | **Verified: all 5 pages are 612 x 792 pt** |
| High-clarity figures | All four manuscript figures are placed at `\columnwidth` (3.5 in). Fig. 1 is rendered at 3.5 in, 400 dpi (`src/09_paper_figures.py`); Figs. 2-4 are rendered on a 7.0 in canvas at 300 dpi (`src/09b_paper_figures_unseen.py`) and so are downscaled about 2x on the page, which raises effective dpi and reduces effective font size. Two further figures (`fig2_regimes.png`, `fig6_tabpfn.png`) are held as supplementary and their content stated numerically |
| PDF eXpress validation (Conference Record TBD) | Pre-checked: **10 embedded font descriptors, all Type 1C / CID Type 0C subsets, zero Type 3 fonts.** Final validation must still be run on the CPS site once the Conference Record ID is published |

## Structure present in the manuscript

Title, author block, Abstract (**209 words**), Index Terms, Introduction, Related Work,
Data and Methods, Results, Discussion, Conclusion, Reproducibility and Data Availability,
15 references. No page numbers, per IEEEtran conference style.

## Reference integrity

All 15 published reference titles were checked word-for-word against Crossref during this
pass. IEEE renders article titles in sentence case, so capitalisation differences from the
published form are correct and were not counted as errors; only word-level differences were.
**Two word-level discrepancies were found and are listed as open items below.** Reference
[15] (`importance_weight`, arXiv:2112.10157) is a preprint with no journal record and was
checked against the arXiv identifier only.

## Review model

The site states the review process is **single-blind** (reviewers anonymous, authors
visible). No anonymization is required, so author names and affiliations stay in the
submitted PDF. Do not strip them.

## Open items the authors must complete before submitting

1. ~~**Two reference titles differ from the published form at the word level.**~~
   RESOLVED: both titles now read in the published form in `paper_v1.tex`.
   - `obermeyer_bias`: "Dissecting racial bias in an algorithm used to manage the health of
     populations" (Science 366:447-453, doi 10.1126/science.aax2342).
   - `sud_ml_review`: "...machine learning algorithms on substance-use disorders treatment
     outcomes" (Int J Ment Health Addict 24(2):1090-1117, doi 10.1007/s11469-024-01403-z).
2. **Author block placeholder** in `paper_v1.tex` (line 31): replace `<author email>` with the
   first author's contact address. The affiliation is already filled in (White Station High
   School, Memphis, Tennessee, USA) and no designation line remains to delete.
3. **Eligibility**: the first author must be an undergraduate or high school student at the
   time of submission, and at least one student author must attend in person or the paper
   will not appear in the IEEE proceedings.
4. **High school authors only**: email the signed parent/guardian information document
   (paper ID, title, parent contact) to the address given on the symposium site.
5. **Repository metadata**: `LICENSE` and `CITATION.cff` still carry `[Author Name(s)]`,
   `[Last]`, `[First]` placeholders.
6. **Originality**: the site prohibits dual submission. This manuscript must not be under
   review elsewhere.
7. **Deadlines** as posted: paper submission August 31, 2026, 11:59 PM AoE. Submission via
   the IEEE CPS site; use the conference submission ID (not the PDF eXpress ID).
8. **In-person attendance**: BIBM 2026 is in Dallas, TX, and at least one author must
   register and present in person.

## Caveat on the page count

Overleaf compiles with pdflatex; only tectonic is available in this environment. The 5-page
result is established under tectonic plus the measured 2.63-line margin in the final column,
not by reproducing pdflatex output. The rebuilt `overleaf_submission.zip` was extracted to a
clean directory and compiled standalone during this pass to confirm it still produces 5
pages with 0 unresolved references and 15 listed references.

If the Overleaf build spills to 6 pages, trimming Discussion prose is the safest lever, as
it touches no reported number; reducing the height of one of the three double-column figures
is next-safest.

Reaching 5 pages from 6 cost two figures, not any claim. `fig2_regimes.png` and
`fig6_tabpfn.png` were removed from the manuscript and their content restated numerically in
the Results text. Both remain in the repository under `paper/` as supplementary material.

## Provenance of the requirements quoted above

Every requirement was matched verbatim against the symposium page as fetched on 2026-08-16,
committed alongside this file as `docs/ughs_site_requirements_2026-08-16.txt`. Line numbers:
format guidelines 177, UG 6 pages 178, HS 5 pages 179, affiliation designation 180,
single-blind 195, high-clarity figures 200, PDF eXpress 202, submission ID 203, deadline 186,
in-person Dallas 191, attend and present 169, first-author eligibility 165, parent document
173, dual submission 182.

Note that "US Letter" does **not** appear anywhere on the symposium page (verified by string
search); that row is attributed to the IEEE template, not to the site.

## Re-running these measurements

```bash
pdfinfo   paper/paper_v1.pdf | egrep -i 'Pages|Page size'
pdffonts  paper/paper_v1.pdf                 # expect 10 rows, zero "Type 3"
pdftotext -bbox -f 5 -l 5 paper/paper_v1.pdf - | grep '<word' | tail -3   # final baseline
```
