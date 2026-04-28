# OpenReview Submission Fields — N134 / Thesis B

**Target venue:** Transactions on Machine Learning Research (TMLR)
**Submission portal:** https://openreview.net/group?id=TMLR
**Anonymization status:** double-blind — `\usepackage{tmlr}` (no `[accepted]` option) auto-anonymizes title/author block at compile time
**Tarball:** `tmlr_main_submission_v2.tar.gz` (117 KB, verified clean four-pass extract)
**Compiled PDF:** `draft_v2_thesis_b.pdf` (17pp, 455 KB)
**Supplementary tarball:** `supplementary_bundle.tar.gz` (separate upload, 908 KB)

---

## Title

Measurement Discipline for ML Diagnostics: A Psychometric Framework with a LoRA-Merging Case Study

---

## Abstract

(Lifted verbatim from `draft_v2_thesis_b.tex` lines 72–103. Paste directly into the OpenReview abstract field; OpenReview's plain-text field strips LaTeX, so use the rendered form below.)

> ML diagnostic metrics — scores claimed to predict model behavior, merging outcomes, training dynamics, or capability — are routinely reported as point estimates to three or four decimal places without the measurement-theoretic infrastructure that psychological assessment has treated as baseline methodology for decades. We argue that this practice systematically overclaims precision and generality, and we introduce a framework for applying classical measurement theory (reliability coefficients, standard error of measurement, confound decomposition, pre-registered decision rules with explicit tolerance schedules) to ML diagnostic reporting.
>
> We demonstrate the framework through a pre-registered decoder-scale study of spectral LoRA-merging diagnostics (N = 45 cross-task adapter pairs on Mistral-7B-v0.3). Under the framework's discipline, the case study does three things: it produces a clean pre-registered null on per-pair merge prediction under family-confound control; it exposes that the headline statistic's floating-point precision at this sample size is of order 10⁻², comparable to its sampling variability, so ordinary third-decimal reporting is spurious; and it bounds the diagnostic's reliability with an explicit regime-scope caveat. The case study illustrates the paper's central claim: measurement discipline applied to ML diagnostics changes what can responsibly be claimed — not by decorating point estimates with error bars, but by forcing refusals the default reporting view would not.

---

## Keywords

OpenReview typically asks for 3–5 keywords. Candidates ordered by load-bearing relevance to the manuscript:

1. **measurement validity**
2. **reliability and reproducibility**
3. **construct validity**
4. **LoRA merging**
5. **psychometric methods for ML**

If 3 keywords are required, drop #4 and #5 — the methodological keywords are the load-bearing ones; LoRA merging is the substrate of the case study, not the contribution.

---

## TL;DR (one-sentence summary)

(Some venues ask for this; TMLR may. If asked.)

> A four-component measurement-discipline framework — reliability, validity, tolerance, confound decomposition — applied to ML diagnostic reporting, demonstrated via a pre-registered LoRA-merging case study that produces a null on per-pair merge prediction and surfaces an underreported floating-point precision limitation in rank-based correlations on small-N residuals.

---

## Subject area / primary research area

TMLR uses general subject areas. Primary fits:
- **Methodology / measurement and evaluation**

If TMLR requires a more specific category:
- *General Machine Learning* — fits the methodological-framework register
- *Probabilistic Methods* — applicable to the reliability/variance content
- *Other → Measurement and Evaluation* — most precise if available

---

## Author list

For double-blind review:
- Author 1: Anonymous Author
- Affiliation: Anonymous Institution

The OpenReview UI typically auto-handles this if you submit logged in as an author with anonymization enabled. Verify: the rendered PDF's title page should show **anonymized title block** (the `\usepackage{tmlr}` without `[accepted]` already produces this — the recent compile passes confirm).

---

## Conflict-of-interest declarations

Standard TMLR fields. Declare any of:
- Anthropic / OpenAI / DeepMind / Meta affiliations
- Recent co-authorships with potential reviewers
- Advisor/student relationships within the past 5 years

(User-side: fill from your own situation.)

---

## Action editor preference (optional)

TMLR matches submissions to action editors based on declared expertise. The right action editor for this paper is someone literate in:
- Measurement theory or psychometrics applied to ML
- Reliability / reproducibility / evaluation methodology
- Construct validity in ML benchmarks

If TMLR's action-editor list includes anyone with NeurIPS-D&B-track work, eval-methodology papers, or psychometric-ML cross-pollination publications, list them. The right action editor measurably improves review quality.

(User-side: scan the TMLR action-editor list when filling the form; this field is optional but worth taking 5 minutes on.)

---

## Code and data availability

TMLR asks for explicit code/data statements.

**Code availability:** Analysis scripts that produce the headline numbers are included in the supplementary materials accompanying this submission (three scripts: H1 main, ICC, four-method comparison). At camera-ready, full repository will be released under the project's standard open-source licence.

**Data availability:** Pre-registration documents, the four top-level analytical JSONs (H1, ICC, secondary, four-method comparison), and the 24 per-adapter raw audits with their meta-JSONs are in the supplementary materials. Per-adapter SVD factor files (~1.2 GB pod-only) are not in the supplementary; this is documented in Appendix E (Tier 4 gap documentation).

**Reproducibility statement:** The committed-state reproducibility check (Appendix E) verifies all six qualitative claims and all quantitative scalars within the amended tolerance schedule, with one value (partial Spearman ρ) at the rank-on-residuals precision regime and explicitly documented.

---

## Supplementary materials upload

`supplementary_bundle.tar.gz` (908 KB) — separate upload from the main-manuscript tarball. Contains:
- Pre-registration documents (main spec + ICC analysis supplement)
- Four top-level analytical JSONs (H1, ICC, secondary, four-method comparison)
- 24 per-adapter raw audits + four meta-JSONs (pair alignments, adapter profiles, base-model reference)
- Three analysis scripts that produce the headline numbers
- Supplementary `README.md` and `ANON_AUDIT_CHECKLIST.md`

Do **not** upload the internal-development files (`internal_memo.md`, `internal_summary.md`, `pre_submission_edit_spec*.md`, `revision_notes.md`, `CHANGELOG.md`, `BUILD.md`, draft outline files). Those are program-internal and not for review.

---

## Final pre-upload checklist

Before clicking submit:

- [ ] PDF is the post-rewrite compile (17pp, post-pre-tarball-pass App E/F/G fixes)
- [ ] Tarball is `tmlr_main_submission_v2.tar.gz` (the new one, not the Apr-23 stale `tmlr_main_submission.tar.gz`)
- [ ] Supplementary is `supplementary_bundle.tar.gz` (Apr-23, still current — no changes since)
- [ ] Title block on PDF shows anonymized author/affiliation (`\usepackage{tmlr}` without `[accepted]` confirms)
- [ ] PDF metadata clean (already verified: empty `kMDItemAuthors`, generic `kMDItemCreator`)
- [ ] No `% ANON:` markers visible in rendered PDF (markers are LaTeX comments, won't render — but worth a final glance)
- [ ] Abstract field on OpenReview matches the rendered abstract on the PDF (verbatim match)
- [ ] Keywords match the load-bearing register, not the substrate

---

## Post-submission

Once submission lands:
1. Note the OpenReview submission ID for the task list
2. TMLR enters rolling review (~3-month typical cycle to first decision)
3. Tag the repo state at `v2-submitted` (or similar) so the submitted commit is recoverable
4. Update `RESEARCH_INVENTORY.md` Section 7 trajectory with the submission date
5. The benchmark-reliability paper's "Relationship to N134" section now has a stable target

---

## Contingencies

**If OpenReview's PDF upload fails:** the source-bundle tarball is the fallback. TMLR accepts either single-PDF or full-source. Try PDF first; tarball if needed.

**If a missing-package error appears post-upload:** TMLR's compile environment may differ from local MacTeX. The `tmlr.sty` and `fancyhdr.sty` are bundled; standard packages (microtype, hyperref, natbib, amsmath, graphicx, booktabs) should be in TMLR's TeX Live install.

**If keyword limit is 3 not 5:** drop "LoRA merging" and "psychometric methods for ML" — keep the three load-bearing methodological keywords.

**If the form asks about overlap with other submissions:** the benchmark-reliability paper is the closest neighbor. State explicitly: "A separate paper (in preparation) applies the same four-component framework to LLM benchmark evaluation as a second worked demonstration; that paper has not been submitted to any venue at the time of this submission."
