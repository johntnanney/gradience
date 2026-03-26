# Case Study Publication Summary

## Title

**One Inventory, End to End**  
*From adapter pool to merge neighborhoods with a Gradience preflight pass*

---

## Purpose

This note summarizes one complete Gradience case-study run on a small real adapter inventory.

Its purpose is to support:
- a Hugging Face walkthrough post
- a README advanced example
- a paper appendix / supplement example
- collaborator or practitioner onboarding

This is **not** a benchmark report. It is a worked example designed to show how the Gradience workflow changes a real decision.

---

## Inventory Overview

### Inventory ID
`case_study_qnli4_realmix_20260318`

### Base model
`distilbert-base-uncased`

### Inventory size
`4` adapters

### Practical question
This inventory was selected to answer a practical question of the form:

> Given several plausible adapters, which ones are worth preserving, which pairs are worth considering, and whether the inventory has any local merge structure at all before downstream evaluation begins.

### Adapter pool

- **Adapter A** — `final_uniform_median_r16`  
  Intended role: strong anchor adapter

- **Adapter B** — `qnli_probe_elig`  
  Intended role: related secondary adapter

- **Adapter C** — `qnli_uniform_weak`  
  Intended role: weak or exclusion-worthy adapter

- **Adapter D** — `qnli_per_layer_elig`  
  Intended role: ambiguous / structurally non-obvious adapter

### Why this inventory was chosen
This inventory was chosen because it contains:
- one strong anchor source
- one clearly weak source likely to be excluded
- one pair that looked benign under ordinary pair-risk but still warranted deeper inspection
- a nontrivial neighborhood outcome (one 3-adapter group plus one exclusion)

---

## Step 1 — Single-Adapter QA

### Summary
The first pass was single-adapter QA. This immediately narrowed the candidate set.

### Adapter QA outcomes

- **Adapter A (`final_uniform_median_r16`)** — `eligible`  
  Confidence: `high`  
  Reason: outperforms base on QNLI accuracy; structural flags show low utilization/high rank waste.

- **Adapter B (`qnli_probe_elig`)** — `eligible`  
  Confidence: `high`  
  Reason: outperforms base on QNLI dev accuracy; structurally similar utilization profile to A.

- **Adapter C (`qnli_uniform_weak`)** — `flagged_weak`  
  Confidence: `high`  
  Reason: underperforms base on QNLI dev accuracy.

- **Adapter D (`qnli_per_layer_elig`)** — `eligible`  
  Confidence: `high`  
  Reason: outperforms base on QNLI dev accuracy; structurally plausible with A/B.

### Interpretation
The first important decision changed here:

- `qnli_uniform_weak` was downgraded early (`flagged_weak`)
- the candidate set narrowed before pairwise merge planning

This is the first practical value of the workflow: it prevents weak sources from distorting later merge reasoning.

---

## Step 2 — Pairwise Merge-Risk Pass

### Summary
The candidate pairs were evaluated using the standard pairwise merge-risk report.

### Pairwise outcomes

| Pair | Pair risk | Dominant issue | Recommended strategy | Note |
|---|---|---|---|---|
| A × B | `low` | `none` | `linear` | clean anchor-secondary candidate |
| A × C | `medium` | `partial_redundancy` | `norm_equalized` | includes weak source C; not priority |
| A × D | `low` | `none` | `linear` | plausible anchor-ambiguous pairing |
| B × C | `low` | `none` | `linear` | structurally benign, but weak source caveat |
| B × D | `low` | `none` | `linear` | selected ambiguous pair for deeper inspection |
| C × D | `medium` | `partial_redundancy` | `audit_aware` | weak source plus cautionary structure |

### Interpretation
This inventory did not remain a flat compatibility pool:
- A/B/D looked broadly mergeable
- C-related pairs were less trustworthy because C was already flagged weak
- B×D remained the most interesting ambiguous pair despite a low-risk ordinary report

---

## Step 3 — The Ambiguous Pair

### Chosen pair
`qnli_probe_elig × qnli_per_layer_elig`

### Why this pair was selected for deeper inspection
This pair was selected because:
- the ordinary pair-risk report was low risk and not obviously decisive
- the pair remained practically plausible for real use
- it was important enough that deeper structural context could change local decision quality

### Ordinary pair-risk view
- Pair risk: `low`
- Dominant issue: `none`
- Recommended strategy: `linear`

At this stage, the pair looked ordinary and mergeable.

---

## Step 4 — Core-Space Audit on the Ambiguous Pair

### Core-space result
- `shared_basis_score`: `0.9078`
- `basis_distortion`: `0.00346`
- `effective_shared_rank`: `22`
- `status`: `incompatible`

### Interpretation
Although the ordinary pair-risk report did not make this pair look dangerous, core-space flagged the pair as structurally incompatible in shared-basis terms. This changed local handling from:

- "safe default merge candidate"

to:

- "caution track: defer or inspect further before merge"

### Key point
The value of core-space here was not replacing pair-risk globally. It provided extra structural signal on one ambiguous pair that changed the practical next step.

---

## Step 5 — Inventory Summary

### Summary outcome
Inventory summary aggregated:
- adapter statuses: `eligible=3`, `flagged_weak=1`
- pair risk counts: `low=4`, `medium=2`
- dominant issues: `none=4`, `partial_redundancy=2`
- strategy counts: `linear=4`, `norm_equalized=1`, `audit_aware=1`
- strict QA block candidates: `3`

### High-level result
- inventory was not uniformly safe
- source QA narrowed the pool meaningfully
- pair structure supported local organization, not flat exploration
- evidence did not justify treating all six pairs as equally plausible

---

## Step 6 — Merge Neighborhoods

### Neighborhood output

- **Group 1** — `final_uniform_median_r16`, `qnli_probe_elig`, `qnli_per_layer_elig`  
  Characterization: `likely-safe neighborhood`  
  Common strategy: `linear`

- **Excluded** — `qnli_uniform_weak`  
  Reason: `flagged_weak`

- **Boundary warning** — none in this run

### Interpretation
The inventory stopped being a flat pool:
- one local neighborhood became the default exploration region
- one weak source was removed from the active merge plan

Even without a boundary warning, the neighborhood output was operationally useful because it converted pairwise clutter into a clear group/exclude structure.

---

## Final Decision

### Before the workflow
Before running the preflight pass, the inventory looked like:
- 4 plausible adapters
- 6 plausible pair combinations
- no strong reason to avoid broad exploration

### After the workflow
After the full pass:
- `qnli_uniform_weak` was excluded early
- `qnli_probe_elig × qnli_per_layer_elig` moved to caution track after core-space
- one local neighborhood (`A/B/D`) became the primary next evaluation region
- the inventory no longer looked like one flat merge pool

### What decision changed

> The main effect of the workflow was not to produce more scores. It reduced the action space from six plausible merge attempts to one local neighborhood-first plan, with one weak source excluded early and one ambiguous pair explicitly cautioned.

---

## Why This Case Matters

This case shows the full workflow in one place:

1. **Source QA matters first**
2. **Pairwise structural risk is necessary but not always sufficient**
3. **One ambiguous pair may deserve deeper inspection**
4. **The inventory has local structure**
5. **The final decision becomes narrower and more defensible**

That is the practical value of the Gradience workflow.

---

## Output Bundle

### Results directory
`results/case_study/case_study_qnli4_realmix_20260318/`

### Included artifacts
- QA artifacts
- pair reports
- core-space-enhanced pair report for one ambiguous pair
- inventory summary
- neighborhood report
- final decision note
- corpus manifest

### Corpus registration
Manifest:
`results/corpus/manifests/case_study_qnli4_realmix_20260318_20260318.json`

This keeps the worked example inside the same evidence and review ecosystem as the rest of the project.

---

## Publication Readiness Check

- [x] inventory uses 4-5 real adapters
- [x] at least one adapter is excluded or clearly downgraded
- [x] at least one ambiguous pair justifies deeper inspection
- [x] core-space either changes judgment or clearly confirms why it does not
- [x] neighborhood output is nontrivial and interpretable
- [x] final decision note shows a narrower action set than the naive baseline
- [x] all artifacts validate
- [x] corpus manifest exists and validates

---

## One-Paragraph Summary

> In this worked example, Gradience was applied to a small real adapter inventory rather than an isolated pair. The workflow first narrowed the candidate set through single-adapter QA, then identified pairwise structural risk, then used a deeper core-space audit on one ambiguous pair, and finally organized the surviving candidates into a conservative merge neighborhood. The practical effect was a narrower and more defensible action set: one weak adapter excluded early, one ambiguous pair moved to caution, and one local neighborhood identified as the best next evaluation target.
