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
`<inventory_id>`

### Base model
`<base_model>`

### Inventory size
`4` or `5` adapters

### Practical question
This inventory was selected to answer a practical question of the form:

> Given several plausible adapters, which ones are worth preserving, which pairs are worth considering, and whether the inventory has any local merge structure at all before downstream evaluation begins.

### Adapter pool
Example presentation format:

- **Adapter A** — `<name>`  
  Intended role: strong anchor adapter

- **Adapter B** — `<name>`  
  Intended role: related secondary adapter

- **Adapter C** — `<name>`  
  Intended role: weak or exclusion-worthy adapter

- **Adapter D** — `<name>`  
  Intended role: ambiguous / structurally non-obvious adapter

- **Adapter E** *(optional)* — `<name>`  
  Intended role: boundary / cross-neighborhood adapter

### Why this inventory was chosen
This inventory was chosen because it contains:
- at least one plausible strong source
- at least one likely weak or exclusion-worthy source
- at least one pair that is not trivially resolved by standard pair-risk logic
- enough diversity to make neighborhood output nontrivial

---

## Step 1 — Single-Adapter QA

### Summary
The first pass was single-adapter QA. This immediately narrowed the candidate set.

### Adapter QA outcomes

- **Adapter A (`<name>`)** — `eligible`  
  Confidence: `<high|medium|low>`  
  Reason: `<brief reason>`

- **Adapter B (`<name>`)** — `eligible` or `uncertain`  
  Confidence: `<high|medium|low>`  
  Reason: `<brief reason>`

- **Adapter C (`<name>`)** — `flagged_weak` or `unknown_no_behavioral_eval`  
  Confidence: `<high|medium|low>`  
  Reason: `<brief reason>`

- **Adapter D (`<name>`)** — `eligible` / `uncertain`  
  Confidence: `<high|medium|low>`  
  Reason: `<brief reason>`

- **Adapter E (`<name>`)** *(optional)* — `<status>`  
  Confidence: `<high|medium|low>`  
  Reason: `<brief reason>`

### Interpretation
The first important decision changed here:

- `<Adapter C>` was excluded or downgraded early
- the candidate set was reduced before any pairwise merge decision was made

This is the first practical value of the workflow: it prevents weak sources from distorting later merge reasoning.

---

## Step 2 — Pairwise Merge-Risk Pass

### Summary
The remaining candidate pairs were evaluated using the standard pairwise merge-risk report.

### Pairwise outcomes

| Pair | Pair risk | Dominant issue | Recommended strategy | Note |
|---|---|---|---|---|
| A × B | `<low|medium|high>` | `<issue>` | `<strategy>` | `<1 short note>` |
| A × D | `<low|medium|high>` | `<issue>` | `<strategy>` | `<1 short note>` |
| B × D | `<low|medium|high>` | `<issue>` | `<strategy>` | `<1 short note>` |
| A × E | `<low|medium|high>` | `<issue>` | `<strategy>` | `<1 short note>` |
| D × E | `<low|medium|high>` | `<issue>` | `<strategy>` | `<1 short note>` |

Use only actual pairs present in the case study.

### Interpretation
At this stage, the inventory usually stops looking like a flat compatibility pool.

Typical pattern:
- one or two pairs are clearly benign
- one pair is clearly poor or cautionary
- one pair remains genuinely ambiguous

That ambiguous pair is what justifies the next step.

---

## Step 3 — The Ambiguous Pair

### Chosen pair
`<Adapter X × Adapter Y>`

### Why this pair was selected for deeper inspection
This pair was selected because:
- the ordinary pair-risk report was not obviously decisive
- the pair remained practically plausible
- the pair was important enough that further structural context could change the local decision

### Ordinary pair-risk view
- Pair risk: `<low|medium|high>`
- Dominant issue: `<issue>`
- Recommended strategy: `<strategy>`

At this stage, the pair looked:
- `<plausible / ordinary / not obviously alarming / medium-risk but still ambiguous>`

---

## Step 4 — Core-Space Audit on the Ambiguous Pair

### Core-space result
- `shared_basis_score`: `<value>`
- `basis_distortion`: `<value>`
- `effective_shared_rank`: `<value>`
- `status`: `<compatible|marginal|incompatible|not_applicable>`

### Interpretation
This is where the advanced diagnostic either earned its place or did not.

#### Example case where it matters
Although the ordinary pair-risk report did not make this pair look especially dangerous, the core-space result suggested that the two adapters did **not** fit cleanly into a shared low-rank basis.

That changed the local judgment from:

- "plausible merge candidate"

to:

- "caution / defer / inspect further before merge"

#### Example case where it does not matter
If the core-space result did **not** materially change judgment, say so plainly:

> In this case, core-space largely confirmed the ordinary pair-risk view and did not change the practical decision. This is still a useful result because it shows the advanced diagnostic is being used selectively rather than as mandatory decoration.

### Key point
The value of the core-space audit is not that it always overrides the standard pair-risk report. Its value is that, in a narrower class of ambiguous cases, it can add structural clarity that changes or sharpens the decision.

---

## Step 5 — Inventory Summary

### Summary outcome
The inventory summary aggregated:
- adapter statuses
- pairwise risk distribution
- dominant issue distribution
- strategy distribution

### High-level result
Example interpretation:

- the inventory was not uniformly safe
- the candidate set narrowed meaningfully after QA
- pairwise structure suggested local organization rather than a flat merge pool
- no evidence justified treating all remaining pairs as equally plausible

This step matters because it turns isolated judgments into a coherent inventory-level picture.

---

## Step 6 — Merge Neighborhoods

### Neighborhood output

Example format:

- **Group 1** — `<Adapter A, Adapter B>`  
  Characterization: `likely-safe neighborhood`  
  Common strategy: `<linear / norm_equalized / audit_aware>`

- **Group 2** — `<Adapter D>`  
  Characterization: `caution neighborhood` or singleton

- **Excluded** — `<Adapter C>`  
  Reason: `<flagged_weak / missing evidence / other>`

- **Boundary warning** — `Group 1 ↔ Adapter D`  
  Reason: `<short reason>`

### Interpretation
This is the point where the inventory stops being a flat pool of merge options.

Instead, it resolves into:
- one local safe or safer region
- one excluded source
- one cautionary boundary or standalone candidate

This is often more useful than a long list of pairwise reports because it tells the practitioner **how to think about the whole pool**, not just one edge at a time.

---

## Final Decision

### Before the workflow
Before running the Gradience pass, the inventory looked like:
- `<N>` plausible adapters
- several plausible pair combinations
- no strong reason not to explore the pool broadly

### After the workflow
After the full pass:
- `<Adapter C>` was excluded or downgraded early
- `<Pair X × Y>` was narrowed, deferred, or treated with extra caution
- one local neighborhood emerged as the best next place to explore
- the inventory no longer looked like one flat merge pool

### What decision changed
This is the most important sentence in the document:

> The main effect of the workflow was not to produce more scores. It was to reduce the action space from "several plausible merge attempts" to "one safe local neighborhood plus one explicitly cautionary case," while excluding one weak source before wasted evaluation began.

If that is not true for the selected inventory, the inventory is probably not strong enough for publication as the worked example.

---

## Why This Case Matters

This case matters because it shows the full workflow in one place:

1. **Source QA matters first**
2. **Pairwise structural risk is necessary but not always sufficient**
3. **One ambiguous pair may deserve deeper inspection**
4. **The inventory has local structure**
5. **The final decision becomes narrower and more defensible**

That is the practical value of the Gradience workflow.

---

## Output Bundle

### Results directory
`results/case_study/<inventory_id>/`

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
`results/corpus/manifests/<case_study_manifest>.json`

This keeps the worked example inside the same evidence and review ecosystem as the rest of the project.

---

## Publication Readiness Check

Use this checklist before treating the case as outward-facing material.

- [ ] inventory uses 4-5 real adapters
- [ ] at least one adapter is excluded or clearly downgraded
- [ ] at least one ambiguous pair justifies deeper inspection
- [ ] core-space either changes judgment or clearly confirms why it does not
- [ ] neighborhood output is nontrivial and interpretable
- [ ] final decision note shows a narrower action set than the naive baseline
- [ ] all artifacts validate
- [ ] corpus manifest exists and validates

---

## One-Paragraph Summary

Here is the one-paragraph version that can be reused later:

> In this worked example, Gradience was applied to a small real adapter inventory rather than an isolated pair. The workflow first narrowed the candidate set through single-adapter QA, then identified pairwise structural risk, then used a deeper core-space audit on one ambiguous pair, and finally organized the surviving candidates into conservative merge neighborhoods. The practical effect was not simply more reporting. It was a narrower and more defensible action set: one weak adapter excluded early, one ambiguous pair downgraded or treated cautiously, and one local neighborhood identified as the most plausible next step for evaluation.
