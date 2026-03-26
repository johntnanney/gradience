# Verified Adjudication Implications

## Purpose

This note records the implications of the verified adjudication study on freshly trained DistilBERT LoRA adapters.

The main question of the study was:

> When ordinary pair-risk and core-space disagree, which signal better predicts downstream merge behavior?

The study produced a clear regime-bound result.

---

## Study setup

The adjudication set used:

- 6 LoRA adapters trained from scratch on `distilbert-base-uncased`
- 3 SST-2 adapters with independently verified performance above base
- 3 QNLI adapters with independently verified performance above base
- all 15 pairwise merges
- downstream evaluation on both tasks after merge
- core-space analysis and ordinary pair-risk reporting for all pairs

This is the first adjudication set in the project where source quality was independently verified rather than user-claimed.

---

## Main findings

### 1. Same-task merges are safe even when core-space says "incompatible"

All 6 same-task pairs preserved accuracy within approximately 1.2 percentage points of the best individual source adapter.

This includes pairs where core-space returned `incompatible` with shared-basis scores in the `0.870-0.873` range.

Interpretation:

- core-space is detecting real structural divergence between adapters trained on the same task
- but that divergence is not behaviorally decisive in this regime

This means structural incompatibility at the shared-basis level is **not sufficient** to infer harmful merge behavior for same-task pairs.

### 2. Cross-task merges degrade both tasks substantially

Cross-task merges degraded both tasks by approximately 8-18 percentage points.

Interpretation:

- the main signal here is task mismatch
- ordinary pair-risk already separates these pairs from same-task safe pairs
- deeper structural information adds only modest additional discrimination inside an already-bad regime

### 3. Ordinary pair-risk already separates safe from unsafe in this regime

Within this verified DistilBERT setting:

- same-task pairs were classified as `redundant` and behaved safely
- cross-task pairs were classified as `imbalanced` and degraded substantially

Interpretation:

- the stable pair-risk layer performed well on the central safe/unsafe distinction
- core-space was not needed to draw the main boundary

### 4. Core-space is structurally informative but not broadly decision-changing here

Core-space did produce real structure:

- same-task seed variants were sometimes marked `incompatible`
- cross-task pairs were also marked `marginal` or `incompatible`

But behaviorally:

- it overwarned on same-task pairs
- it added only about ~1 percentage point of additional discriminative signal inside the already-unsafe cross-task group

Interpretation:

- core-space remains a real structural diagnostic
- but its behaviorally useful role is narrower than previously assumed

---

## Main implication for Gradience

The stable pair-risk workflow is strengthened by this study.

In this regime, ordinary pair-risk already separates:

- safe same-task merges

from

- degraded cross-task merges

That is a strong validation of the core decision layer.

The implication for core-space is narrower.

Core-space should now be treated as:

> an advanced structural diagnostic whose decision value is regime-dependent and likely concentrated in pairs where task relationship is genuinely ambiguous

It should **not** currently be positioned as a broadly necessary advanced layer for ordinary same-task or obvious cross-task decisions.

---

## What this result does not imply

This study does **not** justify removing core-space from the project.

It does justify narrowing its claim.

The initial study used one regime (DistilBERT, SST-2 x QNLI). A subsequent replication on roberta-base confirmed the same pattern, strengthening the claim to small encoder models generally. But the scope remains:
- freshly trained adapters
- verified-source adjudication

The result shows where core-space does **not** add broad decision value. It does not yet answer whether it becomes more useful in harder intermediate regimes, such as:

- related but non-identical tasks
- same task family across domain shift
- style/control variants
- structurally plausible but semantically ambiguous adapters

Those are now the relevant next test regimes.

---

## Updated interpretation of core-space

### Previous implicit position

Core-space may often provide meaningful advanced discrimination beyond ordinary pair-risk.

### Updated position

Core-space is:

- structurally real
- advanced and non-default
- useful to inspect in selected ambiguous cases

But in this verified adjudication regime it is:

- not broadly behaviorally decisive
- too pessimistic for same-task seed variants
- only modestly informative inside obvious cross-task mismatch cases

This means its strongest current claim is:

> core-space may matter in genuinely ambiguous relationships where ordinary pair-risk looks permissive and task mismatch is not already doing the main explanatory work

---

## Recommended project actions

### 1. Tighten core-space claims everywhere

Update documentation and outward-facing language so core-space is described as:

- narrow
- regime-dependent
- advanced
- structurally informative
- not broadly decision-changing

### 2. Preserve the stable pair-risk story

This study strengthens the case that:

- source QA
- ordinary pair-risk
- neighborhood-level inventory compression

remain the center of the workflow

### 3. Design the next adjudication regime carefully

The next useful adjudication study should target:

- related-but-not-identical tasks
- nontrivial but plausible pair relationships
- cases where ordinary pair-risk is permissive but task relationship is genuinely unclear

---

## Bottom line

The verified adjudication study did not show that core-space is broadly necessary.

It showed something more useful:

- ordinary pair-risk is strong in this regime
- task relationship dominates deeper geometry here
- core-space remains structurally meaningful but behaviorally narrow

That is a cleaner and more trustworthy position for the project.
