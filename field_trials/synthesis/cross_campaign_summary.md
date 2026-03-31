# Cross-Campaign Summary — CPU Field Research Protocol

**Date:** 2026-03-29
**Campaigns completed:** A (Task-Family Equivalence), B (Marginal-Adapter Behavior)
**Campaigns deferred:** C (Large-Inventory Stress), D (Public-Ecosystem Robustness)

---

## What Was Confirmed

### 1. Task-family equivalence is real and actionable (Campaign A)

Same-family cross-dataset merges (SST-2 x IMDB) behave identically to same-task retained merges. The avg delta for family-test pairs (-0.022) falls within the retained range (-0.017), with a gap of only 0.005. Gradience's strict task-identity boundary — which treats SST-2 x IMDB the same as sentiment x AG News — is overprotective for this case.

**Evidence strength:** 7 merge evaluations, 4 family-test pairs across 2 SST-2 and 2 IMDB adapters on DistilBERT. Single backbone, single task family.

### 2. Barely-weak adapters are safe merge partners (Campaign B)

Near-miss pairs with barely-weak sources (delta -0.002 to -0.004 vs base) show avg merge delta of -0.007, better than retained same-task pairs (-0.018). The evidence gate is slightly overprotective for marginal adapters, but the exclusion is defensible as conservative stance.

**Evidence strength:** 4 near-miss pairs from Phase 2b across DistilBERT and BERT. Observational split, not designed. Barely-weak band well-covered; intermediate band (-0.005 to -0.050) unrepresented.

### 3. Deeply-weak adapters carry genuine risk (Campaign B)

Near-miss pairs with deeply-weak sources (delta -0.150 vs base) show avg merge delta of -0.045, approaching the cross-task control range (-0.096). Merging with a fundamentally broken adapter degrades results even when structural compatibility is fine.

**Evidence strength:** 2 pairs from one deeply-weak source (HateXplain emotion on BERT). Single extreme case; moderate weakness unrepresented.

## What Remains Ambiguous

### 1. Whether the task-family finding generalizes beyond binary sentiment

SST-2 and IMDB are a favorable case: same label semantics, same label count, similar decision structure. Whether NLI variants (MNLI x SNLI x RTE), multi-class classification variants (different topic taxonomies), or cross-format pairs (single-sentence x sentence-pair) show similar patterns is unknown.

### 2. Whether the weakness severity boundary is reliable

The gap between barely-weak (-0.007 avg delta) and deeply-weak (-0.045 avg delta) is clear at the extremes, but the intermediate range (source delta -0.010 to -0.050) is unrepresented. We don't know where the transition occurs.

### 3. Ergonomics at scale (Campaign C, deferred)

Current field trials validated inventories up to 28 pairs. Whether the HTML reports, action plans, and region summaries remain useful at 66+ pairs (12 adapters) has not been tested. This is a usability question, not a decision-quality question.

### 4. Ecosystem robustness (Campaign D, deferred)

How gracefully Gradience handles messy real-world adapters (unusual target modules, transfer-chain bases, sparse metadata, partial failures) was partially tested in Phases 1-2 with TransferGraph adapters, but not systematically. This is an operational hardening question.

## Product Questions Now Settled

| Question | Answer | Source |
|----------|--------|--------|
| Is exact task identity too strict? | Yes, for binary sentiment at minimum | Campaign A |
| Do barely-weak adapters behave like retained? | Yes (avg delta -0.007 vs -0.018) | Campaign B |
| Do deeply-weak adapters carry risk? | Yes (avg delta -0.045, approaching controls) | Campaign B |
| Is the current near-miss category correct? | Yes, but would benefit from severity ranking | Campaign B |

## Product Questions Still Open

| Question | Status | What would settle it |
|----------|--------|---------------------|
| Does task-family equivalence generalize? | Needs 1+ more task family | Replicate on NLI (MNLI x RTE) |
| Where is the weakness severity boundary? | Needs intermediate data | Adapters with delta -0.010 to -0.050 |
| Does scale ergonomics hold at 12+ adapters? | Deferred (Campaign C) | 12-adapter inventory, qualitative |
| Is ecosystem robustness sufficient? | Deferred (Campaign D) | Deliberately messy adapters |

## Recommendation on Campaigns C and D

**Campaign D (robustness)** is the more informative of the two remaining campaigns. The TransferGraph adapters used in Phases 1-2 already exercise the transfer-chain pattern and some unusual configs, but a deliberate search for edge cases would strengthen confidence in product-readiness.

**Campaign C (scale stress)** is lower priority. The current validated ceiling (28 pairs / 8 adapters) covers the majority of practical use cases. Scaling to 66+ pairs is useful for large teams but not urgently needed.

**Suggested disposition:** Run Campaign D when convenient; defer Campaign C unless a specific user need arises.
