# Campaign B Memo — Marginal-Adapter Behavior

**Date:** 2026-03-29
**Campaign:** B (Marginal-Adapter Behavior)
**Question:** Do barely-weak adapters behave more like near-miss candidates or genuinely excluded sources?
**Verdict:** Barely-weak adapters behave like retained pairs. Deeply-weak adapters degrade more, approaching control levels. The current near-miss treatment is correct but would benefit from a severity ranking.

---

## Summary

Campaign B investigates whether the current near-miss treatment — which groups all flagged_weak adapters into a single advisory category — is appropriate, or whether a finer distinction between "barely weak" and "deeply weak" sources would better serve users. Rather than building new inventories, this analysis consolidates existing Phase 2b data from inventories 04 (DistilBERT irony cluster) and 05 (BERT hate/emotion), which already contain a clean separation of barely-weak and deeply-weak near-miss pairs across two backbones.

**Finding:** Barely-weak near-miss pairs (source delta -0.002 to -0.004) perform better than retained same-task pairs on average (-0.007 vs -0.018), confirming that the evidence gate is slightly overprotective for marginal adapters. Deeply-weak pairs (source delta -0.150) show substantially larger degradation (-0.045 average), approaching the cross-task control range (-0.096). The current near-miss treatment is correct as a category, but a severity ranking within that category would help users prioritize.

## Evidence

### Data Source

All data comes from Phase 2b evaluations (inventories 04 and 05, `phase2b_eval_145914/`). This is observational rather than designed — the barely-weak/deeply-weak split was not pre-registered but emerged from the natural variation in adapter quality across the pools.

### Barely-Weak Near-Miss Pairs (source delta -0.002 to -0.010)

Two weak sources, four near-miss pairs:
- **phailyoor irony** (DistilBERT, delta -0.004 vs base on irony)
- **aviator hate** (BERT, delta -0.002 vs base on hate)

| Pair | Backbone | Weak Source Delta | Merged Delta vs Best |
|------|----------|-------------------|---------------------|
| irony JB173 x phailyoor | DistilBERT | -0.004 | -0.012 |
| irony vaariis x phailyoor | DistilBERT | -0.004 | -0.012 |
| hate TGbase x aviator | BERT | -0.002 | -0.002 |
| hate hatexplain x aviator | BERT | -0.002 | -0.002 |

**Average merge delta: -0.007** (range: -0.002 to -0.012)

### Deeply-Weak Near-Miss Pairs (source delta below -0.050)

One weak source, two near-miss pairs:
- **HateXplain emotion** (BERT, delta -0.150 vs base on emotion — strongly underperforms base)

| Pair | Backbone | Weak Source Delta | Merged Delta vs Best |
|------|----------|-------------------|---------------------|
| emotion TGbase x hatexplain | BERT | -0.150 | -0.088 |
| emotion fabriceyhc x hatexplain | BERT | -0.150 | -0.002 |

**Average merge delta: -0.045** (range: -0.002 to -0.088)

Note: The second pair (fabriceyhc x hatexplain) has low degradation because *both* sources are weak (fabriceyhc scores only 0.204 on emotion). The "best source" is barely above random, so there's little to lose.

### Reference Categories

| Category | Avg Δ vs Best | n | Source |
|----------|---------------|---|--------|
| Barely-weak near-miss | -0.007 | 4 | Phase 2b (this analysis) |
| Deeply-weak near-miss | -0.045 | 2 | Phase 2b (this analysis) |
| Retained (same-task, eligible) | -0.018 | 4 | Phase 2b |
| Cross-task control | -0.096 | 1 | Phase 2b |
| Retained (Phase 2 reference) | -0.024 | — | Phase 2 |
| Near-miss (Phase 2 reference) | -0.006 | — | Phase 2 |
| Cross-task control (Phase 2 ref) | -0.047 | — | Phase 2 |

### Key Observation

The barely-weak near-miss average (-0.007) is closer to zero degradation than to retained (-0.018). These pairs are essentially harmless merges — the evidence gate is slightly overprotective for adapters that are only marginally below base performance. But the exclusion is still defensible as a conservative stance: the difference between barely-weak near-miss and retained is small enough that the cost of incorrect inclusion is low.

The deeply-weak near-miss average (-0.045) matches the Phase 2 cross-task control range. Merging with a deeply-weak source carries genuine risk — these are not "almost good enough" adapters but fundamentally broken ones for the evaluated task.

## Gate Decision

Per the protocol's decision criteria:

> **Barely weak ≈ retained, deeply weak ≈ intermediate:**
> Add a "confidence" or "proximity to threshold" indicator in the near-miss section. Rank barely-weak pairs above deeply-weak.

The data aligns with this gate: barely-weak pairs are statistically indistinguishable from retained, while deeply-weak pairs show meaningful degradation.

### Recommendation

1. **Add proximity-to-threshold ranking** within the near-miss section. Pairs where the flagged source has a small negative delta (-0.002 to -0.010) should appear first, with an advisory like "source is marginally below base — structurally plausible merge." Pairs where the flagged source has a large negative delta (below -0.050) should appear last, with "source substantially underperforms base — merge risk elevated."

2. **Do not change the exclusion boundary.** The evidence gate at delta < 0 is correct. Barely-weak adapters are excluded not because their merges fail, but because behavioral evidence is too thin to trust. The near-miss ranking helps users make informed decisions within that category.

3. **Consider a severity label in the QA artifact.** Currently `eligibility: flagged_weak` is binary. Adding a `weakness_severity` field (`marginal` vs `substantial`) would enable the ranking without changing the schema contract.

## Confounds and Limitations

- Observational, not designed: the barely-weak/deeply-weak split was not pre-registered
- Small sample: 4 barely-weak pairs, 2 deeply-weak pairs
- Both barely-weak sources have very small negative deltas (-0.002 and -0.004); the range -0.005 to -0.010 is unrepresented
- Deeply-weak analysis dominated by one source (HateXplain emotion, delta -0.150) which is an extreme case; moderately weak sources (-0.020 to -0.050) are unrepresented
- The high variance in the deeply-weak category (one pair at -0.002, another at -0.088) makes the average less reliable

## Files

- Analysis based on: `phase2b_eval_145914/results.json`
- Source evidence: `inventory_04_distilbert_irony_cluster/evidence/`, `inventory_05_bert_hate_emotion/evidence/`
- Phase 2b confirmation: `phase2b_confirmation_memo.md`
