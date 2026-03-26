# Real Inventory Case Series — Synthesis

## Inventories covered

| # | Inventory ID | Category | Adapters | Base model | Key story |
|---|-------------|----------|----------|------------|-----------|
| 0 | case_study_qnli4_realmix_20260318 | First published example | 4 | distilbert | 1 weak excluded, 1 ambiguous pair cautioned, neighborhood-first plan |
| 1 | roberta_mixed_evidence_4 | Messier mixed-quality | 3 | roberta | All 3 QA statuses different, core-space marginal on only low-risk pair |
| 2 | distilbert_large_pool_6 | Larger pool / neighborhoods | 6 | distilbert | 15 pairs → 1 defensible merge, strongest core-space disagreement |
| 3 | all_eligible_qnli_control | Low-drama control | 3 | distilbert | All eligible, same task. Workflow merely confirmatory. |
| 4 | behaviorally_complete_5 | Behaviorally complete | 5 | distilbert | QA helps but core-space becomes primary driver for credible pool |
| 5 | core_space_hunt_4 | Core-space census | 4 | distilbert | Full 6-pair census: ALL pairs incompatible/marginal. Calibration concern. |
| 6 | two_cluster_neighborhood_7 | Neighborhood stress | 7 | distilbert | 21 pairs compressed to 3 groups. Grouping is QA-driven, not structural. |
| 7 | messy_heterogeneous_6 | Messy pool | 6 | distilbert | Confirms wave 1 pattern: messy pools solved by source QA. |

## Where the workflow stayed strong

### Source QA narrows effectively when pool quality is mixed

In messy pools (wave 1 + T3), source QA produced the largest single reduction. This held across:
- Different base models (distilbert, roberta)
- Different inventory sizes (3, 4, 6 adapters)
- Different QA compositions (mostly unknown, mixed, all-eligible subsets)

The pattern: adapters without behavioral evaluation dominate real-world pools. Source QA catches this before any spectral analysis runs.

### The workflow always changed the practical next step

In all three inventories, the final action set was materially different from what a naive "try all low-risk pairs" approach would have produced:

| Inventory | Naive approach | Workflow outcome |
|-----------|---------------|-----------------|
| case_study_qnli4 | 6 pairs, explore freely | 1 excluded, 1 cautioned, neighborhood-first |
| roberta_mixed_evidence_4 | 1 low-risk pair, merge it | That pair is marginal at depth; inventory is thin |
| distilbert_large_pool_6 | 6 low-risk pairs, try several | 1 defensible pair; cross-group merges are illusory |

### Core-space found real structural signal every time

3/3 uses changed structural judgment. 2/2 target-class instances (low-risk + core-space disagrees) produced genuine structural findings. However, verified adjudication (2026-03) showed that structural judgment changes are not always behaviorally decisive: same-task merges were safe even when flagged as incompatible, and cross-task degradation was already captured by ordinary pair-risk. The diagnostic is narrow, selective, and structurally informative, but its behavioral decision value is more regime-dependent than the case series alone suggested.

## Where the workflow was merely helpful

### Neighborhoods at small inventory size

At 3–4 adapters, neighborhoods confirmed what the pair matrix already showed. They did not add new structure — they restated it in a more compressed form. This is not useless (confirmation has value), but it is not the strong case for neighborhoods.

### Pair-risk distribution alone

Knowing "5 high, 4 medium, 6 low" is informative but does not by itself tell the practitioner which low-risk pairs are truly safe. Pair-risk needs to be combined with QA stratification and (selectively) core-space to become actionable.

## Where the workflow starts to strain

### Neighborhoods fragment at moderate size with heterogeneous pools

In distilbert_large_pool_6, neighborhoods produced 4 singleton caution clusters — one per cycle02 adapter. This is structurally correct but not very informative: "every unknown adapter is alone" is a restatement of the QA stratification, not a new insight. Neighborhoods become most useful when there are enough *credible* adapters to form multi-member groups.

### The workflow cannot substitute for behavioral evaluation

When 4 of 6 adapters lack behavioral eval, the workflow correctly identifies the gap but cannot resolve it. A practitioner who needs those adapters still has to run downstream eval. The workflow tells you *where* the gap is, not *what* the adapters do.

### Small-pool inventories produce a lot of machinery for small decisions

In roberta_mixed_evidence_4 (3 adapters), the full pipeline (QA + 3 pairs + core-space + neighborhoods + inventory summary) is thorough but heavyweight relative to the decision. The overhead is justified only because core-space actually changed the outcome. For an inventory where all adapters are eligible and all pairs are safe, the pipeline would produce a lot of artifacts with little decision value.

## Cross-cutting findings

1. **Source QA > pair audit > neighborhoods** as a narrowing hierarchy, measured by how much each step reduced the candidate space.

2. **Core-space disagrees with low-risk pair reports more often than expected.** 2/2 is a small sample, and the shared_basis_score threshold (~0.87-0.88) reflects real structural divergence. However, verified adjudication showed that structural divergence in this range is not sufficient to predict behavioral harm for same-task pairs. Cross-task pairs below this range did degrade, but ordinary pair-risk already captured that boundary. The threshold remains worth tracking but should not be treated as a behavioral safety boundary without further regime-specific evidence.

3. **Cross-task pairs are the most likely to show core-space disagreement.** Both target-class instances involved adapters from different tasks or training regimes. Same-task, same-policy pairs (like qnli_probe × qnli_uniform) are likely to confirm compatibility.

4. **The norm-imbalance signal is very strong.** Per-layer adapters with custom alpha scaling produced high norm imbalance in both distilbert inventories. This is the most reliable early exclusion signal after source QA.

## Updated narrowing hierarchy (post-adjudication)

Verified adjudication and the task-relationship advisory validation round produced a clearer hierarchy:

| Regime | Primary narrowing | Secondary narrowing | Advisory role |
|--------|------------------|--------------------|-|
| Messy pools | Source QA | Neighborhoods (confirm QA) | Mostly silent |
| Same-task, all eligible | None needed | — | Silent |
| Adjacent-task, credible | **Task advisory** | Pair-risk + neighborhoods | Primary discriminator |
| Distant cross-task | Pair-risk | Task advisory | Clarifying / reinforcing |
| Large, mixed | Source QA + neighborhoods | Task advisory | Partitions matrix into safe/caution zones |

The task-relationship advisory is most valuable in the adjacent-task regime — exactly where pair-risk alone cannot distinguish safe same-task redundancy from harmful cross-task redundancy. See `docs/internal/phase2_rq_synthesis.md` for the full regime map.
