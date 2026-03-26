# Preflight Utility — Evidence Card

## Claim

Gradience reduces wasted merge exploration by partitioning mixed-task inventories into same-task safe zones and cross-task caution zones, typically eliminating 65-90% of candidate pairs before evaluation begins.

## Evidence summary

| Inventory | Category | Adapters | Total pairs | Retained | **Reduction** |
|-----------|----------|----------|-------------|----------|---------------|
| same_task_qnli_4 | Same-task control | 4 | 6 | 6 | 0% |
| mixed_sst2_qnli_4 | Standard mixed-task | 4 | 6 | 2 | **67%** |
| large_4task_8 | Large 4-task | 8 | 28 | 4 | **86%** |
| messy_mixed_5 | Messy mixed-quality | 5 | 10 | 8 | 20% |
| confusing_nli_5 | Confusing NLI+SST-2 | 5 | 10 | 1 | **90%** |
| **Total** | | **26** | **60** | **21** | **65%** |

## Key metrics

| Metric | Value |
|--------|-------|
| Inventories tested | 5 |
| Total pairs | 60 |
| Average reduction (all mixed-task) | 66% |
| Average reduction (advisory-dominant) | **81%** |
| Strongest reduction | 90% (confusing NLI+SST-2) |
| Advisory false positives | 0 |
| Same-task pairs correctly retained | 21/21 |

## Where the workflow is strongest

Mixed-task inventories where pair-risk rates most pairs as "medium" — structurally indistinguishable. The advisory is the only signal that separates safe same-task pairs from cross-task caution pairs.

## Where it is confirmatory

Same-task pools and messy pools where QA or high pair-risk already dominate the narrowing.

## Provenance

`docs/internal/preflight_utility_round_01_synthesis.md`
