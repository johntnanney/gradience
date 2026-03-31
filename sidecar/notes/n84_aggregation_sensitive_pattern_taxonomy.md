# n84 -- Aggregation-Sensitive Pattern Taxonomy

**Type:** findings note
**Date:** 2026-03-31
**Program:** Aggregation-Sensitive Compatibility (Route 2)
**Stage:** D
**Depends on:** n81 (panel), n82 (family audit), n83 (comparison analysis)
**Status:** complete

---

## Question

Do aggregation-sensitive divergences cluster into recognizable, recurrent patterns? If so, what are the patterns and what determines which pattern a case falls into?

---

## Method

Cluster the 12 cases from the Stage C comparison by their agreement profile (which families agree, which diverge, and why). Identify structural features that predict pattern membership.

---

## Taxonomy

### Pattern 1: Aggregation-invariant exclusion (2/12)

**Cases:** mrg_cross_task_control, tri_cross_task_qa_clear

All four aggregation families reach the same operational conclusion. Cross-task relation + clear or irrelevant QA. The structural signal is strong enough and the task boundary is clear enough that the aggregation rule does not matter.

**Predictor:** Cross-task relation with no QA complication.

### Pattern 2: Distributional gradient case (4/12)

**Cases:** rte_same_task_confusable, mnli_qnli_moderate, qnli_rte_separable, mrg_near_miss_substantial

Distributional aggregation reveals a meaningful gradient (confusable → moderate → separable) that worst-case collapses to a single label. Same-task or same-family relation with moderate structural compatibility and no QA constraint.

**Predictor:** Same-task or same-family with structural-only QA regime and no binding evidence constraint.

### Pattern 3: QA dominance override (2/12)

**Cases:** tri_same_task_qa_blocked, tri_same_family_qa_blocked

QA-dominant aggregation blocks cases that structural aggregation would pass. The pair with the highest structural compatibility in the entire panel (0.892) falls here. Evidence status overrides structural truth.

**Predictor:** QA-blocked regime regardless of structural compatibility level.

### Pattern 4: QA-gated enrichment (3/12)

**Cases:** tri_same_task_qa_clear, tri_same_family_qa_clear, mrg_safe_same_task

When QA clears, the hybrid (QA-gated distributional) produces richer output than either QA-dominant or worst-case alone. It preserves both the evidence constraint and the structural gradient.

**Predictor:** QA-clear regime with structural data available for distributional analysis.

### Pattern 5: Mixed evidence nuance (1/12)

**Cases:** tri_cross_task_qa_review

QA regime is neither fully clear nor fully blocked. One source eligible, one uncertain. QA-dominant produces the most granular label in the panel (review rather than blocked/clear).

**Predictor:** Mixed evidence status across sources.

---

## Pattern predictors

The two dominant predictors are:

1. **QA regime** — determines whether the case is in pattern 1/4 (clear), pattern 3 (blocked), pattern 5 (mixed), or pattern 2 (structural-only, no QA data).
2. **Task relation** — separates pattern 1 (cross-task, aggregation-invariant) from patterns 2/4 (same-task or same-family, aggregation-sensitive).

These two features together predict all 5 patterns without reference to any structural compatibility score.

---

## Key finding

The patterns are not arbitrary — they are predictable from two observable features (QA regime and task relation) that are known before any aggregation computation. This means a system could select the appropriate aggregation family based on case metadata, without running all four families on every case.

---

## Output artifacts

- `sidecar/results/aggregation_sensitive_compatibility/pattern_taxonomy.json`
- `sidecar/results/aggregation_sensitive_compatibility/pattern_taxonomy.md`
- `sidecar/notes/n84_aggregation_sensitive_pattern_taxonomy.md` (this note)
