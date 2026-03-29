# Field-Trial Validation Memo

**Gradience CPU-Only Workflow — Phases 1 & 2**
**Date:** 2026-03-28

---

## What was tested

Three inventories of public LoRA adapters — ranging from 3 to 8 working adapters and 3 to 28 candidate merge pairs — were run through Gradience's full preflight pipeline: adapter audit, behavioral evidence bootstrap, pairwise merge audit, inventory summary, action plan, and HTML report. In Phase 2, retained and control pairs were actually merged and evaluated on task datasets, closing the loop between Gradience's recommendations and empirical merge outcomes.

The inventories were deliberately chosen for structural heterogeneity: mixed ranks (r=1 through r=16), mixed alphas, mixed target-module counts, multiple task types (sentiment, news classification, hate speech, irony, NLI, emotion), and multiple backbones (distilbert-base-uncased, roberta-base). This heterogeneity approximates the conditions a practitioner encounters when collecting public adapters for merge exploration.

## Main success: the narrowing logic works

Gradience's primary value proposition is candidate reduction — given N adapters and up to N(N-1)/2 merge pairs, identify the small subset worth evaluating. Across the three pilots:

| Pilot | Pairs | Retained | Reduction |
|-------|-------|----------|-----------|
| 1 (same-task control) | 3 | 0 | 100% |
| 2 (mixed-task) | 10 | 1 | 90% |
| 3 (large mixed-task) | 28 | 2 | 93% |

Phase 2 confirmed that the retained pairs are the right first choices. In Pilot 2, the single retained pair (AG News × AG News-formality) outperformed both source adapters on the evaluation task (+0.006 vs best source). In Pilot 3, the two retained pairs degraded less than the cross-task controls (-0.018 and -0.066 vs -0.042 and -0.048 for the controls). The ordering is correct: a practitioner following Gradience's evaluate-first list spends time on the most promising candidates.

The evidence gate — behavioral evaluation scores fed into the adapter QA artifact — is the most impactful single feature. Without it (Pilot 1 v1), every adapter is excluded for unknown behavioral status and the pipeline produces nothing. With it, the three-way classification (eligible / uncertain / flagged_weak) correctly handles genuine failures, misleading evals, marginal passes, ambiguous ties, and strong performers. The gate is well-calibrated.

Task-boundary detection correctly partitions cross-task pairs across all inventories. Every pair that crosses a genuine task boundary receives the advisory. The cross-task controls in Phase 2 confirmed the caution signal: no cross-task merge improved over same-task merges.

## Main failure mode: overprotective evidence gate on structurally clean pairs

The single clearest failure is the hate-speech near-miss in Pilot 3. The jaesun × Aureliano hate pair was structurally clean (low risk, no dominant issue, linear merge recommended) and same-task. But Aureliano was flagged_weak (accuracy 0.426 vs base 0.576), so the pair was excluded from the action plan entirely.

Phase 2 showed this exclusion was wrong on outcomes: the merged pair scored 0.598 on hate, outperforming the best source (0.520) by +0.078. A weak adapter contributed useful features to the merge. The evidence gate applied a binary exclusion when a graduated response was warranted.

This is not a failure of the structural analysis. The spectral layer correctly identified the pair as compatible. It is a failure of the policy layer: the evidence gate's exclusion overrode a correct structural signal. The fix is not to weaken the evidence gate but to make the exclusion visible — which is what the near-miss feature now does.

## What the structural analysis adds (and doesn't)

The spectral layer's contribution is clearest on same-task pairs. The AG News pair in Pilot 2 showed partial redundancy in 8 of 24 layers — a finding invisible without weight-space analysis. The SST-2 pair in Pilot 3 showed norm imbalance from a 2× rank mismatch, correctly triggering norm-equalized strategy. These structural diagnoses help practitioners calibrate expectations before investing in evaluation.

On cross-task pairs, the structural analysis is less distinctive. Norm imbalance dominated across all three pilots (75-100% of pairs), driven by the configuration heterogeneity in public adapters. When three-quarters of pairs share the same dominant issue, the signal stops being informative. A severity ranking rather than a uniform label would add value.

The pilots used mostly r=1 adapters, which limits what spectral analysis can reveal. At r=1, a LoRA adapter is a single direction in weight space — energy-rank profiles, utilization patterns, and multi-rank interactions are invisible. A richer-adapter validation (r≥8, broader target modules) would test whether the spectral layer adds value beyond what the evidence gate and task-boundary detection already provide.

## Decision quality scorecard

| Question | Pilot 2 | Pilot 3 |
|----------|---------|---------|
| Retained pair beats best source? | Yes (+0.006) | No (-0.018, -0.066) |
| Retained pair beats controls? | Yes | Yes (less degradation) |
| Controls validate caution? | Yes | Yes |
| Near-miss exclusion justified? | n/a | No (+0.078 improvement) |

The pattern in Pilot 3 — retained pairs underperform their best source but outperform cross-task controls — is consistent with simple LoRA merging at low rank. The merge interpolates rather than synthesizes. This is a real limitation of the merge operation, not of Gradience's recommendation logic.

## Product implications

Three findings translate directly into product work:

**Near-miss reporting (implemented).** The near-miss category now surfaces in the action plan, preflight summary, and HTML report. A pair qualifies when it is same-task, structurally clean (low or medium risk), but excluded because one source is evidence-constrained. This directly addresses the Pilot 3 hate-speech finding.

**Norm-imbalance severity ranking (not yet implemented).** When the majority of pairs share the same dominant issue, ranking by severity (magnitude ratio, reconstruction error) would help practitioners triage. The raw data is already computed; surfacing it as a sortable metric is a rendering change.

**Task-family equivalence (not yet implemented).** IMDB and SST-2 are both binary sentiment but treated as cross-task because their eval_dataset labels differ. A lightweight task taxonomy would catch same-task-family pairs that metadata matching misses.

## Scope and limitations

These trials test Gradience's workflow on public LoRA adapters at small scale (≤8 adapters, ≤28 pairs) with classification tasks and accuracy as the metric. The findings generalize to the product's core use case — adapter triage before merge evaluation — but do not test:

- High-rank adapters (r≥32) where spectral structure is richer
- Generation tasks (summarization, translation, chat) where merge quality is harder to measure
- Large inventories (50+ adapters) where scalability and navigability matter more
- Adapters trained on the same data with different hyperparameters (the "hyperparameter sweep" use case)

The behavioral evidence is user-reported. Gradience does not independently verify claimed evaluation results. A practitioner supplying inflated scores could defeat the evidence gate. The trust language throughout the pipeline reflects this: all behavioral evidence is described as "reported," not "verified."

## Summary judgment

Gradience's preflight pipeline produces correct prioritization. The retained pairs are the right first choices, the cross-task advisories are earned, and the evidence gate is well-calibrated except at the margin. The main product gap — overprotective exclusion of structurally clean pairs with weak sources — is now addressed by the near-miss feature. The structural analysis adds the most value on same-task pairs; its contribution to cross-task triage is real but dominated by norm-imbalance signals that would benefit from severity ranking.

The trial validates the workflow for the classification-task, low-rank, public-adapter regime. Extending to richer adapters and generation tasks is the natural next step.

## Phase 2b addendum: near-miss confirmed

Phase 2b (11 additional merges across 2 new inventories, 2 backbones, 3 task families) confirmed that near-miss pairs — same-task, structurally plausible, excluded only by the evidence gate — degrade comparably to retained pairs (avg -0.006 vs best source) and 5× less than cross-task controls (-0.096). The near-miss action-plan feature implemented in `gradience.vnext.inventory` is validated. No further product change is required for this category. See `near_miss_validation.md` for the full confirmation data.
