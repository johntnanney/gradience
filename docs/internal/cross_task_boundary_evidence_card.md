# Cross-Task Boundary — Evidence Card

## Cumulative evidence base

| Evidence area | Result |
|--------------|--------|
| Same-task blind-spot studies | 3 (style, domain, strength) |
| Same-task pairs tested | 49 (45 blind-spot + 4 subtype controls) |
| Material same-task degradations | 0 |
| Cross-task subtype + severity study pairs | 28 (4 tasks, 8 adapters) |
| Cross-task pairs evaluated | 24 |
| Advisory false positives | 0 across 132+ pairs |
| Main regime boundary | **Task identity** |

## Severity decomposition (24 cross-task pairs)

| Severity | Count | % | Description |
|----------|-------|---|-------------|
| near_safe | 0 | 0% | — |
| mild_degradation | 8 | 33% | Both tasks degrade modestly (1.5-4pp) |
| asymmetric_dilution | 9 | 38% | Stronger task preserved; weaker degrades >4pp |
| broad_degradation | 5 | 21% | Both tasks degrade >4pp |
| catastrophic | 3 | 13% | One task degrades >15pp or both >8pp |

**No cross-task pair was near-safe.** All 24 degraded by at least 1.5pp on one task.

## Key explanatory finding: format axis does NOT predict severity

| Format axis | Pairs | Mean max delta | Max delta |
|-------------|-------|---------------|-----------|
| Same-task | 4 | 0.8pp | 2.2pp |
| Same-format (sentence-pair × sentence-pair) | 12 | 9.7pp | **41.7pp** |
| Cross-format (sentence-pair × single-sentence) | 12 | 8.8pp | 12.7pp |

**Same-format pairs are NOT safer than cross-format.** The worst pair in the study (QNLI × MRPC, -41.7pp) is same-format. Task-format similarity is misleading, not protective.

## What DOES predict catastrophic severity

| Factor | Catastrophic pairs (3) | Non-catastrophic cross-task (21) |
|--------|----------------------|--------------------------------|
| All involve QNLI × MRPC | 3/3 | 0/21 |
| Source-strength gap band | medium_gap | mostly high_gap |
| Pair-risk | medium | medium (mostly) |

**The QNLI × MRPC interaction is specifically catastrophic.** This is not explained by format (both are sentence-pair tasks), source-strength gap (medium), or pair-risk (medium). It is a task-specific interaction where the functional mappings are incompatible despite structural similarity.

## Asymmetric dilution pattern

9 of 24 cross-task pairs show asymmetric dilution. The pattern is consistent:
- SST-2 (stronger, higher-accuracy) preserves its task
- The weaker task (QNLI, RTE, MRPC) degrades 5-13pp
- The dominant adapter's task performance drops only 1-3pp

This is the most common cross-task failure mode and is driven by source-strength asymmetry: the stronger adapter contributes more to the linear merge.

## What current Gradience signals capture

| Signal | Captures severity? |
|--------|--------------------|
| Pair-risk | No — rates 20/24 cross-task pairs as medium |
| Task advisory | Yes for same/different boundary; no severity grading |
| Core-space | Not tested in this study |
| Source-strength gap | Partially — catastrophic pairs have medium gap, not high |

**Gap: no current signal distinguishes mild from catastrophic within cross-task pairs.** The advisory correctly flags all 24 as cross-task, but all 24 get the same warning regardless of whether they'll degrade by 2pp or 42pp.

## Verdict

**`severity_subtypes_confirmed`**

Cross-task pairs decompose into at least 4 distinct severity levels. The key driver of catastrophic interference is task-specific functional incompatibility (QNLI × MRPC), not format similarity, source-strength gap, or structural risk. Current Gradience signals do not distinguish severity within cross-task pairs.

## Provenance

- `results/cross_task_subtype_study_01/`
- `docs/internal/regime_map_after_phase2.md`
- `docs/internal/same_task_regime_closed_note.md`
