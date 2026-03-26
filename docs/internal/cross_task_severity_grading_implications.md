# Cross-Task Severity Grading — Implications

## Anchor finding

Cross-task severity is primarily determined by **exact task-pair identity**, not by format similarity, source-strength gap, or coarse structural signals. Each task pairing has a characteristic severity profile that is stable across seed combinations.

## Factor ranking

| Factor | Severity-relevant? | Role |
|--------|-------------------|------|
| **Exact task-pair identity** | **Yes — dominant** | Determines severity class; stable across seeds |
| **Core-space shared-basis** | **Yes — secondary** | Separates catastrophic (~0.85) from rest (~0.90+); first structural signal with credible severity relevance inside cross-task |
| Task-direction dominance | Partial | Explains asymmetry direction (stronger task survives) |
| Source-strength gap | No | Medium-gap pairs are actually worse (confounded with task identity) |
| Format similarity | **No — misleading** | Same-format contains all catastrophic cases |
| Pair-risk | No | Rates 83% of cross-task as medium regardless of severity |
| Task advisory | No — by design | Catches boundary, not gradient |

## What this means for Gradience

### Advisory: boundary layer, confirmed and sufficient
The advisory cleanly separates same-task (safe) from cross-task (degraded). It fires uniformly on all 24 cross-task pairs. It is not designed for severity grading and should not be expected to provide it. Its role is established and correct.

### Core-space: first credible severity-relevant use case
This is the first study where core-space adds something not captured by other signals. Catastrophic pairs cluster at shared-basis ~0.85 while non-catastrophic cross-task pairs are at ~0.90+. The overlap between non-catastrophic categories (mild/asymmetric/broad) is substantial, so core-space does not grade finely. But it may distinguish "likely catastrophic" from "likely degraded but manageable."

This does not justify making core-space default. It suggests a narrower and more defensible role: **severity triage within advisory-flagged cross-task pairs**, specifically for identifying likely catastrophic interactions.

### Pair-risk: still too coarse for this regime
Pair-risk rates 20/24 cross-task pairs as medium. It does not distinguish a 2pp mild pair from a 42pp catastrophic pair. This is not a failure of pair-risk — it was designed for structural compatibility, not task-functional compatibility. But it means pair-risk alone is insufficient for severity interpretation in cross-task inventories.

### Exact task-pair identity: the elephant in the room
The strongest severity predictor is which two tasks are being merged. This is not currently a Gradience signal. Whether it should become one is a design question that should not be resolved prematurely. The current evidence shows that task-pair identity matters, but it does not yet show how to operationalize it without building a task-family ontology (which the project has correctly avoided so far).

## What not to do yet

1. **Do not add a task-pair severity feature.** The evidence shows task-pair identity matters but does not yet show how to generalize it beyond the 6 task pairs tested.

2. **Do not make core-space default for cross-task pairs.** The evidence is suggestive but not yet strong enough to change the non-default tier. One study with one catastrophic task pair is not sufficient.

3. **Do not build a task-family ontology.** The finding that "related" tasks are NOT safer makes a closeness-based ontology actively misleading.

4. **Do not retune pair-risk.** Pair-risk's coarseness in this regime is a feature boundary, not a bug to fix. The right response is better interpretation, not threshold changes.

## What to do next

1. **Collect more task pairs.** The current study has 6 cross-task pairings. To test whether core-space severity grading generalizes, more task combinations are needed — especially more pairs in the catastrophic zone to see if shared-basis ~0.85 is a consistent marker.

2. **Test on a second backbone.** The severity profiles observed here may be distilbert-specific. A roberta-base replication would strengthen or weaken the generalization claim.

3. **Document the QNLI × MRPC catastrophic case.** This should be a named reference case in project documentation. It is the clearest counterexample to "related tasks merge safely" and the strongest evidence for exact task-pair identity as the key factor.

## Bottom line

The advisory catches the cross-task boundary. Core-space may help triage catastrophic pairs within that boundary. But the strongest explanatory factor — exact task-pair identity — is not yet operationalizable as a Gradience signal. The right next step is more evidence on whether core-space severity triage generalizes, not premature feature construction.
