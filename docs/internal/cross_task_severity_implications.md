# Cross-Task Severity Decomposition — Implications

## Study summary

28 pairs across 4 tasks (QNLI, RTE, MRPC, SST-2) on distilbert-base-uncased. 8 verified-good adapters (2 per task). All 24 cross-task pairs merged with uniform_linear and evaluated on both tasks.

## Core findings

### 1. Cross-task subtype structure is confirmed

Cross-task merges are not one uniform "bad" bucket. They decompose into 4 distinct severity levels:

| Severity | Count | % | Pattern |
|----------|-------|---|---------|
| Mild degradation | 8 | 33% | Both tasks degrade 1.5-4pp |
| Asymmetric dilution | 9 | 38% | Stronger task preserved; weaker degrades 5-13pp |
| Broad degradation | 5 | 21% | Both tasks degrade >4pp |
| Catastrophic interference | 3 | 13% | One task degrades >15pp or both >8pp |

### 2. No cross-task pair is near-safe

Zero of 24 cross-task pairs remained within 1.5pp of best source on both tasks. This is the strongest evidence yet that **cross-task merging is always degrading** on small encoder models — the question is only how much.

### 3. Same-format does NOT imply safer merging

| Format axis | Pairs | Mean max delta | Worst pair |
|-------------|-------|---------------|------------|
| Same-format (sentence-pair × sentence-pair) | 12 | 9.7pp | 41.7pp (QNLI × MRPC) |
| Cross-format (sentence-pair × single-sentence) | 12 | 8.8pp | 12.7pp |

The intuition that structurally similar tasks should merge more safely is **wrong** in this regime. Same-format pairs are actually worse on average due to the QNLI × MRPC catastrophic interaction.

### 4. QNLI × MRPC is a catastrophic reference case

All 3 catastrophic pairs involve QNLI × MRPC. This pair:
- Looks structurally benign (pair-risk = medium, same format, medium strength gap)
- Is semantically related (both are sentence-pair NLI-adjacent tasks)
- But produces catastrophic functional interference (up to -42pp on one task)

This is the clearest single counterexample to "related tasks merge more safely." It should be treated as a reference case in all future cross-task discussions.

### 5. The current blind spot is severity grading inside cross-task

The task-relationship advisory correctly flags all 24 cross-task pairs. But it gives the same warning to:
- a mild pair (RTE × MRPC, -3pp)
- a catastrophic pair (QNLI × MRPC, -42pp)

No current Gradience signal distinguishes these. This is the real next blind spot.

## What this changes

### For the regime map
Cross-task is no longer one row. It is at least 4 severity subtypes, each with different practical implications.

### For the advisory
The advisory is doing its job — it flags cross-task pairs. But its value ceiling is now clear: it partitions same-task from cross-task, but does not grade severity within cross-task.

### For the project's research direction
The next useful question is no longer "are there safe related-task merges?" (answer: no, in this regime). It is:

> **What explains the severity gradient within cross-task pairs, and can a lightweight signal predict whether a cross-task merge will be mild or catastrophic?**

That is a stronger and more honest research line than looking for safe cross-task pairs.

## What this does NOT change

- Same-task safety: confirmed again (4 control pairs, 3/4 near-safe)
- Advisory value: still the primary same/different discriminator
- Source QA: still the anchor of the workflow
- Core-space status: still narrow and advanced

## Candidate explanatory variables for severity (for future work)

The data hints at but does not yet confirm:
1. **Task-specific functional incompatibility** — the QNLI × MRPC pattern suggests certain task pairs have fundamentally incompatible decision boundaries
2. **Source-strength asymmetry** — drives asymmetric dilution but does not explain catastrophic cases
3. **Label-space alignment** — QNLI (entailment/not) vs MRPC (paraphrase/not) may use the same classifier architecture to make fundamentally different discriminations

These are hypotheses for the next study, not conclusions from this one.

## Bottom line

The cross-task regime has real internal structure. The advisory catches the boundary but not the gradient. The next empirical question is what predicts severity, not whether subtypes exist.
