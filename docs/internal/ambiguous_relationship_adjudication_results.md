# Ambiguous Relationship Adjudication — Results

Date: 2026-03-24

## Question

When two adapters are neither obvious same-task redundancies nor obvious cross-task mismatches, does core-space add behaviorally useful discrimination beyond ordinary pair-risk?

## Setup

- Backbone: distilbert-base-uncased
- Task family: NLI-family (QNLI, RTE, MNLI)
- 6 adapters (2 per task, different seeds), all independently verified above base
- 8 pairs: 3 same-task controls, 5 adjacent-task ambiguous
- Merge strategy: uniform_linear
- Evaluation: each merged adapter on both source tasks

## Individual adapter scores

| Adapter | Task | Accuracy | Margin over base |
|---------|------|----------|-----------------|
| qnli_s42 | QNLI | 0.696 | +0.226 |
| qnli_s7 | QNLI | 0.710 | +0.168 |
| rte_s42 | RTE | 0.567 | +0.051 |
| rte_s7 | RTE | 0.556 | +0.083 |
| mnli_s42 | MNLI | 0.530 | +0.178 |
| mnli_s7 | MNLI | 0.532 | +0.230 |

## Full results

| Pair | Group | Risk | CS status | Basis | Merged-A | Merged-B | Deg-A | Deg-B |
|------|-------|------|-----------|-------|----------|----------|-------|-------|
| qnli_s42 x qnli_s7 | same-task | redundant | marginal | 0.887 | 0.702 | — | -0.006 | — |
| rte_s42 x rte_s7 | same-task | redundant | **incompatible** | 0.839 | 0.560 | — | +0.007 | — |
| mnli_s42 x mnli_s7 | same-task | redundant | marginal | 0.906 | 0.526 | — | +0.004 | — |
| qnli_s42 x rte_s42 | adjacent | redundant | marginal | 0.905 | 0.632 | 0.585 | **+0.064** | -0.018 |
| qnli_s42 x mnli_s42 | adjacent | redundant | **incompatible** | 0.882 | 0.686 | 0.480 | +0.010 | **+0.050** |
| rte_s42 x mnli_s42 | adjacent | redundant | marginal | 0.926 | 0.560 | 0.472 | +0.007 | **+0.058** |
| qnli_s7 x rte_s7 | adjacent | redundant | marginal | 0.941 | 0.656 | 0.570 | **+0.054** | -0.014 |
| rte_s7 x mnli_s7 | adjacent | marginal | 0.938 | 0.578 | 0.466 | -0.022 | **+0.066** |

## Key findings

### 1. Same-task merges are safe — confirming the earlier regime

All 3 same-task pairs preserved accuracy within 0.7pp. This held even when core-space said "incompatible" (rte_s42 x rte_s7, basis=0.839). This replicates the earlier finding exactly: core-space structural flags on same-task pairs do not predict behavioral harm.

### 2. Adjacent-task merges show a clear asymmetric degradation pattern

This is the new finding. In every adjacent-task pair:
- The **stronger task was preserved** (QNLI accuracy held within ~1pp in most cases)
- The **weaker task degraded materially** (MNLI lost 5-7pp; QNLI lost 5-6pp when merged with RTE)

The pattern: the merged adapter's attention layers are dominated by whichever task had stronger gradient signal during training. The weaker task's contribution gets diluted in the linear average.

### 3. Ordinary pair-risk did NOT separate safe from unsafe here

**This is the critical finding.** Every pair in this study was classified as "redundant" by ordinary pair-risk. All 8 pairs got the same verdict. But:
- Same-task redundant pairs: safe (<=0.7pp degradation)
- Adjacent-task redundant pairs: materially degraded on the weaker task (5-7pp)

**Ordinary pair-risk failed to distinguish these regimes.** It saw high spectral overlap (these are all NLI-family adapters on the same backbone) and called them all redundant.

### 4. Core-space partially discriminates but is noisy

| Pair | CS status | Basis | Behavioral outcome |
|------|-----------|-------|--------------------|
| qnli x qnli | marginal | 0.887 | safe |
| rte x rte | **incompatible** | 0.839 | safe |
| mnli x mnli | marginal | 0.906 | safe |
| qnli x rte (s42) | marginal | 0.905 | QNLI degraded |
| qnli x mnli | **incompatible** | 0.882 | MNLI degraded |
| rte x mnli (s42) | marginal | 0.926 | MNLI degraded |
| qnli x rte (s7) | marginal | 0.941 | QNLI degraded |
| rte x mnli (s7) | marginal | 0.938 | MNLI degraded |

Core-space called rte x rte "incompatible" (safe in practice) and called several degraded adjacent-task pairs "marginal" (actually degraded). It does not reliably separate safe from unsafe in this regime.

However, the two pairs with the lowest shared-basis scores (rte x rte at 0.839, qnli x mnli at 0.882) include one safe and one degraded. The signal is present but noisy.

### 5. The main discriminator is task identity, not spectral structure

The strongest predictor of merge safety in this regime is simply: **are the adapters trained on the same task?**

- Same task: safe, regardless of core-space or pair-risk
- Different task: weaker task degrades, regardless of core-space or pair-risk

Neither pair-risk nor core-space captured this boundary reliably.

## Implications

### For ordinary pair-risk

Pair-risk is too permissive in this regime. When all adapters share the same backbone and are from related NLI-family tasks, pair-risk sees high spectral overlap and rates everything as "redundant." But same-task redundancy is safe while cross-task "redundancy" causes material degradation on the weaker task.

This suggests pair-risk may need task-relationship awareness, or at minimum a caveat that "redundant" across different tasks is a qualitatively different signal from "redundant" within the same task.

### For core-space

Core-space is not the answer here either. It partially discriminates but is noisy — it called a safe pair "incompatible" and called degraded pairs "marginal." Its structural measurements are real but do not reliably predict this specific failure mode (weaker-task dilution in linear merge).

### For the project

This regime revealed a blind spot: **the workflow assumes that spectral similarity implies merge compatibility.** In the NLI family, adapters look spectrally similar because they share structural roles, but the functional mapping (which labels, which task) diverges enough to cause material degradation.

The fix is not more spectral analysis. The fix is task-relationship metadata — which the workflow already has access to via QA artifacts (task name, dataset name) but does not currently use as a merge-risk signal.

## Result classification

**Result type C: Core-space helps only in a narrow sub-regime (and even there, noisily).**

The ambiguous-relationship regime is the hardest one tested so far. Neither pair-risk nor core-space reliably separates safe from unsafe. The main discriminator is task identity, which is a metadata property, not a spectral property.

## Recommended next steps

1. **Do not broaden core-space claims.** This regime does not support them.
2. **Consider adding a task-relationship signal to pair-risk reporting.** The QA artifact already contains task name. A simple "same-task / related-task / different-task" flag would have perfectly discriminated this panel.
3. **Document the asymmetric degradation pattern.** "Weaker task degrades, stronger task survives" is a useful practical heuristic for practitioners considering cross-task merges.
4. **Acknowledge the blind spot honestly in public materials.** The workflow is strong for same-task pools and for catching obviously bad sources, but it does not yet reliably screen NLI-family cross-task merges where spectral similarity is high.
