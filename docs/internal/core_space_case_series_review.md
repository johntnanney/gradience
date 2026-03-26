# Core-Space Case Series Review

Review of core-space usage across the real-inventory case series.

## How often was core-space used?

3 times across 3 inventories (1 per inventory). Always on a single justified pair, never on every pair.

## How often did it disagree with ordinary low-risk pair reports?

**2 of 2 target-class instances** (pairs where ordinary pair_risk=low and core-space was computed).

| Inventory | Pair | Ordinary risk | Core-space | Disagreement |
|-----------|------|---------------|------------|--------------|
| roberta_mixed_evidence_4 | uniform_elig × probe_uncertain | low | marginal (0.878) | Yes |
| distilbert_large_pool_6 | probe_r16 × qnli_probe | low | incompatible (0.867) | Yes — strong |

The first published case study (case_study_qnli4) used core-space on an "ambiguous" pair rather than a strictly low-risk one, so it is not a clean target-class instance. It also produced a judgment change.

## How often did it actually change the next action?

**3 of 3 uses** changed the practical next action:

1. case_study_qnli4: ambiguous pair moved to caution track
2. roberta_mixed_evidence_4: low-risk pair demoted to "proceed carefully"
3. distilbert_large_pool_6: low-risk pair removed from candidate set entirely

Core-space changed structural judgment every time it was used in the case series. This is partly a selection effect (pairs were chosen because they were ambiguous). However, verified adjudication (2026-03) showed that structural judgment changes do not always correspond to behavioral harm: same-task merges were safe even when flagged as incompatible. The diagnostic finds real structural signal, but that signal is not broadly behaviorally decisive.

## Is it still correctly tiered as advanced and non-default?

**Yes.**

Evidence:
- Used selectively (1 pair per inventory, not every pair)
- Only applied when the pair met at least 2 of the 5 selection criteria
- The diagnostic changes action but does not dominate the workflow — source QA and neighborhoods do the bulk of the narrowing
- Running core-space on every pair would be unnecessary: the high-risk and medium-risk pairs don't need deeper inspection, and obviously safe same-group pairs (qnli_probe × qnli_uniform) would likely confirm compatibility without adding new information

**Recommendation:** Keep core-space as advanced and non-default. The current selection criteria correctly target structurally ambiguous pairs. However, verified adjudication showed that "changed action" in the case series does not necessarily mean "prevented behavioral harm." Core-space remains appropriately classified as advanced, but its strongest supported role is narrower and more regime-dependent than earlier evidence suggested. Its decision value is likely concentrated in pairs where task relationship is genuinely ambiguous and ordinary pair-risk is not already decisive.

## Open question for future work

The 2/2 disagreement rate on low-risk pairs is striking. Is this because:

(a) the selection criteria correctly identify pairs where layer safety is misleading, or
(b) core-space disagrees with low-risk more often than expected, and the selection criteria are not yet filtering enough?

A future inventory with an unambiguously safe low-risk pair (e.g., same-task, both eligible, similar rank) could test whether core-space confirms safety in the easy case. The qnli_probe × qnli_uniform pair in distilbert_large_pool_6 would have been a natural control, but core-space was not run on it.
