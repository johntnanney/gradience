# Selective Calibration Decision — 2026-05

Cycle metadata:
- cycle: `Corpus Review Cycle 03`
- freeze status: `active`
- approved scope: `at most one small change this cycle`
- default decision bias: `no_change unless corpus evidence is specific and consistent`

## Decision Options (Single Select)

- `no_change`
- `targeted_calibration`
- `defer`

## Required Gate Alignment

Decision is valid only if Cycle-03 gate coverage is explicit:

1. at least one inventory likely to produce non-singleton neighborhoods
2. at least one inventory with mixed behavioral evidence (strict-QA middle-case)
3. explicit low-risk/core-space mismatch tracking completed
4. corpus identity hardening follow-through so adapter-instance counting is trustworthy across runs

## Gate Status (Pass/Fail)

| Gate | Status | Evidence |
|---|---|---|
| Non-singleton neighborhood target | `PASS` | Non-singleton neighborhoods observed in all 4 inventories. |
| Mixed behavioral-evidence strict-QA middle case | `PASS` | Mixed-status inventories included and produced both block and non-block outcomes. |
| Low-risk/core-space mismatch tracking completed | `PASS` | 4 mismatches captured among 8 low-risk pairs. |
| Identity hardening follow-through | `PASS` | Implemented post-cycle as scoped infrastructure hardening in `scripts/summarize_corpus.py`; counting now uses deterministic identity-safe instance keys. |

## Metadata

- Decision date: `2026-03-17`
- Author(s): `Codex`
- Related corpus review memo: `docs/internal/corpus-review-memo-2026-05.md`
- Decision id: `calibration-202605-01`

## Decision Context

- Triggering observation: Cycle-03 produced stable distributional behavior plus recurring low-risk/core-space mismatch cases. The identity-hardening blocker identified during review has since been resolved via a scoped post-cycle infrastructure patch.
- Scope of decision: Keep current policy behavior unchanged; record the post-cycle hardening closure without reclassifying it as calibration.
- Non-goals:
  - no feature expansion
  - no default workflow redesign
  - no unrelated threshold changes

## Change Under Consideration

- component: `scripts/summarize_corpus.py` (corpus accounting layer)
- current behavior: adapter-instance counting now resolves deterministic identity keys and dedupes unique instances across manifests.
- observed issue: historical cycle snapshots captured before hardening may show non-comparable adapter-instance totals.
- proposed change: no policy calibration this cycle; identity hardening patch already applied as behaviorally neutral infrastructure correction.
- why not broader changes: default preflight and advanced workflows are stable; evidence does not justify broader calibration.

## Evidence Summary

### Aggregate evidence

- Corpus inventory count used: `4`
- Relevant pair/report count: `12`
- Relevant neighborhood/core-space slice: `4` core-space pair reports and `5` neighborhood groups across cycle-03 set.

### Strict-QA middle-case evidence

- mixed-evidence inventory count: `3`
- block/non-block split in mixed-evidence slice: `6/3`
- interpretation: Mixed-evidence inventories consistently produced both strict-block and non-block outcomes. This indicates middle-case discrimination is active and stable, not collapsed into a one-sided behavior. No strict-QA semantic change is warranted from this slice.

### Low-risk/core-space mismatch evidence

- mismatch population count: `4`
- mismatch share among low-risk pairs: `50.0%` (`4/8`)
- repeatability across inventories: `yes` (present in all cycle-03 inventories)

### Neighborhood diversity evidence

- non-singleton neighborhoods observed: `yes`
- if yes, count and run ids: observed in all 4 runs: `cycle03_qnli_all_eligible_triplet_20260317`, `cycle03_qnli_mixed_behavior_triplet_20260317`, `cycle03_roberta_mixed_evidence_triplet_20260317`, `cycle03_real_adapter_triplet_20260317`

### Identity hardening follow-through

- status: `implemented (post-cycle patch)`
- effect on adapter-instance counting trust: Corpus adapter-instance totals now use identity-safe dedupe semantics and no longer depend on display-name uniqueness. This clears the metadata trustworthiness blocker while leaving policy behavior unchanged.

### Representative examples

1. `cycle03_roberta_mixed_evidence_triplet_20260317`: strict-QA produced a balanced 2 blocked / 1 non-block pattern.
2. `results/real_inventory_runs/20260317/cycle03_real_adapter_triplet/reports/final_vs_qnli_core_space.json`: low-risk pair with `core_space.status=marginal`.
3. `cycle03_qnli_all_eligible_triplet_20260317`: non-singleton neighborhood with no exclusions, showing stable conservative grouping.

### Counter-evidence / uncertainty

- Cycle-03 corpus size remains moderate (`12` pairs), so calibration confidence is still limited.
- Adapter-instance totals before and after hardening are not directly comparable without noting the counting-semantics change.

## Candidate Actions

### Option A — `no_change`

- Description: keep current behavior for next cycle.
- Benefit: preserves comparability and stability.
- Cost: known rough edge remains.

### Option B — `targeted_calibration`

- Description: one narrow change only.
- Proposed change location: `scripts/summarize_corpus.py`
- Expected behavior shift: adapter-instance counters become identity-safe in name-collision cases; no policy behavior changes.
- Blast radius: `low` (reporting-only layer, no merge recommendation path impact)

### Option C — `defer`

- Description: postpone decision until more corpus evidence.
- Benefit: avoids premature tuning.
- Cost: delays improvement.

## Decision

Selected option: `no_change`

Rationale: Cycle-03 met diversity and mismatch-tracking goals, and system behavior remained coherent under freeze constraints. Strict-QA middle-case behavior was stable, and neighborhood outputs remained conservative and useful across all inventories. Low-risk/core-space mismatches are recurring and worth continued advanced-tier tracking, but this signal alone does not justify recommendation-path calibration. The post-cycle identity-hardening patch resolved a corpus trustworthiness issue, but it is infrastructure correction rather than policy calibration. The cycle decision remains `no_change`.

## Validation Criteria

1. Targeted unit/integration tests pass.
2. Affected fixture harnesses still pass.
3. Corpus summary counters remain coherent after change.
4. No regressions in core preflight outputs.
5. No unintended changes to strict-QA semantics.

## Rollback Criteria

Rollback immediately if any of:

- strict block behavior regresses
- neighborhood/core-space signal collapses or becomes noisy
- core workflow outputs change unexpectedly
- recommendation behavior drifts outside approved scope

## If `targeted_calibration` Is Selected

### Exact change specification

- Current behavior: not selected this cycle
- New behavior: not selected this cycle
- Files to update:
  - n/a

### Invariants to preserve

- Keep strict-QA semantics unchanged unless explicitly in scope.
- Keep stable v1 artifact contracts unchanged.
- Keep default recommendation logic unchanged unless explicitly in scope.

### Validation plan

1. Run targeted unit/integration tests.
2. Re-run affected fixture harness(es).
3. Re-run corpus summary and compare key counters.
4. Verify no regressions in core preflight paths.

## If `no_change` or `defer` Is Selected

- Next evidence checkpoint date: `2026-05-31`
- Required additional evidence:
  - maintain identity-safe counting semantics across future corpus summaries
  - add at least 3 additional diverse inventories and re-check mismatch recurrence plus strict-block rate trend

## Approval

- Reviewer(s): pending investigator review
- Status: `approved with caveats`
- Follow-up item(s):
  - annotate pre-hardening vs post-hardening adapter-instance totals when comparing historical cycle snapshots
  - continue mismatch tracking in next cycle without default-policy changes
