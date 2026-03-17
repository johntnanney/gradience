# Selective Calibration Decision — 2026-04

Cycle metadata:
- cycle: `Corpus Review Cycle 02`
- freeze status: `active`
- approved scope: `at most one small change this cycle`
- default decision bias: `no_change unless corpus evidence is specific and consistent`

## Decision Options (Single Select)

- `no_change`
- `targeted_calibration`
- `defer`

## Metadata

- Decision date: `2026-03-17`
- Author(s): `Codex + investigator`
- Related corpus review memo: `docs/internal/corpus-review-memo-2026-04.md`
- Decision id: `calibration-202604-01`

## Decision Context

- Triggering observation: `Cycle-02 completed 3 new real inventories and surfaced one low-risk/core-space mismatch case, without repeated strict-policy pressure.`
- Scope of decision: `whether to keep behavior unchanged vs apply one narrow calibration`
- Non-goals:
  - no feature expansion
  - no default workflow redesign
  - no unrelated threshold changes

## Change Under Consideration

- component: `core-space interpretation policy (review layer only)`
- current behavior: `core-space remains optional advanced diagnostic with no default recommendation effect`
- observed issue: `single low-risk/core-space-incompatible mismatch observed`
- proposed change: `none this cycle`
- why not broader changes: `evidence is specific but not yet consistent/repeated across inventories`

## Evidence Summary

### Aggregate evidence

- Corpus inventory count used: `3` (cycle-02 slice)
- Relevant pair/report count: `12`
- Relevant neighborhood/core-space slice: `3 core-space pairs; 10 groups; 12 boundary warnings`

### Low-risk/core-space mismatch evidence

- mismatch population count: `1`
- mismatch share among low-risk pairs: `20.0%` (1/5)
- repeatability across inventories: `no` (single observed case in qnli triplet)

### Representative examples

1. `cycle02_qnli_triplet_20260317`: low-risk pair with `core_space.status=incompatible`.
2. `cycle02_roberta_sst2_triplet_20260317`: mixed high/medium/low risk with one core-space marginal case.
3. `cycle02_final_test_quartet_20260317`: larger inventory with high-risk and high-redundancy coverage.

### Counter-evidence / uncertainty

- Core-space disagreement is currently sparse, not a repeated pattern.
- No non-singleton neighborhoods appeared in this cycle, limiting calibration confidence for grouping behavior.

## Candidate Actions

### Option A — `no_change`

- Description: keep current behavior for next cycle.
- Benefit: preserves comparability and stability.
- Cost: known rough edge remains.

### Option B — `targeted_calibration`

- Description: one narrow change only.
- Proposed change location: `<file/module/flag>`
- Expected behavior shift: `<1-3 sentences>`
- Blast radius: `<low/medium/high with reason>`

### Option C — `defer`

- Description: postpone decision until more corpus evidence.
- Benefit: avoids premature tuning.
- Cost: delays improvement.

## Decision

Selected option: `no_change`

Rationale: `Cycle-02 produced coherent aggregate behavior and one meaningful advanced diagnostic mismatch, but not a repeated, specific issue that warrants even a narrow calibration yet. Strict-QA block pressure was absent in this slice, and neighborhood behavior remained stable under conservative grouping rules. Changing thresholds or logic now would reduce comparability without enough gain. Continue frozen behavior and collect additional diverse inventories for a stronger signal.`

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

- Current behavior: `<explicit before>`
- New behavior: `<explicit after>`
- Files to update:
  - `<path_1>`
  - `<path_2>`

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

- Next evidence checkpoint date: `2026-04-15`
- Required additional evidence:
  - at least `1-2` additional diverse inventories, including one likely non-singleton neighborhood case
  - at least one repeated low-risk/core-space mismatch pattern before any calibration proposal

## Approval

- Reviewer(s): `pending investigator confirmation`
- Status: `approved with caveats`
- Follow-up item(s):
  - continue cycle-02 collection toward 4–5 inventories
  - re-check decision after next evidence checkpoint
