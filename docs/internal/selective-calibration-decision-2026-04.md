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

- Decision date: `<YYYY-MM-DD>`
- Author(s): `<name(s)>`
- Related corpus review memo: `docs/internal/corpus-review-memo-2026-04.md`
- Decision id: `<calibration-202604-XX>`

## Decision Context

- Triggering observation: `<1-2 sentences>`
- Scope of decision: `<single behavior to keep/change>`
- Non-goals:
  - no feature expansion
  - no default workflow redesign
  - no unrelated threshold changes

## Change Under Consideration

- component: `<module/component>`
- current behavior: `<explicit before>`
- observed issue: `<what appears miscalibrated>`
- proposed change: `<one small change>`
- why not broader changes: `<why scope is intentionally constrained>`

## Evidence Summary

### Aggregate evidence

- Corpus inventory count used: `<int>`
- Relevant pair/report count: `<int>`
- Relevant neighborhood/core-space slice: `<int + short note>`

### Low-risk/core-space mismatch evidence

- mismatch population count: `<int>`
- mismatch share among low-risk pairs: `<pct>`
- repeatability across inventories: `<yes/no + note>`

### Representative examples

1. `<example 1 path/run_id + 1 sentence>`
2. `<example 2 path/run_id + 1 sentence>`
3. `<example 3 path/run_id + 1 sentence>`

### Counter-evidence / uncertainty

- `<uncertainty 1>`
- `<uncertainty 2>`

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

Selected option: `<no_change | targeted_calibration | defer>`

Rationale: `<4-8 sentences>`

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

- Next evidence checkpoint date: `<YYYY-MM-DD>`
- Required additional evidence:
  - `<evidence requirement 1>`
  - `<evidence requirement 2>`

## Approval

- Reviewer(s): `<name(s)>`
- Status: `<approved | approved with caveats | rejected>`
- Follow-up item(s):
  - `<task 1>`
  - `<task 2>`
