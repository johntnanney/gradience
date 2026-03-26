# Selective Calibration Decision Memo — Template

## Metadata

- Decision date: `<YYYY-MM-DD>`
- Author(s): `<name(s)>`
- Related corpus review memo: `<path>`
- Decision id: `<calibration-YYYYMMDD-XX>`

## Decision Context

- Triggering observation: `<1-2 sentences>`
- Scope of decision: `<single behavior to keep/change>`
- Non-goals:
  - no feature expansion
  - no default workflow redesign
  - no unrelated threshold changes

## Evidence Summary

### Aggregate evidence

- Corpus inventory count used: `<int>`
- Relevant pair/report count: `<int>`
- Relevant neighborhood/core-space slice: `<int + short note>`

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

### Rollback trigger

- Roll back if any of:
  - strict block behavior regresses
  - neighborhood/core-space signal collapses or becomes noisy
  - core workflow outputs change unexpectedly

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
