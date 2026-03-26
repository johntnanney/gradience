# Selective Calibration Decision — 2026-03

Cycle metadata:
- cycle: `Corpus Review Cycle 01`
- freeze status: `active`
- approved scope: `at most one small change this cycle`
- default decision bias: `no_change unless corpus evidence is specific and consistent`

## Decision Options (Single Select)

- `no_change`
- `targeted_calibration`
- `defer`

## Metadata

- Decision date: `2026-03-17`
- Author(s): `codex`
- Related corpus review memo: `docs/internal/corpus-review-memo-2026-03.md`
- Decision id: `calibration-202603-01`

## Decision Context

- Triggering observation: first real-inventory corpus slice (3 inventories) is available with reproducible manifests and aggregate counters.
- Scope of decision: determine whether any one small calibration is justified now.
- Non-goals:
  - no feature expansion
  - no default workflow redesign
  - no unrelated threshold changes

## Change Under Consideration

- component: `corpus manifest adapter naming/aggregation semantics`
- current behavior: `adapter_names` are derived from QA `adapter_name` and deduplicated.
- observed issue: inventories built from checkpoint directories can collapse multiple adapters into one name (`checkpoint-50`), reducing manifest-level adapter counts.
- proposed change: `none in cycle-01` (documented caveat only).
- why not broader changes: this is a data-shape caveat, not yet a policy miscalibration affecting merge behavior.

## Evidence Summary

### Aggregate evidence

- Corpus inventory count used: `3`
- Relevant pair/report count: `9`
- Relevant neighborhood/core-space slice: `9` neighborhood reports/sections, `3` pair reports with `core_space`

### Representative examples

1. `study17_cache_triplet_20260317`: all pair reports medium risk with high redundancy and strict block candidates.
2. `core_space_real_adapter_triplet_20260317`: all pair reports low risk; core-space on one ambiguous pair returned `incompatible`.
3. `canonical_test2_triplet_20260317`: mixed pair risk (`2 low`, `1 high`) with one subspace-conflict case.

### Counter-evidence / uncertainty

- sample size is still small and skewed toward structurally clean internal adapters.
- all strict blocks are driven by missing behavioral evaluation, limiting policy-sensitivity analysis.

## Candidate Actions

### Option A — `no_change`

- Description: keep current behavior for next cycle.
- Benefit: preserves comparability and stability.
- Cost: known rough edge remains.

### Option B — `targeted_calibration`

- Description: one narrow change only.
- Proposed change location: `append/summarize corpus adapter-name handling`.
- Expected behavior shift: improved adapter counting and per-inventory accounting in corpus summaries.
- Blast radius: low.

### Option C — `defer`

- Description: postpone decision until more corpus evidence.
- Benefit: avoids premature tuning.
- Cost: delays improvement.

## Decision

Selected option: `no_change`

Rationale: no calibration candidate is yet both urgent and evidence-complete. The observed adapter-name collapse is valid but currently bookkeeping-level and does not justify changing policy or recommendation behavior in this cycle. Core-space disagreement signals are interesting but still early; neighborhood conservatism is stable but not yet a clear bug. Preserve current behavior and gather more inventories before any targeted change.

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

- Next evidence checkpoint date: `2026-03-24`
- Required additional evidence:
  - at least `2` additional real inventories with behavioral evaluation coverage
  - at least `1` inventory where neighborhoods produce a non-singleton high-compatibility group

## Approval

- Reviewer(s): `<name(s)>`
- Status: `approved`
- Follow-up item(s):
  - continue cycle-01 collection beyond minimum target if clean inventories are available
  - revisit adapter-name accounting caveat in next cycle if repeated
