# Core-Space Feature Status

## Purpose

Optional pairwise structural diagnostic for shared-basis compatibility in `merge-audit`.

## Current status

- classification: **advanced optional diagnostic**
- CLI exposure: `merge-audit --compute-core-space`
- schema surface: optional `core_space` block in `gradience.merge_qa_report/v1`
- public API export: **not promoted**

## Realism pass status

- realism benchmark pass completed (`scripts/run_core_space_benchmark.py`)
- realistic fixtures were added and evaluated alongside baseline synthetic categories
- realistic fixture inputs are prepared via `scripts/prepare_core_space_realistic_fixtures.py`
- rubric decision: `promote_advanced`

## Verified adjudication update (2026-03)

A verified adjudication study on freshly trained DistilBERT adapters (3 SST-2 + 3 QNLI, all independently verified above base) produced regime-bound results:

- **Same-task merges were safe even when core-space said "incompatible."** All 6 same-task pairs preserved accuracy within ~1.2pp of best individual. Core-space overwarned.
- **Cross-task merges degraded substantially (~8-18pp).** But ordinary pair-risk already separated these from same-task safe pairs.
- **Core-space added only modest additional discrimination** inside the already-unsafe cross-task group.

**Updated position:** Core-space remains a real advanced structural diagnostic, but verified adjudication shows its behaviorally useful role is narrower and more regime-dependent than previously assumed. It is most worth using when task relationship is genuinely ambiguous and ordinary pair-risk is not already decisive.

See: `docs/internal/verified_adjudication_implications.md`

## Promotion gate (evidence required)

Further promotion beyond current advanced-optional tier requires:

1. evidence that core-space changes behavioral outcomes (not just structural judgments) in a regime where ordinary pair-risk is permissive.
2. status behavior remains stable across releases (no collapse back to a single bucket).
3. verified adjudication in harder intermediate regimes (related-but-not-identical tasks, domain shift, style variants).
4. runtime and UX remain acceptable in normal preflight usage.

Use:
- `examples/core_space_benchmark/`
- `scripts/run_core_space_benchmark.py`
- `docs/internal/core-space-promotion-rubric.md`

## Near-term guidance

- keep optional in execution semantics
- keep documented as an advanced workflow diagnostic
- keep out of core/default workflow claims — verified adjudication shows it is not broadly behaviorally decisive in the tested regime
- describe as regime-dependent, structurally informative, and narrower than earlier case-series evidence suggested
- next useful evidence: adjudication in ambiguous-relationship regimes where pair-risk is permissive and task mismatch is not already doing the main explanatory work
