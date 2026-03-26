# Core-Space Realism Pass Decision

## What was added

- two non-synthetic benchmark fixtures:
  - `realistic_ambiguous_pair`
  - `realistic_semantic_mismatch`
- harness support for real adapter pair fixtures (`generator.mode = real_adapter_pair`)
- updated benchmark docs and harness tests

## What happened

Run:
- `python3 scripts/prepare_core_space_realistic_fixtures.py`
- `python3 scripts/run_core_space_benchmark.py --require-realistic-fixtures`
- output: `results/core_space_benchmark/20260317_165152/`

Observed:
- realistic ambiguous pair resolved to dominant `marginal`
- realistic semantic mismatch resolved to dominant `incompatible`
- existing synthetic directional checks remained stable
- runtime stayed within budget

## Final decision

- **`promote_advanced`**

## Operational meaning

- core-space is now classified as an advanced optional diagnostic
- default merge recommendations remain unchanged
- no public API promotion in this pass
- further promotion beyond advanced usage should depend on corpus-backed real usage, not additional synthetic benchmark crafting

## Post-adjudication note (2026-03)

Verified adjudication showed core-space's behavioral decision value is narrower than the realism-pass evidence alone suggested. The `promote_advanced` decision stands, but further promotion requires evidence of behavioral impact in a regime where ordinary pair-risk is permissive. See `docs/internal/verified_adjudication_implications.md`.
