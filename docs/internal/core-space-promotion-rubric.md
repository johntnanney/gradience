# Core-Space Promotion Rubric (Internal)

## Purpose

Define a repeatable decision rule for core-space feature status after running:

```bash
python3 scripts/prepare_core_space_realistic_fixtures.py
python3 scripts/run_core_space_benchmark.py --require-realistic-fixtures
```

Allowed decisions:
- `retain_optional`
- `promote_advanced`
- `shelve`

This rubric is diagnostic-first and additive. It does not modify default merge recommendations.

## Required Fixture Categories

The benchmark must include these categories:
- balanced same-domain
- balanced cross-domain
- moderate-risk pairs
- same-rank semantically distant pairs
- deliberately mismatched/random pairs

## Criteria

The script evaluates five criteria:

1. `same_domain_compatible`
2. `distant_and_mismatch_downgraded`
3. `ambiguous_band_behavior`
4. `status_spread_nontrivial`
5. `runtime_within_budget`

Definitions:
- `same_domain_compatible`: same-domain fixture dominant status is `compatible`.
- `distant_and_mismatch_downgraded`: distant + mismatched fixtures are both downgraded (`marginal`/`incompatible`), with at least one `incompatible`.
- `ambiguous_band_behavior`: cross-domain remains in `{compatible, marginal}` and moderate-risk remains in `{marginal, incompatible}`.
- `status_spread_nontrivial`: at least two dominant statuses appear across fixtures.
- `runtime_within_budget`: max fixture mean runtime per layer is within benchmark budget.

## Decision Logic

`shelve` when:
- `same_domain_compatible` fails, or
- `distant_and_mismatch_downgraded` fails, or
- `status_spread_nontrivial` fails.

`promote_advanced` only when:
- all five criteria pass, and
- at least two fixtures are marked `evidence_tier = realistic`.

Otherwise:
- `retain_optional`.

## Current Guidance

Synthetic-only passes are useful for calibration and regression protection, but they are not sufficient for promotion to advanced status.

## Post-Realism Pass (2026-03-17)

Realistic fixtures added:
- `realistic_ambiguous_pair`
- `realistic_semantic_mismatch`

Observed behavior (run: `results/core_space_benchmark/20260317_165152/`):
- `realistic_ambiguous_pair`: dominant status `marginal`
- `realistic_semantic_mismatch`: dominant status `incompatible`
- all baseline synthetic categories retained expected directional behavior
- runtime remained within budget

Assessment:
- realistic fixtures were consistent with the intended diagnostic semantics
- the ambiguity fixture stayed in an intermediate band
- the superficially plausible mismatch fixture was downgraded

Final near-term decision:
- **`promote_advanced`**

Operational meaning:
- core-space remains optional and diagnostic-only
- no change to default merge recommendation logic
- suitable for advanced workflow documentation, not default onboarding flow

## Post-adjudication update (2026-03)

Verified adjudication on freshly trained adapters showed core-space is structurally informative but its behavioral decision value is narrower and more regime-dependent than the promotion-era evidence suggested. Same-task merges were safe even when flagged as incompatible. Further promotion beyond the current advanced-optional tier requires evidence that core-space changes behavioral outcomes in a regime where ordinary pair-risk is permissive. See `docs/internal/verified_adjudication_implications.md`.
