# Core-Space Benchmark Fixtures

This directory contains fixed, repeatable fixtures for the internal
core-space benchmark harness:

```bash
python3 scripts/run_core_space_benchmark.py
```

Each fixture is intentionally diagnostic-first and synthetic in this phase.
They exist to stress status behavior across archetypes, not to claim
production merge policy changes.

Fixture set:
- `balanced_same_domain`
- `balanced_cross_domain`
- `moderate_risk_pairs`
- `same_rank_semantically_distant`
- `deliberately_mismatched_random`
- `realistic_ambiguous_pair`
- `realistic_semantic_mismatch`

## Realistic fixtures

- `realistic_ambiguous_pair`
  - Real DistilBERT adapters from different benchmark pools.
  - Why realistic: a practitioner could plausibly consider this merge; it is not random and not an obvious scale-only mismatch.
  - What it tests: whether core-space gives a useful ambiguity signal on a plausible, non-trivial pair.
  - Expected qualitative behavior: usually `marginal` (sometimes `compatible`), not uniformly `incompatible`.

- `realistic_semantic_mismatch`
  - Real DistilBERT `r=16` adapters from different benchmark tracks.
  - Why realistic: superficially mergeable (same base family and same nominal rank) but behaviorally distant enough to challenge shared-basis assumptions.
  - What it tests: whether core-space can downgrade a “looks mergeable” pair.
  - Expected qualitative behavior: `incompatible` or strong `marginal` tendency.

## Preparing realistic fixture inputs

Realistic fixtures load adapters from:

- `results/core_space_benchmark/real_adapters/final_uniform_median_r16`
- `results/core_space_benchmark/real_adapters/qnli_per_layer_r8`
- `results/core_space_benchmark/real_adapters/priority_probe_r16`

Prepare them deterministically from local benchmark outputs:

```bash
python3 scripts/prepare_core_space_realistic_fixtures.py
```

The prep script copies minimal adapter files (`adapter_config.*`, `adapter_model.*`) and writes a manifest under:

- `results/core_space_benchmark/real_adapters/manifest.json`

## Availability behavior

By default, `run_core_space_benchmark.py` emits explicit `unavailable` rows for realistic fixtures whose prepared inputs are missing and continues.

To fail fast (recommended for promotion/CI checks), run:

```bash
python3 scripts/run_core_space_benchmark.py --require-realistic-fixtures
```
