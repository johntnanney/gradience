# Core-Space Audit Study (CPU-Only, Internal)

## Scope

This note evaluates whether the optional core-space diagnostic adds explanatory value beyond existing pairwise signals.

Status: internal, diagnostic-only.

## Fixed pair set (synthetic structural archetypes)

We evaluated a fixed archetype set aligned to practical screening goals:

1. `self` (identical updates)
2. `near` (small perturbation around same update)
3. `imbalanced_related` (related direction, lower-magnitude second source)
4. `random_other` (intentionally dissimilar source)

Each archetype used 8 synthetic matched layers (`r=8`, CPU, float64 path), then aggregated with `aggregate_core_space_diagnostics(...)`.

## Results

| Archetype | Shared basis score | Basis distortion | Effective shared rank (median) | Status |
|---|---:|---:|---:|---|
| self | 0.991 | 0.000 | 7 | compatible |
| near | 0.991 | 0.001 | 7 | compatible |
| imbalanced_related | 0.991 | 0.000 | 7 | compatible |
| random_other | 0.810 | 0.016 | 12 | incompatible |

## CPU runtime check

Micro-benchmark (`32` synthetic layers, `r=8`, `d_in=d_out=128`, CPU float64):

- total: `~0.011 s`
- per layer: `~0.354 ms`

## Interpretation

### 1) Agreement with current pair risk where expected

Partial agreement. Redundancy-like and same-domain-like archetypes show high shared-basis scores as expected.

### 2) Does it sharpen ambiguous pairs?

Yes, after calibration changes that penalize shared-rank inflation. The fixed archetype set now produces non-trivial status spread instead of collapsing to `compatible`.

### 3) Information beyond overlap/conflict?

Useful incremental signal:
- explicit shared-rank estimate
- explicit distortion penalty vs separate bases
- status downgrade when shared basis requires substantially higher rank than either source

### 4) CPU runtime acceptability

Yes. Runtime overhead is small in synthetic CPU tests.

## Recommendation

Keep feature as **optional diagnostic-only** for now.

Calibration is improved, but promotion into default recommendation logic should still wait for broader real-adapter evidence.
