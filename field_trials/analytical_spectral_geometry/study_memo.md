# Analytical Spectral Geometry of Merge Operations — Study Memo

**Date**: 2026-04-03
**Outcome**: Negative completion (informative)
**Study spec**: `docs/plans/ANALYTICAL_SPECTRAL_GEOMETRY_OF_MERGE_OPERATIONS_SPEC`

## Executive Summary

We derived closed-form and semi-analytical results for how linear merge
operations transform singular value spectra, expressed in quantities
Gradience already computes. The analytical condition for over-accumulation
is mathematically correct and numerically validated on synthetic matrices.
However, when cross-checked against 30 field-trial adapter pairs with
actual merge outcomes, the analytical condition — like both existing
heuristics (OA-v1 and OA-v2) — fails to predict merge quality. The
Spearman correlation has the wrong sign: higher predicted spectral
inflation correlates with *better* merges, not worse ones.

The root cause is that spectral inflation of aligned features is not
harmful for same-task merges. The dominant failure mode in this cohort
is cross-task interference, a behavioral property not observable in
the spectrum.

This is the study spec's **negative completion condition**: the
quantities Gradience currently computes are not sufficient to predict
merge quality from spectral metrics alone. Task relationship information
is required.

## What Was Derived (Phases 1-2)

### Phase 1: Linear Merge Spectral Analysis

All results implemented in `gradience/vnext/merge/spectral_theory.py`
and validated in `tests/merge/test_spectral_theory.py` and
`tests/merge/test_spectral_theory_validation.py`.

**Rank-1 exact solution** (pre-existing, verified):

For `ΔW_m = αΔW_a + βΔW_b` with rank-1 factors, the merged leading
singular value satisfies the characteristic equation `λ² - Tλ + D = 0`
where `T = a² + b² + 2abz`, `D = a²b²sin²θ sin²φ`, `z = sign(δ)cosθ cosφ`.

**Equal-rank equal-spectrum exact solution** (new):

When `Σ_a = Σ_b` and principal angle decomposition aligns with spectral
decomposition, the Gram matrix decomposes into `r` independent 2×2 blocks.
Each block's eigenvalues are `λ_{i,±} = (T_i ± √(T_i² - 4D_i)) / 2`.
Validated against dense SVD across 12 parameter settings (max error < 0.15).

**Frobenius norm of merge** (new, exact):

`||αA + βB||²_F = α²||A||²_F + β²||B||²_F + 2αβ · sign(δ) · Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)`

Validated across 24 rank-r parameter settings (max error < 0.05).

**General rank-r Gram decomposition** (new):

Cross-term spectral norm bounded by
`σ_1(cross) ≤ 2αβ Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)`.
Weyl bounds validated: 100% containment across 24 parameter settings.

**Stable rank bounds** (new):

`sr_lower = ||ΔW_m||²_F / σ_1_upper²`,
`sr_upper = ||ΔW_m||²_F / σ_1_lower²`.

### Phase 2: Over-Accumulation Theory

**Exact rank-r over-accumulation condition** (new):

Over-accumulation occurs when `σ_1(ΔW_m) > max(α·σ_1(ΔW_a), β·σ_1(ΔW_b))`.
The necessary condition is:

1. Positive directional agreement (`sign(δ) > 0`)
2. Non-zero principal angle overlap in both left AND right subspaces
3. Cross-term magnitude sufficient to push σ_1 above weighted baseline

The cross-term is:

```
cross = 2αβ · Σ_i s_{a,i} · s_{b,i} · cos(θ_i) · cos(φ_i) · sign(δ)
```

**Why the heuristic is structurally wrong** (new finding):

The OA-v1 heuristic `alignment × (0.7·concentration + 0.3·coeff_exposure)`
misses the actual cross-term in three ways:

1. Uses only left-space principal angles (`cos(θ_i)` via `mean_overlap`).
   The actual cross-term requires the product `cos(θ_i) · cos(φ_i)`.
   Two adapters can have high left-space overlap but orthogonal row spaces,
   producing zero inflation. The heuristic flags this as risky.

2. Uses `1 - stable_rank/effective_rank` as concentration proxy. The
   actual cross-term weights each direction by `s_{a,i} · s_{b,i}`.
   These are related but not equivalent.

3. Reduces the full principal angle spectrum to two scalars
   (`mean_overlap`, `max_overlap`). The cross-term is a sum over `k`
   directions, each weighted differently.

**Norm-equalized analysis** (new):

Norm equalization reduces the cross-term when `ρ_F >> 1` by compressing
the dominant adapter's singular values. The reduction factor is
`normeq_ratio / linear_ratio`. Empirically, norm equalization reduced
inflation in only 3.3% of layers across 30 pairs — because most pairs
in this cohort are already approximately balanced.

### Synthetic Matrix Utilities (new)

`spectral_theory_test_utils.py` now provides:

- `make_rankr_pair_with_geometry()`: rank-r pairs with prescribed spectra
  and principal angles (both left and right)
- `make_equal_spectrum_pair()`: equal-spectrum variant
- `frobenius()`, `stable_rank()`, `singular_values()` dense helpers

## Empirical Cross-Check Results

### Setup

30 strict-naive pairs from `field_trials/over_accumulation_followup/
oa_v2_30_40_r1_strict_naive_rerun_results.json`. Each pair has:
- Two source adapter evaluations (accuracy on shared task)
- One merged adapter evaluation (uniform linear, 0.5/0.5)
- `merge_delta_vs_best_source` = merged_accuracy - best_source_accuracy

12 of 30 pairs are "poor" merges (delta ≤ -0.05).

### Spearman Correlations (metric vs merge delta)

A negative Spearman means higher metric → worse merge (desired).
A positive Spearman means higher metric → better merge (wrong sign).

| Metric | Spearman | Direction |
|--------|----------|-----------|
| OA-v1 max score | +0.186 | wrong |
| OA-v2 max score | +0.262 | wrong |
| **Theory risk score** | **+0.382** | **wrong** |
| **Theory inflation ratio** | **+0.386** | **wrong** |
| **Theory cross-term bound** | **+0.385** | **wrong** |

The analytical theory is a better *ranking* metric than v1 or v2
(Spearman magnitude 0.38 vs 0.19/0.26), but all three have the sign
flipped.

### Poor-Merge Enrichment (top tercile alerting)

| Metric | Recall | Lift | Rate (alerted) | Rate (rest) |
|--------|--------|------|----------------|-------------|
| OA-v1 | 0.25 | -0.15 | 0.30 | 0.45 |
| OA-v2 | 0.17 | -0.30 | 0.20 | 0.50 |
| Theory risk | 0.08 | -0.45 | 0.10 | 0.55 |
| Theory inflation | 0.08 | -0.45 | 0.10 | 0.55 |

Negative lift: the top tercile by predicted risk has *fewer* poor
merges than the rest. The analytical theory performs worst here —
its high-confidence inflation predictions select for same-task pairs
that merge well.

### Theory vs Heuristic Agreement (per-layer)

| Classification | Count | Fraction |
|---------------|-------|----------|
| agrees | 182 | 53.2% |
| theory_lower | 110 | 32.2% |
| theory_higher | 50 | 14.6% |

The theory and heuristic agree on majority of layers but diverge
substantially. The theory tends to estimate lower risk (32% of layers).

## Root Cause Analysis

The sign-flipped Spearman reveals a fundamental issue: **spectral
overlap is confounded with task similarity**.

Examining the per-pair results:

**Worst merges** (delta -0.57 to -0.29): All are cross-task pairs
(tweet_eval_hate × sst2, tweet_eval_irony × imdb, bert-hatexplain ×
tweet_eval_emotion). These have LOW spectral overlap, LOW theory risk.
The merge fails because the tasks conflict at a behavioral level — the
adapters push the base model in incompatible directions. No spectral
metric can detect this because the weight matrices look orthogonal
(which is spectrally "safe").

**Best merges** (delta +0.016, +0.004): All are same-task pairs
(IMDB × SST2, both sentiment classification). These have HIGH spectral
overlap, HIGH theory risk. The overlap is real — both adapters learned
similar sentiment features — and the merge succeeds *because* of this
alignment, not despite it.

**The analytical theory correctly predicts spectral inflation but
spectral inflation is not the failure mode.** The theory answers its
stated question ("will σ_1 of the merge exceed the weighted inputs?")
correctly. But that question turns out to be the wrong one for
predicting merge quality in this population.

## Implications for the Pipeline

### What this validates

1. The **verdict tree's branch structure** remains sound as a
   classification tool. It correctly identifies the *geometric regime*
   of a merge (orthogonal, redundant, conflicting, imbalanced).

2. **Norm equalization** has a clear theoretical basis: it reduces the
   cross-term when Frobenius imbalance drives inflation. The empirical
   observation that it's "one of the strongest single interventions"
   is consistent with the theory — it fixes the scale problem without
   needing to know the task relationship.

3. The **spectral audit** (stable rank, effective rank, principal angles)
   provides a valid geometric characterization of adapter pairs. The
   issue is not that the measurements are wrong, but that geometric
   compatibility ≠ behavioral compatibility.

### What this invalidates

1. **Over-accumulation score as a merge quality predictor.** Neither
   the heuristic nor the analytical theory can predict merge quality
   from spectral metrics alone. The OA diagnostic should remain
   "exploratory/informational" — it tells you about spectral geometry
   but not about downstream impact.

2. **Threshold tuning on spectral metrics alone.** The verdict tree
   thresholds (`low_overlap=0.2`, `high_overlap=0.5`, etc.) cannot be
   derived from or justified by the analytical conditions, because the
   analytical conditions measure the wrong thing for quality prediction.

### What additional information is needed

The negative completion condition identifies what's missing: **task
relationship information**. The `task_relationship_advisory` in the
merge QA report already captures this (`same_task`, `same_family`,
`cross_task`). The empirical data shows that task relationship is the
dominant predictor of merge quality in this cohort, not spectral
geometry.

A viable path forward would be:
- Use spectral metrics for *geometric characterization* (what regime)
- Use task relationship for *quality prediction* (will it work)
- Condition the over-accumulation advisory on task relationship:
  within same-task pairs, high overlap is beneficial; across tasks,
  even low overlap can produce harmful interference

## Artifacts

### Code deliverables

| File | Description |
|------|-------------|
| `gradience/vnext/merge/spectral_theory.py` | Phase 1-2 analytical predictions (13 dataclasses, 11 functions) |
| `gradience/vnext/merge/spectral_theory_test_utils.py` | Synthetic matrix generation with controlled geometry |
| `tests/merge/test_spectral_theory.py` | 22 unit tests for analytical predictions |
| `tests/merge/test_spectral_theory_validation.py` | 4 numerical validation sweeps |

### Theory documentation

| File | Description |
|------|-------------|
| `docs/theory/linear_merge_spectral_analysis.md` | Phase 1 derivations and validation hooks |
| `docs/theory/over_accumulation_theory.md` | Phase 2 exact conditions and heuristic comparison |

### Field trial artifacts

| File | Description |
|------|-------------|
| `theory_crosscheck.json` | 30-pair cross-check with all three metrics + outcomes |
| `theory_crosscheck.md` | Per-pair summary table |

### Pre-existing artifacts (unchanged)

| File | Description |
|------|-------------|
| `synthetic_validation_results.json` | Synthetic sweep (6,912 cases) |
| `heuristic_comparison.json` | OA-v1 vs OA-v2 on synthetic data |
| `empirical_crosscheck_oa_v2_30_40_r1.json` | 30-pair v1/v2 cross-check |

## Reproduce

```bash
# Run theory cross-check (requires adapter cache in field_trials/inventory_03)
python3 field_trials/analytical_spectral_geometry/run_theory_crosscheck.py

# Run unit tests
python3 -m pytest tests/merge/test_spectral_theory.py tests/merge/test_spectral_theory_validation.py -v

# Run synthetic heuristic comparison
python3 field_trials/analytical_spectral_geometry/run_phase2_heuristic_comparison.py
```

## Conclusion

The analytical spectral geometry work achieved its mathematical
objectives: closed-form over-accumulation conditions derived, validated
on synthetics, implemented as utilities. It answered the study's central
question ("what can be proved about the merged spectrum?") with correct
theorems.

The empirical cross-check then proved that the right answer to the
mathematical question is the wrong answer to the engineering question.
Spectral inflation is a real phenomenon but not a failure mode — at
least not in populations where task mismatch dominates.

This is a clean negative result. It closes the analytical line with
a clear finding: **spectral metrics characterize geometry; task
relationship predicts quality. The merge pipeline should use both,
not substitute one for the other.**
