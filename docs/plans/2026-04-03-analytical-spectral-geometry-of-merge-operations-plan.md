# ANALYTICAL_SPECTRAL_GEOMETRY_OF_MERGE_OPERATIONS_SPEC
## Repo-Facing CPU-Only Research Plan

## Purpose

Develop closed-form and semi-analytical results for how existing merge
strategies transform singular-value structure, using quantities already exposed
by Gradience merge audit (`SubspaceMetrics`).

Central question:

> Given two LoRA adapters with known spectral geometry, what can be proved
> about merged spectra under linear, TIES, DARE, and norm-equalized merge?

## Why this line

- zero training/GPU dependency
- directly addresses the exploratory status of over-accumulation
- can sharpen or falsify heuristic assumptions in current verdict policy

## In-scope

- exact linear/rank-1 derivations
- general-case bounds (Weyl/Fan style)
- semi-analytical TIES/DARE bounds
- synthetic CPU validation sweeps
- mapping analytical regions to existing verdict branches

## Out-of-scope

- behavioral outcome prediction
- new strategy invention
- production policy replacement from first pass

## Program phases

1. Linear merge exact + bound analysis
2. Over-accumulation inequalities in `SubspaceMetrics` terms
3. Norm-equalized spectral effects
4. TIES expected/worst-case bounds
5. DARE expected + concentration bounds
6. Strategy-selection map vs current verdict tree

## Implemented scaffold (this commit)

- `gradience/vnext/merge/spectral_theory.py`
- `gradience/vnext/merge/spectral_theory_test_utils.py`
- tests:
  - `tests/merge/test_spectral_theory.py`
  - `tests/merge/test_spectral_theory_validation.py`
- theory docs:
  - `docs/theory/*.md`
- field-trial artifact scaffolding:
  - `field_trials/analytical_spectral_geometry/*`

## Success criteria

### Success

Derive and validate a useful over-accumulation condition for linear merge and
at least two additional strategy-relevant bounds.

### Partial

Linear/over-accumulation analysis is useful; TIES/DARE remain looser and
probabilistic.

### Negative

Current observables are insufficient for strong analytical conditioning without
additional geometric measurements.

All outcomes are useful and should be documented.
