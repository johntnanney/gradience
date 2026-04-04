# Over-Accumulation Validation Plan (Hook)

## Purpose

This note defines the follow-up evaluation hook for the phase-1 over-accumulation diagnostic.
Phase 1 ships a diagnostic estimate only. This document specifies how to validate whether that estimate is behaviorally informative.

## Validation Question

Within currently high-overlap / structurally compatible pairs:

- does over-accumulation risk explain outcome variance under naive merge?

This is the key test. The value of the diagnostic is not "does overlap matter" (already known), but whether it adds signal *inside* overlap-safe regions.

## Candidate Study Design

1. Select adapter pairs with high overlap and non-conflicting structural profile.
2. Split by diagnostic tier:
   - low over-accumulation risk
   - high over-accumulation risk
3. Run naive merge baselines with fixed coefficients (e.g., 0.5 / 0.5).
4. Compare downstream outcomes across tiers.
5. Optional phase-2 extension: compare naive vs calibrated merge variants for high-risk tiers.

## Required Logged Fields

The diagnostic should be logged with each pair and layer so downstream analysis is direct:

- per layer:
  - `over_accumulation_score`
  - `over_accumulation_band`
  - `over_accumulation_factors`
- per pair:
  - `over_accumulation_advisory`
  - `over_accumulation_summary`
  - `high_risk_layer_count`
  - `watch_layer_count`
  - `max_over_accumulation_score`

These fields are already emitted in phase 1 for JSON / report / API surfaces.

## Analysis Sketch

Primary comparisons:

- naive-merge retention metrics (`Q_min`, related downstream scores) by low vs high advisory tier
- effect sizes within high-overlap subsets

Secondary checks:

- calibration curves for `over_accumulation_score` vs outcome degradation
- interaction with conflict and norm-imbalance counts
- sensitivity to coefficient settings

## Success Criteria

Primary success:

- over-accumulation advisory explains meaningful variance inside high-overlap regions beyond existing overlap-only framing.

Secondary success:

- score/band stratification is directionally monotonic with observed degradation under naive merge.

Negative outcome (still useful):

- existing metrics are insufficient for robust discrimination; requires redesigned estimator inputs.

## Guardrails

- Do not treat phase-1 scores as validated causal claims before this study.
- Do not collapse main verdict taxonomy into the advisory.
- Keep language in reports explicitly diagnostic and cautionary until validation is complete.

