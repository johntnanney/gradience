# Over-Accumulation Diagnostic (Design Note)

## Scope

This note defines the phase-1 over-accumulation extension for Gradience merge audit.
It is explicitly:

- diagnostic-only
- additive to existing verdicts and recommendations
- CPU-only and lightweight
- built from already-computed merge-audit metrics

It is not a new merge execution method.

## Diagnostic Intuition

Over-accumulation is a merge-risk mode where:

- two adapters share strongly aligned directions
- naive weighted summation reinforces those already-shared directions
- the merged representation becomes disproportionately concentrated in that shared subspace

This can happen even when pair structure appears compatible under the current overlap framing.

## Distinction From Existing Paths

### vs. Conflict

- Conflict: shared directions disagree (opposing effects, cancellation risk).
- Over-accumulation: shared directions agree too much in dominant modes (inflation risk).

These are different failure modes. Conflict is destructive interference. Over-accumulation is excessive constructive reinforcement.

### vs. Norm Imbalance

- Norm imbalance: one adapter dominates by scale.
- Over-accumulation: aligned high-energy shared directions may dominate even when the pair is otherwise compatible.

They may co-occur, but they are not the same mechanism.

## Phase-1 Diagnostic Target

The phase-1 target is **risk estimation**, not proof:

- estimate whether naive linear merge is inflation-prone in a layer/pair
- expose watch/elevated advisory signals for interpretation
- avoid claiming guaranteed degradation without follow-up empirical validation

## Layer-Level Estimator (Phase 1)

The estimator is computed from existing metrics:

- alignment strength:
  - overlap (`mean_overlap`, `max_overlap`)
  - directional alignment (`directional_agreement`)
- shared spectral concentration:
  - stable-rank concentration (`stable_rank` vs `effective_rank`)
  - effective-rank utilization (`effective_rank` vs `nominal_rank`)
- coefficient exposure:
  - assumed merge coefficients (defaults to 0.5 / 0.5 when not otherwise specified)

Outputs per layer:

- `over_accumulation_score` in `[0, 1]`
- `over_accumulation_band`: `low | watch | high`
- `over_accumulation_factors`:
  - `alignment`
  - `concentration`
  - `coefficient_exposure`

## Parallel OA-v2 (Experimental)

The repo now also carries an experimental OA-v2 line in parallel mode.

- OA-v1 remains the authoritative production advisory path.
- OA-v2 is analysis-only and interaction-first.

OA-v2 uses full rank-r geometry when available:

- `principal_angle_cosines` (left)
- `right_principal_angle_cosines` (right)
- `effective_singular_values_a/b` (spectral weighting)

Primary interaction term:

- `spectral_overlap_weighted = Σ_i ŵ_i * (cosθ_i * cosφ_i)`
- `ŵ_i ∝ s_{a,i} * s_{b,i}`

Scoring:

- `interaction_primary = directional_gate * coefficient_exposure * spectral_overlap_weighted`
- `score_v2 = interaction_primary * (0.85 + 0.15 * concentration_secondary)` (clamped)

OA-v2 outputs appear in analysis artifacts and do not currently alter
verdict/recommendation policy.

## Pair-Level Advisory Synthesis

Layer outputs are aggregated into:

- `over_accumulation_advisory`: `none | watch | elevated`
- `over_accumulation_summary`
- `high_risk_layer_count`
- `watch_layer_count`
- `max_over_accumulation_score`

This advisory is additive and does not replace:

- top-level verdict (`safe/redundant/conflicting/imbalanced`)
- pair risk level (`low/medium/high`)
- primary strategy recommendation

## Interpretation Contract

Expected interpretation pattern:

- `safe` + `over_accumulation_watch`: structurally compatible, but monitor shared-direction inflation exposure.
- `redundant` + `over_accumulation_watch/elevated`: compatible and overlapping, but concentration/exposure suggests stronger inflation sensitivity.
- `conflicting`/`imbalanced` + advisory: existing pathology may remain primary; over-accumulation is secondary context.

Language remains cautious:

- "may be susceptible"
- "watch condition"
- "diagnostic estimate"

not:

- "guaranteed failure"
- "unsafe by proof"
