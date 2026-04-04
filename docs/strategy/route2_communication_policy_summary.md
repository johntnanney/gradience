# Route 2 Communication Policy Summary

Date: 2026-04-03  
Status: reinforced ladder policy active + rank-proxy bounded policy freeze

## Purpose

This document defines how Route 2 claims should be communicated across public, product, architecture, and sidecar contexts after:

- the original claims ladder,
- mixed-evidence triage reinforcement,
- collapse-vs-contamination bounded replication,
- and R1 edge refinement.

## Safe stable language

Use directly (with scope bounds):

1. Evidence gating and conservative narrowing are stable workflow invariants.
2. Aggregation is a first-class seam.
3. Worst-case, distributional, and QA-dominant are distinct families.
4. Workflow-level portability exceeds metric-level portability.
5. The broadened substrate is real but narrow.

## Guarded-but-usable language

Use with explicit caveats:

1. Checkpoint triage is stable within validated scope.
2. Same-task/cross-task directional separation remains useful.
3. Same-family intermediate/optional states are review-relevant with threshold guardrails.
4. Taxonomy usage is coarse-grained; fine thresholds are non-canonical.
5. Collapse-vs-contamination is bounded merge-facing explanatory language.
6. Rank-proxy language is bounded and asymmetric:
gradient is the operational comparator, attenuate is companion ablation evidence, and rank-reduction remains paused in the current encoder/compressible regime.

## Suppressed language

Do not use in broad public/product messaging:

1. Optional/near-miss portability outside LoRA.
2. Routing-confusability non-transfer as a broad behavioral rule.
3. Any wording that implies ablation parity or rank-reduction viability as an operational default in the current bounded CPU encoder setup.

## Zone guidance

### Public writing

- Lead with core stable claims.
- Use bounded claims only with explicit caveats.
- Suppress thin claims.

### Product and alpha docs

- Use stable core + guarded operational claims.
- Avoid hard thresholds.
- Keep thin claims out.

### Internal architecture docs

- Include representation-local and scenario-local distinctions.
- Keep bounded behavioral channel language explicit.

### Sidecar/research docs

- Keep thin claims and cross-context extrapolations sidecar-only.
- Track reinforcement targets without promoting them.

## Policy source artifacts

- `sidecar/notes/n123a_route2_claims_ladder_reinforcement_baseline.md`
- `sidecar/notes/n123b_route2_claim_reinforcement_map.md`
- `sidecar/notes/n123_route2_claims_edge_refinement_r1.md`
- `sidecar/notes/n123d_route2_communication_policy.md`
- `sidecar/results/route2_claims_ladder/communication_policy.json`
