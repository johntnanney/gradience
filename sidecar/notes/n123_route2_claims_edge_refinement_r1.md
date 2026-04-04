# n123 -- Route 2 Claims-Ladder Edge Refinement Pass (R1)

**Type:** synthesis refinement note  
**Date:** 2026-04-01  
**Program:** Route 2 Claims Stability Ladder  
**Stage:** R1 edge refinement overlay  
**Depends on:** n111, n112, `sidecar/results/route2_claims_ladder/stability_ladder.json`, `sidecar/results/route2_claims_ladder/claim_scoring.json`  
**Status:** complete

---

## Objective

Apply one in-place + overlay refinement pass to clarify edge-case communication posture across all 20 claims without introducing a new ladder.

Target edge buckets:

1. `stable_but_local`
2. `moderate_but_product_relevant`
3. `thin_suppress_public`

Non-edge stable claims remain `core_stable_non_edge`.

---

## Required Stage C fields

Primary Stage C communication-policy fields:

- `claim_id`
- `original_ladder_status`
- `refined_status_r1`
- `communication_policy_tag`
- `short_explanation`
- `allowed_use` (`public` | `product_guarded` | `internal_only` | `research_only`)

For backward compatibility with earlier ladder artifacts, the overlay also retains:

- `final_ladder_status`
- `edge_refinement_bucket`
- `public_language_policy`
- `justification`

---

## Fixed rubric and precedence

Deterministic rules:

1. `thin_suppress_public` if `final_ladder_status == thin` or `product_implication_tag == research_only` with thin evidence.
2. `stable_but_local` if claim is strong/moderate in bounded scope and depends on narrow scenario/class coverage.
3. `moderate_but_product_relevant` if `final_ladder_status == moderately_stable` and `product_implication_tag == safe_with_guardrails` and claim is workflow-relevant.
4. Else `core_stable_non_edge`.

Precedence:

1. `thin_suppress_public`
2. `stable_but_local`
3. `moderate_but_product_relevant`
4. `core_stable_non_edge`

---

## Per-claim edge classification rationale

### `core_stable_non_edge` (8)

- `A1`, `A2`: broad cross-artifact workflow invariants.
- `C1`, `C2`: core cross-artifact portability calibration claims.
- `D1`, `D2`: seam-level aggregation claims with stability support.
- `F1`, `F2`: core bounded substrate synthesis and portable value framing.

### `stable_but_local` (5)

- `A3`: checkpoint triage is strong but bounded to tested checkpoint envelope.
- `C3`: checkpoint broadening is representation-path-specific (summary route).
- `D3`: QA-dominant distinctness is strong but scenario-local to triage interpretation.
- `E1`: behavioral profile distinction is strong but still bounded by current behavioral coverage.
- `E2`: collapse-vs-contamination is replication-supported but merge-facing and non-universal.

### `moderate_but_product_relevant` (5)

- `B1`: directional same-task/cross-task separation is useful with parity caveats.
- `B2`: same-family intermediate signal useful but panel-sensitive.
- `B3`: same-family optional review signal useful with threshold guardrails.
- `D4`: aggregation taxonomy useful at coarse product/workflow grain.
- `E4`: optionality safe-like behavioral framing useful with guardrails.

### `thin_suppress_public` (2)

- `B4`: optional/near-miss portability outside LoRA remains thin.
- `E3`: routing-confusability behavioral transfer remains thin.

---

## Public-language implications

1. `allow_core`: only `core_stable_non_edge` claims.
2. `allow_bounded`: both `stable_but_local` and `moderate_but_product_relevant` claims with explicit scope caveats.
3. `suppress_public`: all `thin_suppress_public` claims.

This enforces conservative external language while preserving useful bounded internal/product guidance.

---

## Deliverables

- `sidecar/results/route2_claims_ladder/edge_refinement_table.json`
- `sidecar/results/route2_claims_ladder/edge_refinement_table.md`
- Updated in-place ladder artifacts:
  - `claim_scoring.json/.md`
  - `stability_ladder.json/.md`
  - `implications_summary.json`

Allowed-use mapping applied in this pass:

- `core_stable_non_edge` -> `public`
- `stable_but_local` -> `product_guarded`
- `moderate_but_product_relevant` -> `product_guarded`
- `thin_suppress_public` -> `research_only`
