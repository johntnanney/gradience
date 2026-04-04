# Route 2 Claims Ladder — Edge Refinement Table (R1)

Program: route2_claims_stability_ladder  
Stage: R1_edge_refinement  
Generated: 2026-04-01

## Summary counts
- core_stable_non_edge: 8
- stable_but_local: 5
- moderate_but_product_relevant: 5
- thin_suppress_public: 2

## Stage C required fields table

| claim_id | original_ladder_status | refined_status_r1 | communication_policy_tag | allowed_use | short_explanation |
|---|---|---|---|---|---|
| A1 | stable | core_stable_non_edge | allow_core | public | Broad cross-artifact workflow invariant; foundational and not scenario-local. |
| A2 | stable | core_stable_non_edge | allow_core | public | Broad repeated workflow invariant across tested artifacts and inventories. |
| A3 | stable | stable_but_local | allow_bounded | product_guarded | Strong but bounded to checkpoint triage envelope and single-class operational path. |
| B1 | moderately_stable | moderate_but_product_relevant | allow_bounded | product_guarded | Directionally useful for triage/ranking decisions with explicit parity caveats. |
| B2 | moderately_stable | moderate_but_product_relevant | allow_bounded | product_guarded | Useful middle-state signal for workflow decisions but panel-sensitive ordering. |
| B3 | moderately_stable | moderate_but_product_relevant | allow_bounded | product_guarded | Operationally useful review/optional framing with threshold guardrails. |
| B4 | thin | thin_suppress_public | suppress_public | research_only | Thin portability evidence outside LoRA; not reliable for public claims. |
| C1 | stable | core_stable_non_edge | allow_core | public | Stable cross-artifact guardrail about metric locality; core calibration claim. |
| C2 | stable | core_stable_non_edge | allow_core | public | Core Route 2 portability framing across programs. |
| C3 | stable | stable_but_local | allow_bounded | product_guarded | Strong but representation-path-specific (checkpoint summary route), not universal factor equivalence. |
| D1 | stable | core_stable_non_edge | allow_core | public | Seam-level aggregation claim repeatedly stability-checked. |
| D2 | stable | core_stable_non_edge | allow_core | public | Distinct aggregation families replicated and core to decision-dependent interpretation. |
| D3 | stable | stable_but_local | allow_bounded | product_guarded | Strong within QA-dominant triage family, but scenario-local by design. |
| D4 | moderately_stable | moderate_but_product_relevant | allow_bounded | product_guarded | Taxonomy useful for product/workflow messaging at coarse grain with guarded boundaries. |
| E1 | moderately_stable | stable_but_local | allow_bounded | product_guarded | Behavioral profile differentiation is strong but still bounded to current behavioral coverage. |
| E2 | moderately_stable | stable_but_local | allow_bounded | product_guarded | Replication-supported channel distinction in merge-facing setting; bounded cross-context scope. |
| E3 | thin | thin_suppress_public | suppress_public | research_only | Routing behavioral transfer remains sparse and should stay out of public framing. |
| E4 | moderately_stable | moderate_but_product_relevant | allow_bounded | product_guarded | Useful behavioral guidance for optional/review handling with guardrails. |
| F1 | stable | core_stable_non_edge | allow_core | public | Core synthesis claim with explicit bounded-scope wording already embedded. |
| F2 | stable | core_stable_non_edge | allow_core | public | Primary Route 2 portable value claim for product/public explanation. |

## Legacy compatibility fields

| claim_id | final_ladder_status | edge_refinement_bucket | public_language_policy | justification |
|---|---|---|---|---|
| A1 | stable | core_stable_non_edge | allow_core | Broad cross-artifact workflow invariant; foundational and not scenario-local. |
| A2 | stable | core_stable_non_edge | allow_core | Broad repeated workflow invariant across tested artifacts and inventories. |
| A3 | stable | stable_but_local | allow_bounded | Strong but bounded to checkpoint triage envelope and single-class operational path. |
| B1 | moderately_stable | moderate_but_product_relevant | allow_bounded | Directionally useful for triage/ranking decisions with explicit parity caveats. |
| B2 | moderately_stable | moderate_but_product_relevant | allow_bounded | Useful middle-state signal for workflow decisions but panel-sensitive ordering. |
| B3 | moderately_stable | moderate_but_product_relevant | allow_bounded | Operationally useful review/optional framing with threshold guardrails. |
| B4 | thin | thin_suppress_public | suppress_public | Thin portability evidence outside LoRA; not reliable for public claims. |
| C1 | stable | core_stable_non_edge | allow_core | Stable cross-artifact guardrail about metric locality; core calibration claim. |
| C2 | stable | core_stable_non_edge | allow_core | Core Route 2 portability framing across programs. |
| C3 | stable | stable_but_local | allow_bounded | Strong but representation-path-specific (checkpoint summary route), not universal factor equivalence. |
| D1 | stable | core_stable_non_edge | allow_core | Seam-level aggregation claim repeatedly stability-checked. |
| D2 | stable | core_stable_non_edge | allow_core | Distinct aggregation families replicated and core to decision-dependent interpretation. |
| D3 | stable | stable_but_local | allow_bounded | Strong within QA-dominant triage family, but scenario-local by design. |
| D4 | moderately_stable | moderate_but_product_relevant | allow_bounded | Taxonomy useful for product/workflow messaging at coarse grain with guarded boundaries. |
| E1 | moderately_stable | stable_but_local | allow_bounded | Behavioral profile differentiation is strong but still bounded to current behavioral coverage. |
| E2 | moderately_stable | stable_but_local | allow_bounded | Replication-supported channel distinction in merge-facing setting; bounded cross-context scope. |
| E3 | thin | thin_suppress_public | suppress_public | Routing behavioral transfer remains sparse and should stay out of public framing. |
| E4 | moderately_stable | moderate_but_product_relevant | allow_bounded | Useful behavioral guidance for optional/review handling with guardrails. |
| F1 | stable | core_stable_non_edge | allow_core | Core synthesis claim with explicit bounded-scope wording already embedded. |
| F2 | stable | core_stable_non_edge | allow_core | Primary Route 2 portable value claim for product/public explanation. |
