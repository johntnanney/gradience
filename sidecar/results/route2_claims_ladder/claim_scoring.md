# Route 2 Claim Dimension Scoring

Program: Route 2 Claims Stability Ladder  
Stage: C  
Generated: 2026-04-01

## Scores

| Claim | Evidence base | Perturbation survival | Artifact coverage | Behavioral grounding | Product relevance | Edge bucket | Public policy |
|---|---|---|---|---|---|---|---|
| A1 | strong | survived_cleanly | multiple_classes_broad | indirect_behavioral_support | safe_to_expose | core_stable_non_edge | allow_core |
| A2 | strong | survived_cleanly | multiple_classes_broad | workflow_only | safe_to_expose | core_stable_non_edge | allow_core |
| A3 | strong | survived_with_caveats | single_class | indirect_behavioral_support | safe_with_guardrails | stable_but_local | allow_bounded |
| B1 | moderate | survived_with_caveats | multiple_classes_partial | indirect_behavioral_support | safe_with_guardrails | moderate_but_product_relevant | allow_bounded |
| B2 | moderate | panel_sensitive | multiple_classes_partial | indirect_behavioral_support | safe_with_guardrails | moderate_but_product_relevant | allow_bounded |
| B3 | moderate | survived_with_caveats | multiple_classes_partial | direct_behavioral_support | safe_with_guardrails | moderate_but_product_relevant | allow_bounded |
| B4 | thin | panel_sensitive | multiple_classes_partial | structural_only | research_only | thin_suppress_public | suppress_public |
| C1 | strong | survived_cleanly | multiple_classes_broad | structural_only | safe_with_guardrails | core_stable_non_edge | allow_core |
| C2 | strong | survived_cleanly | multiple_classes_broad | workflow_only | safe_to_expose | core_stable_non_edge | allow_core |
| C3 | strong | survived_cleanly | multiple_classes_partial | structural_only | safe_with_guardrails | stable_but_local | allow_bounded |
| D1 | strong | survived_cleanly | multiple_classes_partial | workflow_only | safe_to_expose | core_stable_non_edge | allow_core |
| D2 | strong | survived_cleanly | multiple_classes_partial | structural_only | safe_with_guardrails | core_stable_non_edge | allow_core |
| D3 | strong | survived_cleanly | multiple_classes_partial | indirect_behavioral_support | safe_with_guardrails | stable_but_local | allow_bounded |
| D4 | moderate | survived_with_caveats | multiple_classes_partial | structural_only | safe_with_guardrails | moderate_but_product_relevant | allow_bounded |
| E1 | moderate | not_yet_stress_tested | single_class | direct_behavioral_support | safe_with_guardrails | stable_but_local | allow_bounded |
| E2 | moderate | survived_with_caveats | single_class | direct_behavioral_support | safe_with_guardrails | stable_but_local | allow_bounded |
| E3 | thin | not_yet_stress_tested | single_class | direct_behavioral_support | research_only | thin_suppress_public | suppress_public |
| E4 | moderate | survived_with_caveats | multiple_classes_partial | direct_behavioral_support | safe_with_guardrails | moderate_but_product_relevant | allow_bounded |
| F1 | strong | survived_with_caveats | multiple_classes_broad | workflow_only | safe_with_guardrails | core_stable_non_edge | allow_core |
| F2 | strong | survived_cleanly | multiple_classes_broad | workflow_only | safe_to_expose | core_stable_non_edge | allow_core |

## Scoring interpretation

- Strongest confidence cluster: workflow and aggregation seam claims.
- Most guarded cluster: same-family ordering and optional portability claims.
- Edge refinement layer separates stable-but-local from moderate-but-product-relevant claims and enforces conservative public suppression for thin claims.
