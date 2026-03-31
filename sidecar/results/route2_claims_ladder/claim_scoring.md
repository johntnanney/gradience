# Route 2 Claim Dimension Scoring

Program: Route 2 Claims Stability Ladder  
Stage: C  
Generated: 2026-03-31

## Scores

| Claim | Evidence base | Perturbation survival | Artifact coverage | Behavioral grounding | Product relevance |
|---|---|---|---|---|---|
| A1 | strong | survived_cleanly | multiple_classes_broad | indirect_behavioral_support | safe_to_expose |
| A2 | strong | survived_cleanly | multiple_classes_broad | workflow_only | safe_to_expose |
| A3 | strong | survived_with_caveats | single_class | indirect_behavioral_support | safe_with_guardrails |
| B1 | moderate | survived_with_caveats | multiple_classes_partial | indirect_behavioral_support | safe_with_guardrails |
| B2 | moderate | panel_sensitive | multiple_classes_partial | indirect_behavioral_support | safe_with_guardrails |
| B3 | moderate | survived_with_caveats | multiple_classes_partial | direct_behavioral_support | safe_with_guardrails |
| B4 | thin | panel_sensitive | multiple_classes_partial | structural_only | research_only |
| C1 | strong | survived_cleanly | multiple_classes_broad | structural_only | safe_with_guardrails |
| C2 | strong | survived_cleanly | multiple_classes_broad | workflow_only | safe_to_expose |
| C3 | strong | survived_cleanly | multiple_classes_partial | structural_only | safe_with_guardrails |
| D1 | strong | survived_cleanly | multiple_classes_partial | workflow_only | safe_to_expose |
| D2 | strong | survived_cleanly | multiple_classes_partial | structural_only | safe_with_guardrails |
| D3 | strong | survived_cleanly | multiple_classes_partial | indirect_behavioral_support | safe_with_guardrails |
| D4 | moderate | survived_with_caveats | multiple_classes_partial | structural_only | safe_with_guardrails |
| E1 | moderate | not_yet_stress_tested | single_class | direct_behavioral_support | safe_with_guardrails |
| E2 | moderate | not_yet_stress_tested | single_class | direct_behavioral_support | research_only |
| E3 | thin | not_yet_stress_tested | single_class | direct_behavioral_support | research_only |
| E4 | moderate | survived_with_caveats | multiple_classes_partial | direct_behavioral_support | safe_with_guardrails |
| F1 | strong | survived_with_caveats | multiple_classes_broad | workflow_only | safe_with_guardrails |
| F2 | strong | survived_cleanly | multiple_classes_broad | workflow_only | safe_to_expose |

## Scoring interpretation

- Strongest confidence cluster: workflow and aggregation seam claims.
- Most guarded cluster: same-family ordering and optional portability claims.
- Behavioral claims are meaningful but still narrower in artifact coverage than workflow claims.
