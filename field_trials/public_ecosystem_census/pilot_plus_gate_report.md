# Pilot-Plus Validation Gate Report

**Overall**: PASS
**Recommendation**: PROCEED to core cohort

## s1_pipeline_viability: PASS

- success_rate: 0.98
- threshold: 0.7
- attempted: 50
- audited: 49

## s2_family_coverage: PASS

- family_counts: {'llama': 18, 'mistral': 15, 'qwen': 16}
- families_with_5_plus: 3
- threshold: 3

## s3_residualized_signal: PASS

- metrics_above_005: 4
- mean_residualized_arch_eta_sq: 0.1009
- threshold: >=1 metric with residualized eta-sq > 0.05

## s4_subtype_coverage: PASS

- subtypes_with_10_plus_layers: 7
- total_subtypes: 7
- threshold: 3
