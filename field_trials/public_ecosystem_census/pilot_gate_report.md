# Pilot Validation Gate Report

**Overall**: FAIL
**Recommendation**: DIAGNOSE pipeline issues before expanding

## c1_pipeline_viability: PASS

- success_rate: 0.98
- threshold: 0.8
- attempted: 50
- audited: 49

## c2_metric_sanity: PASS

- issues: []

## c3_architecture_coverage: PASS

- family_counts: {'llama': 18, 'mistral': 15, 'qwen': 16}
- families_with_8_plus: 3
- threshold: 2

## c4_task_coverage: FAIL

- task_counts: {'classification': 10, 'chat_instruct': 35, 'general_unknown': 3, 'code': 1}
- tasks_with_5_plus: 2
- threshold: 3

## c5_visible_signal: PASS

- note: Non-blocking criterion
- details: stable_rank_mean: llama=2.026 vs mistral=2.409 (diff=0.384, 17.3% relative)
