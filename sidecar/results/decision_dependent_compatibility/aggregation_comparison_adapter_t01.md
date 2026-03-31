# Aggregation Comparison (Adapter Triage Stress Test)

Generated: 2026-03-31T01:23:15.537343+00:00

Pairs analyzed: 3

## Label counts

- Worst-case (merge-like): {'merge_caution': 2, 'merge_risky': 1}
- Distributional (routing-like): {'routing_confusable': 1, 'routing_needs_disambiguation': 1, 'routing_separable': 1}
- QA gate-first (triage-like): {'qa_clear': 3}

## Profile stability check

- Same-task, same-family, and cross-task pairs remain separable under distributional aggregation.
- QA gate is clear in this panel (`qa_clear` for all pairs), contrasting with checkpoint T02's QA-dominant blocking regime.
