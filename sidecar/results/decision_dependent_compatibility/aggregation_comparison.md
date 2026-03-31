# Aggregation Comparison (Checkpoint T02)

Generated: 2026-03-31T01:23:15.534924+00:00

Pairs analyzed: 10

## Label counts

- Worst-case (merge-like): {'merge_caution': 2, 'merge_risky': 8}
- Distributional (routing-like): {'routing_confusable': 1, 'routing_needs_disambiguation': 7, 'routing_separable': 2}
- QA gate-first (triage-like): {'qa_blocked': 9, 'qa_review': 1}

## Collapse vs separation

- `merge_caution` collapsed 2 pairs but split into routing_confusable, routing_needs_disambiguation under distributional aggregation.
- `merge_risky` collapsed 8 pairs but split into routing_needs_disambiguation, routing_separable under distributional aggregation.

## QA overrides

- QA gate overrode structurally non-separable cases for 8 pairs.
