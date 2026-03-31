# Mixed-Evidence Aggregation Comparison

## Agreement distribution

| pattern | count |
|---|---:|
| full_agreement | 0 |
| partial_agreement | 6 |
| strong_divergence | 2 |

## Per-case outputs

| case_id | worst_case | distributional | qa_dominant | qa_gated_distributional | agreement |
|---|---|---|---|---|---|
| anchor_clear_retained_irony | merge_caution | routing_confusable | qa_clear | confusable | partial_agreement |
| anchor_blocked_same_task_checkpoint | merge_caution | routing_confusable | qa_blocked | qa_blocked | strong_divergence |
| anchor_blocked_cross_task_checkpoint | merge_risky | routing_separable | qa_blocked | qa_blocked | strong_divergence |
| review_mixed_cross_task_sst2_mrpc | merge_caution | routing_needs_disambiguation | qa_review | qa_review | partial_agreement |
| review_mixed_cross_task_yelp_mrpc | merge_risky | routing_separable | qa_review | qa_review | partial_agreement |
| optional_same_family_clear_sst2_imdb | merge_caution | routing_needs_disambiguation | qa_clear | needs_disambiguation | partial_agreement |
| optional_same_family_mixed_sst2_yelp_a | merge_risky | routing_needs_disambiguation | qa_review | qa_review | partial_agreement |
| optional_same_family_mixed_sst2_yelp_b | merge_risky | routing_needs_disambiguation | qa_review | qa_review | partial_agreement |

## Soft-middle read

1. QA-dominant still cleanly partitions the panel into `qa_clear`, `qa_review`, and `qa_blocked`.
2. Same-family optional cases remain review/clear, not blocked.
3. Mixed-review cases show secondary structural ordering under distributional labels, even when primary QA-dominant output remains `qa_review`.
