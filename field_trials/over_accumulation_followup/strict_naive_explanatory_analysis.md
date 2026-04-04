# Strict Naive Explanatory Analysis

- Overall successful reruns: 12
- Overall failed reruns: 0

## Group Comparison
- High-tail n=6, mean delta vs best=-0.0183, var=0.0003, poor count=1
- Lower-tail n=6, mean delta vs best=-0.1243, var=0.0421, poor count=2
- Overall task-family mix: {'sentiment_binary': 6, 'tweet_eval': 5, 'topic_classification': 1}

## Continuous Relationship
- Spearman(delta_vs_best, pre_rerun max OA score): -0.0176
- Spearman(delta_vs_best, pre_rerun watch_layer_count): -0.2195

## Benign vs Inflation-Prone Decomposition
- Spearman(delta, mean alignment): -0.0141
- Spearman(delta, mean concentration): -0.0131
- Spearman(delta, mean coefficient exposure): 0.3161
- Spearman(delta, max alignment): -0.3902
- Spearman(delta, max concentration): 0.0169
- Spearman(delta, max coefficient exposure): 0.3951
- Spearman(delta, triple-high layer fraction): -0.1486
