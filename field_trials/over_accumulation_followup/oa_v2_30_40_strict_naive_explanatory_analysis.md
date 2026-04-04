# Strict Naive Explanatory Analysis

- Overall successful reruns: 13
- Overall failed reruns: 0

## Group Comparison
- High-tail n=7, mean delta vs best=-0.0151, var=0.0003, poor count=1
- Lower-tail n=6, mean delta vs best=-0.1267, var=0.0416, poor count=2
- Overall task-family mix: {'sentiment_binary': 6, 'tweet_eval': 5, 'topic_classification': 2}

## Continuous Relationship
- Spearman(delta_vs_best, pre_rerun max OA score): 0.0530
- Spearman(delta_vs_best, pre_rerun watch_layer_count): -0.2350

## Benign vs Inflation-Prone Decomposition
- Spearman(delta, mean alignment): 0.0474
- Spearman(delta, mean concentration): 0.0789
- Spearman(delta, mean coefficient exposure): 0.1774
- Spearman(delta, max alignment): -0.3264
- Spearman(delta, max concentration): 0.0910
- Spearman(delta, max coefficient exposure): 0.3917
- Spearman(delta, triple-high layer fraction): -0.0044
