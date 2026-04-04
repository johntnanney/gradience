# Strict Naive Explanatory Analysis

- Overall successful reruns: 30
- Overall failed reruns: 0

## Group Comparison
- High-tail n=15, mean delta vs best=-0.1047, var=0.0294, poor count=5
- Lower-tail n=15, mean delta vs best=-0.1088, var=0.0188, poor count=7
- Overall task-family mix: {'sentiment_binary': 22, 'topic_classification': 2, 'tweet_eval': 6}

## Continuous Relationship
- Spearman(delta_vs_best, pre_rerun max OA score): 0.1791
- Spearman(delta_vs_best, pre_rerun watch_layer_count): -0.0322

## Benign vs Inflation-Prone Decomposition
- Spearman(delta, mean alignment): 0.2756
- Spearman(delta, mean concentration): 0.1429
- Spearman(delta, mean coefficient exposure): 0.1312
- Spearman(delta, max alignment): 0.1136
- Spearman(delta, max concentration): 0.0951
- Spearman(delta, max coefficient exposure): 0.2041
- Spearman(delta, triple-high layer fraction): 0.1533
