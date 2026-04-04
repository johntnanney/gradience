# Task-Balanced Extension Baseline

- Snapshot date (UTC): 2026-04-03T22:11:34.599717+00:00
- Baseline fingerprints: 26
- Architecture counts: `{"llama": 18, "mistral": 8}`
- Task counts: `{"chat_instruct": 13, "classification": 10, "general_unknown": 3}`

## Frozen Pilot-Plus Signals

- Mean architecture eta-squared: 0.1155
- Mean task eta-squared: 0.2599
- Dominant factor: task
- Architecture kNN purity (mean): 0.9
- Task kNN purity (mean): 0.7
- Module-type sign result: Attention < MLP in 2/8 adapters (25%)

## Limitation

Task balance remains the main limitation if target task categories are sparse or if one category dominates kNN neighborhoods.
