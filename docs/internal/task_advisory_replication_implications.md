# Task Advisory Replication — Implications

## What was replicated

The regime map and task-relationship advisory were tested on roberta-base (125M params) after initial validation on distilbert-base-uncased (66M params). All findings replicated:

| Finding | DistilBERT | RoBERTa | Status |
|---------|-----------|---------|--------|
| Same-task merges safe | 9/9 | 2/2 | Replicated |
| Cross-task merges degrade weaker task | 14/14 | 4/4 | Replicated |
| Task identity discriminates perfectly | 23/23 | 6/6 | Replicated |
| Advisory fires correctly | 46/46 | 6/6 | Replicated |
| Advisory false positives | 0 | 0 | Replicated |
| Pair-risk blind spot on cross-task pairs | Present | Present (worse) | Replicated |

## Key implications

### 1. The blind spot is real and backbone-independent

Pair-risk rates spectrally similar cross-task pairs as safe — on both backbones. On RoBERTa, the blind spot was actually worse: 3 of 4 cross-task pairs received `pair_risk=low` (the most permissive rating) despite degrading QNLI by 7-12pp. The advisory was the only signal that flagged these pairs on either backbone.

This is not a distilbert quirk. It is a structural property of how spectral pair-risk works: related-task adapters on the same backbone share enough spectral structure to look compatible even when their functional mappings diverge.

### 2. The advisory is perfectly selective across backbones

0/16 same-task pairs fired. 36/36 different-task pairs fired. Zero false positives across 52 total advisory checks on two backbones. The advisory's selectivity comes from `eval_dataset` metadata, not spectral analysis, so it generalizes trivially to any backbone where QA artifacts contain task information.

### 3. The advisory should now be considered stable for small encoder inventories

The evidence supports treating the advisory as a stable part of the interpretive layer — not just for distilbert, but for small encoder models generally. The claim is bounded: we have not tested on large models (7B+), decoder-only architectures, or non-GLUE task families. But within the tested regime, the signal is clean.

### 4. Pair-risk can be overly permissive on cross-task pairs

This is the most important structural finding. Pair-risk was designed to measure spectral compatibility, and spectrally, cross-task pairs on the same backbone can look highly compatible. The advisory catches what pair-risk cannot: that spectral compatibility does not imply functional compatibility across task boundaries.

This does not mean pair-risk should be redesigned. It means the advisory is a necessary complement in mixed-task inventories.

### 5. The asymmetric degradation pattern is backbone-independent

On both backbones, cross-task merges consistently preserved the stronger task and degraded the weaker task. This is a property of linear merging, not of any specific backbone: when you average two adapters, the one with stronger gradient signal dominates.

## What this changes

- **Regime map claims:** Now "small encoder models" instead of "distilbert"
- **Advisory status:** From "validated on one backbone" to "validated across two backbones"
- **Evidence totals:** 29 adjudication pairs, 52 advisory checks, 2 backbones, 0 false positives
- **Confidence in public claims:** Strong enough for a blog post or paper subsection about the regime map

## What this does not change

- Core-space status (still narrow, still advanced-only)
- Pair-risk logic (no code changes)
- Advisory implementation (no code changes)
- The need for further testing on larger models before broadening claims beyond small encoders
