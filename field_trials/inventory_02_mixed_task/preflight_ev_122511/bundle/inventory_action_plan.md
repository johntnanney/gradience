# Inventory Action Plan

## Exclude / deprioritize

- none

## Same-task safe zone

- TransferGraph__roberta-base-finetuned-lora-ag_news × TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news  (medium risk, norm_equalized)

## Cross-task caution

- AG_NEWS × MNLI region
- AG_NEWS × TWEET_EVAL/HATE region
- AG_NEWS × TWEET_EVAL/IRONY region
- MNLI × TWEET_EVAL/HATE region
- MNLI × TWEET_EVAL/IRONY region
- TWEET_EVAL/HATE × TWEET_EVAL/IRONY region

## Evaluate first

- TransferGraph__roberta-base-finetuned-lora-ag_news × TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news  (medium risk, norm_equalized)

## Provenance

Sources with behavioral evidence: 5/5

## Summary

Inventory is mostly explained by task boundary. Candidate space reduced from 10 pairs to 1 (90% reduction).
