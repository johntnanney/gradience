# Inventory Action Plan

## Exclude / deprioritize

- TransferGraph__Aureliano_distilbert-base-uncased-if-finetuned-lora-tweet_eval_hate: weak source — low confidence
- jmeneu__distilbert-base-uncased-lora-text-classification: weak source — low confidence

## Same-task safe zone

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × NightPrince__peft-distilbert-sst2  (medium risk, norm_equalized)

## Cross-task caution

- AG_NEWS × SST-2 region
- AG_NEWS × TWEET_EVAL/EMOTION region
- AG_NEWS × TWEET_EVAL/HATE region
- SST-2 × TWEET_EVAL/EMOTION region
- SST-2 × TWEET_EVAL/HATE region
- TWEET_EVAL/EMOTION × TWEET_EVAL/HATE region

## Evaluate first

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × NightPrince__peft-distilbert-sst2  (medium risk, norm_equalized)

## Provenance

Sources with behavioral evidence: 8/8

## Summary

QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 2.
