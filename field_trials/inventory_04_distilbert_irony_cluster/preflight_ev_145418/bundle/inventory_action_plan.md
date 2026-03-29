# Inventory Action Plan

## Exclude / deprioritize

- TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony: weak source — low confidence

## Same-task safe zone

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized)
- TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (low risk, linear)

## Near-miss candidates

Structurally plausible, evidence-constrained. Optional if evaluation budget allows.

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony  (low risk, linear — TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony is evidence-constrained)
- TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (low risk, linear — TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony is evidence-constrained)
- TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony  (low risk, linear — TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony is evidence-constrained)

## Cross-task caution

- AG_NEWS × TWEET_EVAL/EMOTION region
- AG_NEWS × TWEET_EVAL/IRONY region
- TWEET_EVAL/EMOTION × TWEET_EVAL/IRONY region

## Evaluate first

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized)

## Provenance

Sources with behavioral evidence: 8/8

## Summary

QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 5.
