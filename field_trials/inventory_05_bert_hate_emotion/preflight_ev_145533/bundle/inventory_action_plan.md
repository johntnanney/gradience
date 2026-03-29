# Inventory Action Plan

## Exclude / deprioritize

- TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion: weak source — low confidence
- TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate: weak source — low confidence

## Same-task safe zone

- TransferGraph__bert-base-uncased-finetuned-lora-ag_news × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear)

## Near-miss candidates

Structurally plausible, evidence-constrained. Optional if evaluation budget allows.

- TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear — TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate is evidence-constrained)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized — TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion is evidence-constrained)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate  (low risk, linear — TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate is evidence-constrained)
- TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion  (low risk, linear — TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion is evidence-constrained)

## Cross-task caution

- AG_NEWS × TWEET_EVAL/EMOTION region
- AG_NEWS × TWEET_EVAL/HATE region
- TWEET_EVAL/EMOTION × TWEET_EVAL/HATE region

## Evaluate first

- TransferGraph__bert-base-uncased-finetuned-lora-ag_news × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear)

## Provenance

Sources with behavioral evidence: 8/8

## Summary

QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 3.
