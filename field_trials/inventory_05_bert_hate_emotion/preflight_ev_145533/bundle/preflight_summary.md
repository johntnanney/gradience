# Preflight Summary — preflight_ev_145533

**Run:** run_20260328_195623
**Generated:** 2026-03-28 19:56 UTC

## Source QA

- eligible: 6
- flagged_weak: 2

## Task-boundary partition

- Same-task pairs (advisory silent): 7
- Cross-task pairs (advisory active): 21
- Total pairs: 28

## Reduced candidate set

- TransferGraph__bert-base-uncased-finetuned-lora-ag_news × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear)

## Near-miss candidates

Structurally plausible, evidence-constrained. Optional if evaluation budget allows.

- TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear — TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate is evidence-constrained)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized — TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion is evidence-constrained)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate  (low risk, linear — TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate is evidence-constrained)
- TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion  (low risk, linear — TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion is evidence-constrained)

**QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 3.**

## Provenance

Sources with behavioral evidence: 8/8

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

## Inventory action plan

See `inventory_action_plan.md` for the full structured plan.

## Detailed artifacts

- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
- `neighborhoods/` — neighborhood grouping
