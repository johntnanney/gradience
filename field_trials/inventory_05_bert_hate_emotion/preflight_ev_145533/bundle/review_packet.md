# Review Packet — preflight_ev_145533

**Run:** run_20260328_195623
**Generated:** 2026-03-28 19:56 UTC
**Adapters:** 8
**Pairs:** 28
**Retained candidates:** 3

## Inventory Policy Summary

*Not available for this run.*

## Source QA / Trust Snapshot

- eligible: 6
- flagged_weak: 2

**Evidence:** 8/8 sources with behavioral evidence

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

**Excluded sources:**
- TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion: weak source — low confidence
- TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate: weak source — low confidence

## Action Plan

**Starting pairs:** 28
**Retained candidates:** 3
**Reduction:** 89%
**Cross-task excluded:** 12

**Evaluate first:**
- TransferGraph__bert-base-uncased-finetuned-lora-ag_news × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear)

**Same-task safe zone:**
- TransferGraph__bert-base-uncased-finetuned-lora-ag_news × TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion  (low risk, linear)
- TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate × TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate  (low risk, linear)

**Cross-task caution zone:**
- AG_NEWS × TWEET_EVAL/EMOTION region
- AG_NEWS × TWEET_EVAL/HATE region
- TWEET_EVAL/EMOTION × TWEET_EVAL/HATE region

**Summary:** QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 3.

## Artifacts

- `preflight_summary.json` — machine-readable preflight summary
- `preflight_summary.md` — human-readable preflight summary
- `inventory_action_plan.md` — structured action plan
- `run_manifest.json` — run metadata
- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
