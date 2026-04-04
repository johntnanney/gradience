# Merge-Aware Monitor Reference-Choice Demo Design

## Scope

This is a tiny CPU-only interpretability demo for reference selection in the
merge-aware training monitor.

It reuses the existing callback monitor and keeps all training-side settings
fixed while changing only `merge_target`.

## Frozen Setup

- `training_run_id`: `tiny_encoder_reference_choice_demo_v1`
- `training_task`: `sst2_like_sentiment_train_stub` (synthetic label)
- `base_model`: `tiny-peft-monitor-demo`
- trajectory steps: `1, 2, 3, 4, 5, 6`
- monitor mode: diagnostic-only
- guard/monitor heuristics: disabled for this demo to isolate merge-aware output

## Reference Conditions

### A) Same-task

- reference type: `same_task`
- pseudo task: `sst2_like_sentiment_reference`
- rationale: aligned with primary trajectory direction, intended to emulate
  same-task structural compatibility.

### B) Same-family

- reference type: `same_family`
- pseudo task: `imdb_like_sentiment_reference`
- rationale: mixed primary + secondary directions, intended to emulate partial
  family-level compatibility.

### C) Cross-task

- reference type: `cross_task`
- pseudo task: `ag_news_like_topic_reference`
- rationale: primary direction opposition, intended to emulate cross-task
  mismatch.

## Compatibility Contract

- all references are generated for the same base/tiny LoRA structure
- all are loadable by current monitor callback implementation
- only reference choice changes across conditions

## Artifact Mapping

- machine design: `demo_design.json`
- runner: `run_reference_choice_demo.py`
- run outputs: `runs/<reference_type>/run.jsonl`
