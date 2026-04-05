# Decoder Triage Study Scripts

This package implements the GPU study execution flow:

0. `phase0_validate_env.py` — six-gate smoke check (GPU/runtime, deps, model inference, throwaway LoRA training, audit, merge-audit)
1. `train_cohort.py` — train all adapters from `cohort_manifest.json` (resumable)
2. `run_pipeline.py` — run QA + pairwise merge audit + inventory bundle
3. `evaluate_merges.py` — merge/evaluate retained + near-miss + stratified controls
4. `analyze_results.py` — produce post-run analysis and memo

## Example usage

```bash
python scripts/decoder_triage_study/phase0_validate_env.py \
  --base-model mistralai/Mistral-7B-v0.1 \
  --output-dir /workspace/experiments/decoder_merge_triage/phase0_validation

python scripts/decoder_triage_study/train_cohort.py \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --output-dir /workspace/experiments/decoder_merge_triage/adapters \
  --log-dir /workspace/experiments/decoder_merge_triage/training_logs

python scripts/decoder_triage_study/run_pipeline.py \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --adapter-dir /workspace/experiments/decoder_merge_triage/adapters \
  --output-dir /workspace/experiments/decoder_merge_triage/pipeline_output \
  --inventory-id decoder_merge_triage

python scripts/decoder_triage_study/evaluate_merges.py \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --adapter-dir /workspace/experiments/decoder_merge_triage/adapters \
  --pipeline-dir /workspace/experiments/decoder_merge_triage/pipeline_output \
  --output-dir /workspace/experiments/decoder_merge_triage/merge_evaluation \
  --control-sample-size 6 \
  --max-eval-samples 1000

python scripts/decoder_triage_study/analyze_results.py \
  --study-dir /workspace/experiments/decoder_merge_triage \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --output /workspace/experiments/decoder_merge_triage/study_memo.md
```
