# Controlled Decoder Merge Triage — Execution Runbook

## 0) Phase 0 smoke gate (recommended before full run)

```bash
python scripts/decoder_triage_study/phase0_validate_env.py \
  --base-model mistralai/Mistral-7B-v0.1 \
  --output-dir /workspace/experiments/decoder_merge_triage/phase0_validation
```

Gate criteria (must all pass):

1. GPU/runtime readiness
2. Environment/dependency readiness
3. Base model load + 10-example inference
4. Throwaway rank-8 LoRA SST-2 training smoke
5. `gradience audit` on throwaway adapter
6. `gradience merge-audit` on throwaway adapter pair

## 1) Train cohort

```bash
python scripts/decoder_triage_study/train_cohort.py \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --output-dir /workspace/experiments/decoder_merge_triage/adapters \
  --log-dir /workspace/experiments/decoder_merge_triage/training_logs
```

Resumable behavior: completed adapters are skipped based on per-adapter `training_manifest.json`.

## 2) Run pipeline

```bash
python scripts/decoder_triage_study/run_pipeline.py \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --adapter-dir /workspace/experiments/decoder_merge_triage/adapters \
  --output-dir /workspace/experiments/decoder_merge_triage/pipeline_output \
  --inventory-id decoder_merge_triage
```

Outputs include:

- `preflight_summary.json`
- `inventory_action_plan.json`
- `pair_reports/*.json`

## 3) Evaluate retained / near-miss / control pairs

```bash
python scripts/decoder_triage_study/evaluate_merges.py \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --adapter-dir /workspace/experiments/decoder_merge_triage/adapters \
  --pipeline-dir /workspace/experiments/decoder_merge_triage/pipeline_output \
  --output-dir /workspace/experiments/decoder_merge_triage/merge_evaluation \
  --control-sample-size 6 \
  --max-eval-samples 1000
```

Resumable behavior: existing `merge_evaluation/pair_results/*.json` are skipped.

## 4) Analyze

```bash
python scripts/decoder_triage_study/analyze_results.py \
  --study-dir /workspace/experiments/decoder_merge_triage \
  --manifest scripts/decoder_triage_study/cohort_manifest.json \
  --output /workspace/experiments/decoder_merge_triage/study_memo.md
```

## 5) Canonical artifact map

- `adapter_manifest.json`
- `pipeline_output/`
- `merge_evaluation/`
- `analysis/`
- `study_memo.md`
