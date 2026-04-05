# RunPod GPU Resumable Pipeline

**Audience:** maintainers and research operators  
**Status:** workflow recipe for one-block GPU runs  
**Purpose:** run train → preflight → merge/eval in a single resumable pipeline on RunPod/Lambda

## Goal

Use one A100 40GB instance (24-30 hours) to run:

1. Sequential training of all adapters (example cohort: 16)
2. CPU-side QA + pairwise spectral audit + inventory preflight on the same machine
3. Merge/eval pass over retained pairs, near-miss pairs, and a stratified control sample

The pipeline is resumable by phase and by item (adapter/pair). If the pod dies after adapter 12, rerun and it continues from 13.

## Files

- Script: [`scripts/runpod/run_resumable_gpu_pipeline.py`](../../scripts/runpod/run_resumable_gpu_pipeline.py)
- Example config: [`scripts/runpod/runpod_gpu_pipeline_config.example.json`](../../scripts/runpod/runpod_gpu_pipeline_config.example.json)

## Quick Start (RunPod)

```bash
cd /workspace
git clone <your-gradience-repo-url> gradience
cd gradience

pip install -e ".[dev]"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

python3 scripts/runpod/run_resumable_gpu_pipeline.py \
  --config scripts/runpod/runpod_gpu_pipeline_config.example.json \
  --phase all
```

## Resume Commands

Resume full pipeline:

```bash
python3 scripts/runpod/run_resumable_gpu_pipeline.py \
  --config scripts/runpod/runpod_gpu_pipeline_config.example.json \
  --phase all
```

Resume only training:

```bash
python3 scripts/runpod/run_resumable_gpu_pipeline.py \
  --config scripts/runpod/runpod_gpu_pipeline_config.example.json \
  --phase train
```

Resume only preflight:

```bash
python3 scripts/runpod/run_resumable_gpu_pipeline.py \
  --config scripts/runpod/runpod_gpu_pipeline_config.example.json \
  --phase preflight
```

Resume only merge/eval:

```bash
python3 scripts/runpod/run_resumable_gpu_pipeline.py \
  --config scripts/runpod/runpod_gpu_pipeline_config.example.json \
  --phase merge_eval
```

## Output Layout

Under `output_root` from config:

```text
pipeline_meta.json
pipeline_state.json
training_summary.json
adapters/<adapter_name>/...
preflight/
  qa/*.json
  pair_reports/*.json
  inventory/inventory_summary.json
  neighborhoods/neighborhoods.json
  inventory_action_plan.json
  inventory_action_plan.txt
merge_eval/
  selection_manifest.json
  pair_results/*.json
  merge_eval_summary.json
  merged_adapters/<pair_id>/merged_adapter/...
```

## What Is Resumable

- Training phase: each adapter writes `adapters/<name>/training_manifest.json`; completed adapters are skipped.
- Preflight phase: QA files and pair reports are file-checkpointed; existing outputs are skipped.
- Merge/eval phase: each selected pair writes `merge_eval/pair_results/<pair_id>.json`; completed pairs are skipped.
- Global state is tracked in `pipeline_state.json`.

## Config Notes

The example config is a 16-adapter panel (`QNLI`, `RTE`, `MRPC`, `SST-2` x 4 seeds).

Edit these first:

1. `output_root` (persistent disk path)
2. `base_model` (default is `microsoft/deberta-v3-base`)
3. Adapter list under `adapters`
4. `merge_eval.control_sample_size` based on eval budget

Supported tasks in this script are GLUE-style: `qnli`, `rte`, `mrpc`, `sst2`, `mnli`.

## Procedure for One-Block 24-30h Run

1. Start pod with persistent volume and enough disk for checkpoints/artifacts.
2. Launch pipeline in `tmux`.
3. Check `pipeline_state.json` every 30-60 minutes.
4. If interrupted, restart with the same command and config.
5. At completion, archive `output_root` and pull it locally.

Example archive step:

```bash
tar -czf runpod_gpu_pipeline_artifacts.tgz results/runpod_gpu_pipeline/deberta_v3_16_adapter_panel
```

## Budget Guidance

For a single A100 40GB block, a practical target is roughly **$40-60 total** by:

1. Keeping the run to one contiguous 24-30h block
2. Avoiding repeated restarts via resumable checkpoints
3. Limiting phase-3 controls to a stratified sample (not all cross-task pairs)

## Guardrails

- Keep policy/recommendation code unchanged during this run; this is an execution pipeline.
- Treat phase-3 as adjudication: retained + near-miss + stratified controls, not exhaustive all-pairs eval.
- Keep the exact config used for execution with artifacts for reproducibility.

## Decoder Study Script Pack

For the controlled decoder triage study spec, use the dedicated script pack:

- [`scripts/decoder_triage_study/phase0_validate_env.py`](../../scripts/decoder_triage_study/phase0_validate_env.py)
- [`scripts/decoder_triage_study/train_cohort.py`](../../scripts/decoder_triage_study/train_cohort.py)
- [`scripts/decoder_triage_study/run_pipeline.py`](../../scripts/decoder_triage_study/run_pipeline.py)
- [`scripts/decoder_triage_study/evaluate_merges.py`](../../scripts/decoder_triage_study/evaluate_merges.py)
- [`scripts/decoder_triage_study/analyze_results.py`](../../scripts/decoder_triage_study/analyze_results.py)
- [`scripts/decoder_triage_study/cohort_manifest.json`](../../scripts/decoder_triage_study/cohort_manifest.json)
