# Runbook — Controlled Decoder Fingerprinting / Merge Triage

**Study ID:** `decoder_fingerprinting_merge_triage`  
**Target budget:** ~8-12 GPU hours  
**Primary source spec:** [`docs/plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](../../../docs/plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md)

## Objective

Under matched controls, separate architecture-led and task-led spectral structure for decoder adapters and test how that structure relates to merge triage behavior.

## Locked Design

1. Architectures: minimum 2 (preferred 3): Llama, Mistral, Qwen
2. Tasks: 3 matched tasks per architecture
3. Minimum cohort: 18 artifacts (preferred 27)
4. Fixed artifact class: LoRA/low-rank adapters
5. Confound-first analysis: rank/config controls before attribution claims

## Required Outputs

1. `field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/cohort_definition.json`
2. `field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/artifact_manifest.json`
3. `field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/fingerprint_table.json`
4. `field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/architecture_task_decomposition.json`
5. `field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/study_memo.md`

## Execution Checklist

- [ ] Architecture/task cohort locked before runs
- [ ] Matched training/eval protocol documented
- [ ] Fingerprint extraction complete for full cohort
- [ ] Architecture-vs-task decomposition complete (raw + controlled)
- [ ] Census bridge note written (what persisted vs attenuated)

## Gate to Proceed

Proceed to PG3 once decomposition outputs are complete and outcome labels are available for confidence-validation analysis.

