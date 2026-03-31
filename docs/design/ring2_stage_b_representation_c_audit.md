# Ring 2 Stage B — Representation C Audit and Pairwise Validation

Generated: March 30, 2026.

Primary artifacts:
- `experiments/ring2_checkpoint_delta/run_stage_b.py`
- `experiments/ring2_checkpoint_delta/stage_b_representation_c_results.json`
- `experiments/ring2_checkpoint_delta/stage_b_representation_c_results.md`
- `experiments/ring2_checkpoint_delta/results/stage_b_pair_reports/*.json`

## Stage B Goal

Validate that the Stage A-selected checkpoint-delta object (Representation C: layerwise summary) supports:
- single-artifact audit
- pairwise comparison

in a fully CPU-only workflow, without merge execution.

## Inputs

Backbone and panel (from Stage A):
- base model: `distilbert-base-uncased`
- checkpoints: `sst2_s42`, `sst2_s123`, `mrpc_s42`, `qnli_s42`

Layer scope (fixed from Stage A):
- attention projections (`q_lin`, `k_lin`, `v_lin`, `out_lin`)
- FFN dense layers (`lin1`, `lin2`)
- classifier/pre-classifier layers

## Method Notes

Stage B uses Representation C summaries emitted from Stage A artifacts when available, and can recompute summaries from checkpoints if needed.

Pairwise scoring combines:
- summary-profile cosine similarity
- mean absolute deltas for `energy_at_8`, `stable_rank`, `effective_rank`, and SV-decay

This yields a compatibility score with risk buckets (`low` / `medium` / `high`) and per-pair dominant divergence layers.

## Important Integrity Fix Applied

A methodological issue was identified and corrected before final Stage B outputs:

- Sequence-classification heads are randomly initialized when loading the base encoder checkpoint.
- If base initialization is not seed-matched to the training run, classifier deltas are polluted by random init drift.

Fix:
- Stage extraction now supports seed-aware base head initialization.
- Stage A and Stage B outputs were regenerated using checkpoint seed-matched extraction.

This materially reduced false classifier-heavy artifacts and improved audit validity.

## Results Summary

Single-artifact audit:
- `healthy`: 3 / 4 checkpoints
- `review`: 1 / 4 checkpoints (`qnli_s42`)
- dominant `qnli_s42` flags:
  - `diffuse_delta_spectrum`
  - `high_effective_rank`

Pairwise comparison:
- total pairs: 6
- low-risk: 0
- medium-risk: 3
- high-risk: 3
- same-task pair:
  - `sst2_s42::sst2_s123` scored `medium` with higher compatibility than cross-task pairs
- highest-risk pairs involve `qnli_s42` cross-task comparisons

Compatibility separation:
- same-task mean compatibility: `0.8922`
- cross-task mean compatibility: `0.7043`
- same-minus-cross gap: `0.1879`

## Stage B Outcome

- stage_c_readiness: `medium`
- recommendation: `advance_with_caution`
- decision: `advance_to_stage_c`

Interpretation:
- Representation C is operationally viable for both single-artifact audit and pairwise comparison on CPU.
- Pairwise signal is meaningful (same-task vs cross-task separation), but risk concentration around `qnli_s42` indicates Stage C should carry forward pair-level guardrails and dominant divergence-layer analysis.

## Suggested Stage C Entry Conditions

When moving to Stage C, prioritize:
- guardrail-aware handling of high-risk pairs
- explicit logging of dominant divergence layers for any recommended action
- preserving Representation A spot-check capability for faithfulness verification on contentious pairs
