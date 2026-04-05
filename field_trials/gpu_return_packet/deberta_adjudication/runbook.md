# Runbook — DeBERTa Adjudication

**Study ID:** `deberta_adjudication`  
**Target budget:** ~3 GPU hours  
**Primary source spec:** [`sidecar/notes/n07_deberta_adjudication_protocol.md`](../../../sidecar/notes/n07_deberta_adjudication_protocol.md)

## Objective

Decide whether the current encoder-side mechanism account transfers to a third backbone.

## Locked Design

1. Backbone: `microsoft/deberta-v3-base`
2. Adapters: 8 total (`QNLI`, `RTE`, `MRPC`, `SST-2` x 2 seeds)
3. Merge panel: 28 pairs
4. Evaluation: source-task matched evaluation per merged pair
5. Predictions: pre-registered A-E from `n07`

## Required Outputs

1. `field_trials/gpu_return_packet/deberta_adjudication/source_scores.json`
2. `field_trials/gpu_return_packet/deberta_adjudication/adjudication_results.json`
3. `field_trials/gpu_return_packet/deberta_adjudication/prediction_adjudication.md`
4. `field_trials/gpu_return_packet/deberta_adjudication/per_module_summary.md`
5. `field_trials/gpu_return_packet/deberta_adjudication/per_head_summary.md` (if E testable)

## Execution Checklist

- [ ] Training/eval config locked and saved
- [ ] 8 source adapters trained and scored
- [ ] 28 merges evaluated
- [ ] Prediction A-E outcomes recorded as `pass`, `partial`, `fail`, or `untestable`
- [ ] DeBERTa implication note written (backbone-general vs bounded)

## Gate to Proceed

Proceed to PG2 regardless of pass/fail, but do not claim mechanism generality unless the pre-registered gate outcome supports it.

