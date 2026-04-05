# Runbook — Verdict-Confidence Validation

**Study ID:** `verdict_confidence_validation`  
**Target budget:** ~2-4 GPU hours (+ analysis)  
**Primary references:**  
- [`docs/plans/2026-04-04-verdict-boundary-stress-test-spec.md`](../../../docs/plans/2026-04-04-verdict-boundary-stress-test-spec.md)  
- [`scripts/verdict_boundary_stress_test.py`](../../../scripts/verdict_boundary_stress_test.py)

## Objective

Test whether confidence stratification materially improves recommendation quality once outcome data from GPU proving grounds is available.

## Inputs

1. Outcome-labeled merge data from PG1 (DeBERTa adjudication)
2. Outcome-labeled merge data from PG2 (controlled decoder study)
3. Existing encoder stress-test profiles (CPU baseline)

## Required Outputs

1. `field_trials/gpu_return_packet/verdict_confidence_validation/confidence_validation_dataset.json`
2. `field_trials/gpu_return_packet/verdict_confidence_validation/confidence_validation_results.json`
3. `field_trials/gpu_return_packet/verdict_confidence_validation/confidence_validation_report.md`
4. `field_trials/gpu_return_packet/verdict_confidence_validation/policy_recommendation.md`

## Required Analyses

1. Baseline vs confidence-stratified ranking quality
2. High-confidence precision and low-confidence risk capture
3. Same-task vs cross-task stratified behavior
4. Branch-5 heterogeneity impact on confidence calibration
5. Stability check under leave-one-out / bootstrap

## Execution Checklist

- [ ] GPU outcome labels merged into one validation dataset
- [ ] Confidence buckets evaluated against real outcomes
- [ ] Improvement and failure cases documented
- [ ] Recommendation labeled `promote`, `bounded_keep`, or `hold`

## Policy Guardrail

No threshold or policy-code update from this runbook alone.  
Promotion requires explicit closeout approval in the packet summary memo.

