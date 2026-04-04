# Over-Accumulation Refinement Summary

## Status
- Decision: **keep_exploratory**
- Evidence strength: **mixed / bounded**

## Key Evidence
- OA-v2 strict-naive 30-pair cross-check completed (target cohort gate passed: 30/30 in requested window).
- Overall rank relation improved vs OA-v1:
  - Spearman(delta, OA-v1 max) = `0.1791`
  - Spearman(delta, OA-v2 max) = `0.2513`
- Intended high-overlap / low-conflict slice remained unstable (`n=9`):
  - Spearman(delta, OA-v1 max) = `0.1604`
  - Spearman(delta, OA-v2 max) = `-0.0928`
- Promotion gate failed (rules 1/2/3 failed; rule 4 passed).
- Source-quality stratification remained a dominant confound axis in this cohort.

## Interpretation
- OA-v2 remains a plausible structural diagnostic but is not policy-ready.
- Current evidence does not justify threshold or advisory-policy promotion.
- OA-v1 remains authoritative for merge policy/reporting; OA-v2 remains experimental companion analysis.
- Intended-slice feasibility is currently constrained (`9` pairs available at `overlap>=0.25`, `conflict<=0.10`), so a regime-pure confirmation rerun requires either inventory expansion or a pre-registered slice relaxation.

## Canonical Artifacts
- Failure anatomy (no-rerun decomposition):
  - `/Users/john/code/gradience/field_trials/analytical_spectral_geometry/failure_anatomy_oa_v2_30_40_r1.json`
  - `/Users/john/code/gradience/field_trials/analytical_spectral_geometry/failure_anatomy_oa_v2_30_40_r1.md`
- Gate report:
  - `/Users/john/code/gradience/field_trials/analytical_spectral_geometry/gate_report_oa_v2_30_40_r1.json`
  - `/Users/john/code/gradience/field_trials/analytical_spectral_geometry/gate_report_oa_v2_30_40_r1.md`
