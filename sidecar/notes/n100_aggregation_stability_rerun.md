# n100 -- Aggregation Stability Check: Perturbed Rerun

**Type:** substudy findings note
**Date:** 2026-03-31
**Program:** Route2 Substudy 2 -- Aggregation-Sensitive Compatibility Stability Check
**Stage:** C
**Depends on:** n98, n99
**Status:** complete

---

## Objective

Re-run the original aggregation comparison logic on the perturbed panel and check whether the qualitative aggregation story survives.

---

## Method

Applied the same four aggregation families to all 12 perturbed cases:

- worst-case
- distributional
- QA-dominant
- QA-gated distributional

No aggregation definitions were changed.

---

## Results

Agreement distribution:

- full_agreement: 2
- partial_agreement: 8
- strong_divergence: 2

Key rerun observations:

1. Aggregation seam remains visible (10/12 aggregation-sensitive).
2. Worst-case still collapses routing-facing gradation that distributional preserves.
3. QA-dominant still produces operational overrides under blocked/mixed evidence.
4. QA-gated distributional still preserves both evidence constraints and structural gradation.
5. Both invariant and sensitive case types remain present.

---

## Interpretation

The perturbation did not collapse the original qualitative structure. The same classes of divergence reappear with nearby replacements, indicating local robustness of the aggregation-sensitive interpretation.

---

## Outputs

- sidecar/results/route2_stability/aggregation/perturbed_aggregation_comparison.json
- sidecar/results/route2_stability/aggregation/perturbed_aggregation_comparison.md
- sidecar/figures/aggregation_stability_comparison.svg
- sidecar/notes/n100_aggregation_stability_rerun.md (this note)
