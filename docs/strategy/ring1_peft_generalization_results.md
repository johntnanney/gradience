# Ring 1 PEFT Generalization Results

## Summary

Ring 1 tested whether Gradience's substrate generalizes from LoRA to at least one
additional PEFT artifact class.  The candidate was **LoHa** (Low-Rank Hadamard Product),
trained on distilbert-base-uncased for SST-2 binary sentiment classification.

**Result: positive.**  The full Gradience workflow -- single-adapter audit, pairwise
merge comparison, and inventory preflight -- ran on LoHa adapters using a ~160-line
extraction shim.  Zero core code was modified.

---

## A. What generalized cleanly

The following components consumed LoHa-derived inputs with no modification:

1. **Spectral measurement layer** -- `low_rank_singular_values()`, `_effective_rank()`,
   `_energy_rank()`, and `low_rank_stable_rank()` all operated on the shimmed factor
   pairs identically to native LoRA.  All 6 audit runs (3 adapters x 2 modes) succeeded.

2. **Pairwise subspace comparison** -- `compute_subspace_metrics()` and `diagnose_pair()`
   produced valid verdicts for all 3 LoHa adapter pairs.  All pairs returned `SAFE`
   verdicts with low pair-risk and `linear` strategy recommendations, consistent with
   same-task adapters on the same dataset.

3. **Merge-audit CLI** -- `gradience merge-audit --qa-report --emit-report` ran
   unmodified on the shimmed adapter directories.

4. **Inventory pipeline** -- `build_inventory_summary()`, `build_action_plan()`,
   `format_inventory_summary()`, `format_action_plan()`, and `emit_run_bundle()` all
   consumed LoHa-derived reports and QA artifacts without modification.

5. **Report vocabulary** -- The standard terminology (pair risk, recommended strategy,
   dominant issue, action plan zones, evidence provenance) remained meaningful and did
   not require LoHa-specific wording.

6. **QA/eligibility logic** -- The eligibility classification (`unknown_no_behavioral_eval`
   due to absent eval data) worked identically to the LoRA path.

---

## B. What required thin adaptation

One component required a shim: **weight-key extraction**.

The shim (`experiments/peft_ring1/loha_shim.py`, ~160 lines) performs:

- **Key detection**: regex on `.hada_w1_a` keys to discover LoHa modules
- **Factor-level mode**: renames `(hada_w1_a, hada_w1_b)` and `(hada_w2_a, hada_w2_b)`
  pairs to `.lora_A.default.weight` / `.lora_B.default.weight` naming, producing two
  pseudo-LoRA layers per LoHa module
- **Materialized mode**: computes `W_delta = (w1_a @ w1_b) * (w2_a @ w2_b)`, then
  SVD-factors the result into synthetic LoRA-format `(A, B)` pairs
- **Config rewrite**: writes `adapter_config.json` with `peft_type: LORA` and the
  original `r` and `alpha` values

The shim is the **only** non-core code required.  It writes a temporary directory that
the existing audit and merge pipelines consume without modification.

---

## C. What remained LoRA-specific

1. **`_iter_lora_pairs()`** -- Hardcoded to `.lora_A.` / `.lora_B.` key patterns.
   Cannot discover LoHa, LoKr, or IA3 keys directly.  The shim works around this
   by rewriting keys to LoRA format.

2. **`LoRALayerAudit` field names** -- Fields like `a_key`, `b_key`, `alpha`, and
   `scale = alpha/r` carry LoRA semantics.  The shim produces values that fit these
   fields but the names remain LoRA-specific.

3. **`rank_nominal` in QA artifact** -- Assumes a single integer rank.  This maps
   naturally to LoHa (which also has an `r` parameter) but would not fit IA3 (which
   has no rank concept).

4. **Merge executor** -- Writes `.lora_A.weight` / `.lora_B.weight` keys.  Actual
   merged-weight reconstruction for LoHa would require Hadamard-aware logic.

---

## D. Measurement results

### Stage B: Single-adapter audit

| Adapter | Mode | Layers | Mean Utilization | Mean Stable Rank | Energy Rank 90 |
|---------|------|--------|-----------------|-----------------|----------------|
| loha_r4 | factor | 24 | 0.649 | 2.60 | 4.0 |
| loha_r4 | materialized | 12 | 0.564 | 2.26 | 4.0 |
| loha_r8 | factor | 24 | 0.521 | 4.17 | 7.0 |
| loha_r8 | materialized | 12 | 0.392 | 3.14 | 7.0 |
| loha_r16 | factor | 24 | 0.393 | 6.29 | 13.0 |
| loha_r16 | materialized | 12 | 0.236 | 3.78 | 13.0 |

Observations:
- Factor mode shows 24 layers (12 modules x 2 Hadamard factors); materialized shows 12
- Utilization decreases with rank (expected: more capacity, less used)
- Stable rank scales sub-linearly with nominal rank
- Materialized utilization is lower: the Hadamard product concentrates energy more than
  individual factors, so fewer effective dimensions survive

### Stage C: Pairwise comparison + inventory

All 3 pairs (r4xr8, r4xr16, r8xr16) produced:
- pair_risk: low
- recommended_strategy: linear
- dominant_issue: none

This is consistent with same-task adapters on the same dataset with the same target
modules.  The inventory pipeline correctly classified this as a "same-task, mostly
confirmatory" inventory with source QA as the binding constraint (due to absent
behavioral eval data).

---

## E. Practical product implication

**Gradience can credibly claim broader PEFT support in a narrow band.**

Specifically:
- The measurement layer, pairwise comparison, and inventory triage generalize cleanly
  to any PEFT type that exposes low-rank factor pairs (LoHa, and likely LoKr)
- The required adaptation is extraction-level only: a ~160-line shim per artifact class
- The core spectral math, QA logic, and report vocabulary are artifact-agnostic
- Merge execution (actually reconstructing weights) remains LoRA-specific

This means Gradience is best described as:
> **A LoRA-native tool with a PEFT-general audit and triage substrate, available to
> other low-rank PEFT methods via thin extraction shims.**

IA3 and other non-low-rank methods would require a different measurement paradigm
(not just a shim) and are not covered by this result.

---

## F. Limitations

- CPU-only, small encoder model (distilbert-base-uncased), single dataset (SST-2)
- No behavioral eval was run (all adapters classified as unknown_no_behavioral_eval)
- Only 3 adapters, all same-task -- no cross-task or cross-family scenarios tested
- Merge execution was not tested (only audit and comparison)
- LoKr was not tested (rated as viable in the support matrix but deferred)
