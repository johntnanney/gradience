# Note: Per-Module Geometry Protocol

## Metadata

- **Type:** protocol
- **Date:** 2026-03-26
- **Related notes:** n17 (within-layer protocol), n18 (within-layer findings), n19 (per-module subset)
- **Project:** Per-Module Geometry Program, Stage B

---

## Purpose

This protocol specifies the per-module geometry pilot study. It decomposes the within-layer analysis (n17–n18) from concatenated Q/K/V/O matrices into separate per-module comparisons.

**Motivation:** The aggregate within-layer analysis (n18) found MIXED-to-NEGATIVE results. When backbone was controlled, catastrophic cases were indistinguishable from safe controls. The concatenation of four structurally distinct attention modules (query, key, value, output) into a single matrix may have diluted a module-specific signal. This study tests whether the signal concentrates in a specific module.

---

## 1. Metrics

The same four metrics from n17, applied independently per module:

1. **Principal angle spectrum** — subspace alignment between adapter A's module-W and adapter B's module-W, via SVD of the inner product of truncated left singular vectors
2. **Top direction overlap** — absolute cosine between dominant (first) singular vectors per module
3. **Dimensionality ratio** — min/max ratio of effective ranks (Shannon entropy of normalized singular values)
4. **Directional conflict** — projection of both perturbation matrices into a shared top-4 subspace, normalized cosine distance

---

## 2. Input Matrices

For each layer and module, the input is W = lora_B @ lora_A, a (768 × 768) matrix. This is a square matrix (unlike the concatenated (768 × 3072) matrix used in n17–n18). SVD is applied with a 90% energy threshold to select the top-k components.

Module correspondence across backbones is defined in n19 §1.

---

## 3. Contrast Panel

Same 6 cases, 4 seed combos each, 24 total variants as n17. See n19 §2 for the full panel.

---

## 4. Critical Layer Selection

Identical to n17: top layers by combined norm mass ≥ 60% of total. Critical layers are precomputed from the per-layer analysis (n14–n15) and reused here.

---

## 5. Analysis Components

### 5.1 Group-level per-module summary

For each (group × module × metric), compute mean, std, min, max across variants. This produces a 3 × 4 × 4 summary (groups × modules × metrics) that reveals which modules differentiate the groups.

### 5.2 Module discrimination analysis

For the backbone-controlled comparison (RoBERTa cases only: CA-02 vs. SC-QMRB + SC-MSRB), compute Cohen's d and range overlap fraction for each (module × metric) pair. This identifies which module × metric combination best separates catastrophic from safe collision cases without the backbone confound.

### 5.3 Seed sensitivity per module

- **CA-01:** Compare worst variant (s42×s7, Δ=41.7%) vs. mild variant (s7×s7, Δ=12.7%) per module.
- **CA-02:** Compare toxic adapter variants (qnli_s42 pairs) vs. benign adapter variants (qnli_s7 pairs) per module.

---

## 6. Decision Criteria

**POSITIVE:** At least one module shows a clean group separation (Group 1 outside Group 2 range on at least one metric, Cohen's d > 1.5, overlap fraction < 0.2) that was not visible in the aggregate analysis.

**MIXED:** Some module-level differences exist but require interpretation or are confounded.

**NEGATIVE:** No module improves on the aggregate analysis. Escalate to output-space.

---

## 7. Structured Outputs

| File | Location |
|------|----------|
| Module metrics (all variants) | `sidecar/results/per_module_geometry/module_metrics.json` |
| Group module comparison | `sidecar/results/per_module_geometry/group_module_comparison.json` |
| Module discrimination | `sidecar/results/per_module_geometry/module_discrimination.json` |
| Seed sensitivity per module | `sidecar/results/per_module_geometry/seed_sensitivity_per_module.json` |
