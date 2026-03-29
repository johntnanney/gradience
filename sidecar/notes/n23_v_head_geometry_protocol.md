# Note: V-Module Head-Level Geometry Protocol

## Metadata

- **Type:** protocol
- **Date:** 2026-03-26
- **Related notes:** n22 (head panel), n20 (per-module protocol), n21 (per-module findings)
- **Project:** V-Module Head-Level Program, Stage B

---

## Purpose

This protocol specifies the head-level V-module geometry pilot study. It decomposes the per-module V analysis (n21) into per-head comparisons, computing metrics separately for each of the 12 attention heads within the V module at critical layers.

**Motivation:** The per-module analysis (n21) achieved a POSITIVE outcome: V-module dimensionality ratio cleanly separates catastrophic from safe collision (d=3.36, zero overlap). However, CA-01 seed sensitivity remains unexplained at per-module resolution (all deltas < 0.07). Preliminary head-level reconnaissance (n22 §4) found individual head deltas up to 0.23 on dimensionality ratio — 33× larger than the module aggregate — confirming the signal exists at head resolution but is washed out by cross-head averaging.

---

## 1. Head-Level Metrics

The same four metrics from the per-module analysis (n20), adapted for head-level (64, 768) matrices:

### 1.1 Per-head effective rank

Effective rank via Shannon entropy of normalized singular values, computed on W_head_h = W[h*64:(h+1)*64, :]. Maximum possible effective rank is 64 (the row dimension), not 768.

### 1.2 Per-head dimensionality ratio

min(eff_rank_a_h, eff_rank_b_h) / max(eff_rank_a_h, eff_rank_b_h), where eff_rank_a_h and eff_rank_b_h are the effective ranks of adapter A and B at head h. This is the primary metric, given its d=3.36 separation at module level.

### 1.3 Per-head top direction overlap

Absolute cosine between the first left singular vectors of the two adapters' head matrices. Measures alignment of the dominant perturbation direction at each head.

### 1.4 Per-head directional conflict

Projection of both head matrices into a shared top-k subspace (k=4), normalized cosine distance. Measures opposing perturbation directions within each head's subspace.

---

## 2. Summary Descriptors (Computed per Variant)

Beyond per-head metrics, the following summary descriptors aggregate head-level information into variant-level signals:

### 2.1 Head concentration index

The Gini coefficient of head-level norm masses within the V module at each critical layer. A high Gini indicates that a few heads dominate the V-module perturbation; a low Gini indicates uniform distribution.

### 2.2 Worst-head dimensionality ratio

The minimum dimensionality ratio across all heads at each critical layer. If the catastrophic signal concentrates at specific heads, the worst-head ratio should separate catastrophic from safe more cleanly than the module mean.

### 2.3 Head mismatch spread

The standard deviation of per-head dimensionality ratios across the 12 heads. Higher spread means more heterogeneous head-level geometry, which could indicate that some heads are in conflict while others are compatible.

### 2.4 Seed-variant head drift

For CA-01 and CA-02 seed variants: the maximum absolute delta in any single head's dimensionality ratio or top direction overlap between worst and mild (or toxic and benign) seed variants. This is the target metric for CA-01 — the module-level max delta was 0.07; head-level should be substantially larger.

---

## 3. Input Matrices

For each (layer, head, adapter), the input is:

```
W_head = (lora_B @ lora_A)[h*64 : (h+1)*64, :]    # shape: (64, 768)
```

SVD is computed with a 90% energy threshold to select the top-k components, as in the per-module analysis. The maximum number of components is 64 (not 768), which reduces the effective subspace dimensionality relative to the module-level analysis.

---

## 4. Contrast Panel

Same 6 cases, 4 seed combos each, 24 total variants as the per-module analysis (n19/n22). See n22 §3 for the full panel.

---

## 5. Analysis Components

### 5.1 Per-head metrics table

For each (variant × critical_layer × head × metric), compute and store the raw metric value. This produces the complete head-level dataset.

### 5.2 Group-level head summary

For each (group × head × metric), compute mean, std, min, max across variants at critical layers. This reveals which heads differentiate the groups.

### 5.3 Head discrimination analysis (backbone-controlled)

For the backbone-controlled comparison (RoBERTa cases: CA-02 vs. SC-QMRB + SC-MSRB), compute Cohen's d and range overlap for each (head × metric) pair, plus the summary descriptors (worst-head dim ratio, head mismatch spread). This identifies whether head-level resolution improves on the module-level d=3.36.

### 5.4 Seed sensitivity per head

- **CA-01:** Compare worst variant (s42×s7) vs. mild variant (s7×s7) per head at each critical layer. Report the heads with the largest deltas and their layer locations.
- **CA-02:** Compare toxic adapter variants (qnli_s42 pairs) vs. benign adapter variants (qnli_s7 pairs) per head. Report whether the V-module toxic/benign difference concentrates at specific heads.

### 5.5 Head concentration analysis

Compute head concentration index (Gini) and relate it to group membership. Test whether catastrophic variants show more concentrated or more dispersed head-level perturbations.

---

## 6. Decision Criteria

**POSITIVE:** At least one of:
- (a) CA-01 seed sensitivity localizes: ≥1 head at ≥1 critical layer shows Δ_DR ≥ 0.15 or Δ_OV ≥ 0.10 between worst and mild variants. (The module-level max was 0.07.)
- (b) Worst-head dimensionality ratio improves discrimination: d > 3.36 or overlap fraction < 0.05 on the backbone-controlled comparison.
- (c) Head mismatch spread separates groups: catastrophic variants show systematically different head-level heterogeneity than safe collision controls.

**MIXED:** Some head-level differences exist, but:
- CA-01 deltas are present but below 0.15, or
- Head discrimination does not clearly improve on module level, or
- The pattern is inconsistent across layers.

**NEGATIVE:** No head-level signal improves on the module-level analysis. The seed sensitivity variable is below head resolution. Escalate to output-space analysis.

---

## 7. Structured Outputs

| File | Location |
|------|----------|
| Head metrics (all variants) | `sidecar/results/head_level_v/head_metrics.json` |
| Group head comparison | `sidecar/results/head_level_v/group_head_comparison.json` |
| Head discrimination | `sidecar/results/head_level_v/head_discrimination.json` |
| Seed sensitivity per head | `sidecar/results/head_level_v/seed_sensitivity_per_head.json` |
| Head summary descriptors | `sidecar/results/head_level_v/head_summary_descriptors.json` |
| Head discrimination heatmap | `sidecar/figures/head_level_discrimination.svg` |
| CA-01 seed sensitivity per head | `sidecar/figures/head_level_ca01_seed_sensitivity.svg` |
| CA-02 seed sensitivity per head | `sidecar/figures/head_level_ca02_seed_sensitivity.svg` |
| Worst-head spotlight | `sidecar/figures/head_level_worst_head_spotlight.svg` |
