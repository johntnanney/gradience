# Note: Per-Layer Comparison Protocol

## Metadata

- **Type:** protocol
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related notes:** n13 (artifact inventory), n12 (artifact mining inventory), n06 (program statement)
- **Project:** Phase 3, Project G — Per-Layer Structural Analysis

---

## Purpose

This note defines the minimal per-layer comparison protocol for the CPU-only structural analysis of saved LoRA adapter weights. It specifies four metrics, explains what each measures, defines how to compute them, and establishes the comparison logic that will determine whether catastrophic cross-task anchors show a different per-layer structural footprint than stable contrast pairs.

---

## 1. Study Question

**Do catastrophic cross-task anchors show stronger per-layer concentration or sharper per-layer divergence than stable contrast pairs?**

This is a descriptive question, not a causal one. A positive answer would provide evidence for the thresholded subspace interference hypothesis by showing that catastrophic outcomes co-occur with distinctive per-layer geometry. A negative answer would constrain mechanism: if per-layer structure does not differentiate catastrophic from stable pairs, the operative variable is not layer-localized.

---

## 2. Four Metrics

### 2.1 Norm Mass

**What it measures:** How much adaptation weight each layer carries, expressed as a fraction of the total.

**Definition:** For a single adapter on a given backbone, norm mass at layer *l* is:

```
norm_mass(l) = Σ_m ||W_m^l||_F / Σ_l' Σ_m ||W_m^{l'}||_F
```

where *m* ranges over the four target modules (query, key, value, output) and ||·||_F is the Frobenius norm. The sum in the denominator runs over all layers.

**Computed from:** The combined LoRA weight at each module: W = lora_B @ lora_A (shape: hidden_dim × hidden_dim). We take the Frobenius norm of this product, not of lora_A and lora_B separately, because the product represents the actual perturbation to the base model.

**Output:** A vector of length *n_layers* that sums to 1.0 for each adapter. This is a distribution over layers.

**Why it matters:** If catastrophic adapters concentrate their adaptation in a few layers while stable adapters spread it evenly, that is a structural fingerprint of instability.

### 2.2 Pair Divergence

**What it measures:** How differently two adapters distribute their norm mass across layers.

**Definition:** For two adapters *A* and *B* with norm mass vectors *p* and *q*:

```
pair_divergence = JS(p, q) = 0.5 × KL(p || m) + 0.5 × KL(q || m)
```

where *m* = 0.5(p + q) and KL is the Kullback-Leibler divergence. This is the Jensen-Shannon divergence, which is symmetric, bounded in [0, 1] (when using log base 2), and well-defined even when entries are zero (with the convention 0 log 0 = 0).

**Output:** A single scalar per adapter pair. Higher values mean the two adapters concentrate their adaptation in different layers.

**Why it matters:** If catastrophic pairs show high pair divergence (adapters pulling in different layer-directions) while stable pairs show low divergence, that suggests the catastrophic mechanism involves layer-level competition. This is the most direct test of the "different layers, different fights" hypothesis.

### 2.3 Concentration Index

**What it measures:** How peaked or flat an adapter's norm mass distribution is — a single-adapter property, not a pair property.

**Definition:** The Gini coefficient of the norm mass vector:

```
concentration = Gini(norm_mass)
```

where Gini ranges from 0 (perfectly uniform) to 1 (all mass in one layer). Specifically:

```
Gini = (Σ_i Σ_j |x_i - x_j|) / (2 * n * Σ_i x_i)
```

for norm mass values x_1, ..., x_n.

**Output:** A single scalar per adapter, in [0, 1].

**Why it matters:** High concentration means an adapter's learned perturbation is localized to a few layers. If adapters involved in catastrophic pairs tend to have higher concentration than those in stable pairs, it suggests that layer-localized adaptation is a risk factor — the adapter has "put all its eggs in a few baskets," making those baskets vulnerable to interference.

### 2.4 Alignment Proxy

**What it measures:** Whether two adapters' largest perturbations land in the same layers or different layers.

**Definition:** The Spearman rank correlation between two adapters' norm mass vectors:

```
alignment = ρ_Spearman(norm_mass_A, norm_mass_B)
```

**Output:** A scalar in [-1, 1] per adapter pair. +1 means both adapters concentrate in the same layers (potential for collision). -1 means they concentrate in opposite layers (complementary). 0 means no systematic relationship.

**Why it matters:** The thresholded subspace interference hypothesis predicts that catastrophic pairs collide in specific layers. High alignment (both adapters heavy in the same layers) combined with high pair divergence (the *amount* of weight differs) would indicate "same territory, different strength" — a concrete geometric picture of interference. Low alignment would suggest the adapters avoid each other's territory, which should be safer.

---

## 3. Computation Procedure

### Step 1 — Extract per-layer norms

For each of the 16 adapters:

1. Load the safetensors file.
2. For each layer *l* and each target module *m*, extract lora_A and lora_B.
3. Compute the LoRA product: W = lora_B @ lora_A.
4. Compute ||W||_F (Frobenius norm).
5. Store the result in a (layers × modules) matrix.

### Step 2 — Compute per-adapter metrics

For each adapter:

1. Sum across modules to get the per-layer total norm: norm_total(l) = Σ_m ||W_m^l||_F.
2. Normalize to get norm_mass(l) = norm_total(l) / Σ_l' norm_total(l').
3. Compute concentration = Gini(norm_mass).

### Step 3 — Compute per-pair metrics

For each pair in the contrast panel:

1. Take the two adapters' norm mass vectors.
2. Compute pair_divergence = JS(norm_mass_A, norm_mass_B).
3. Compute alignment = ρ_Spearman(norm_mass_A, norm_mass_B).

### Step 4 — Compare across groups

Assemble per-pair metrics for Groups A, B, and C. The core comparisons:

- **Pair divergence:** Group A vs. Group C (catastrophic vs. stable cross-task). Do catastrophic pairs show higher JS divergence?
- **Concentration:** Adapters in Group A vs. adapters in Group C. Do catastrophic-participating adapters have higher Gini?
- **Alignment:** Group A vs. Group C. Do catastrophic pairs show higher or lower rank correlation?
- **Group B baseline:** All metrics for same-task pairs, establishing the noise floor.

---

## 4. Interpretation Framework

### Positive outcome

The per-layer analysis produces a positive outcome if **at least one metric cleanly separates Group A from Group C**. "Cleanly" means: the worst Group A value exceeds the best Group C value for that metric (no overlap), or the group means differ by more than 2× the pooled standard deviation.

A positive outcome supports the thresholded subspace interference hypothesis by demonstrating that catastrophic outcomes have a per-layer structural correlate.

### Mixed outcome

The per-layer analysis produces a mixed outcome if metrics show **trends in the expected direction but with overlap between groups**. This would suggest that per-layer structure is informative but not sufficient — other variables (per-module structure, higher-order interactions) may be needed.

### Negative outcome

The per-layer analysis produces a negative outcome if **no metric differentiates Group A from Group C**. This would constrain the mechanism: whatever drives catastrophic interference, it is not visible in per-layer norm distributions. The search would need to move to per-module or subspace-angle analyses.

---

## 5. Output Specification

### Structured outputs (JSON)

| File | Contents |
|------|----------|
| `per_layer_norms.json` | Per-adapter, per-layer, per-module Frobenius norms |
| `per_layer_metrics.json` | Per-adapter norm_mass and concentration; per-pair divergence and alignment |
| `group_comparison.json` | Group-level statistics (means, ranges) for all four metrics |

All outputs go to `sidecar/results/per_layer_analysis/`.

### Figures

| File | Contents |
|------|----------|
| `norm_mass_profiles.svg` | Norm mass distribution across layers for all adapters, colored by group |
| `group_comparison.svg` | Box/strip plots of pair divergence and concentration by group |

Figures go to `sidecar/figures/`.

### Analysis note

The findings note (n15) will classify the outcome as positive, mixed, or negative and record the specific metric values that support the classification.

---

## 6. Scope Boundaries

This protocol covers **per-layer norm-based analysis only**. The following are explicitly out of scope:

- **Per-module decomposition:** Examining whether specific attention components (e.g., value vs. query) drive the signal. This is a natural follow-up if per-layer signals are found.
- **SVD-based subspace analysis:** Computing principal angles between adapter subspaces at each layer. Higher-value but computationally more involved; deferred to a potential follow-up.
- **Merged model analysis:** All metrics are computed from source adapters, not from merged weights. This is intentional — source adapter structure is a *pre-merge* predictor, which is what the instability program needs for a structural predictor (n06 §4, criterion 2).
- **Statistical testing:** With 2 cases per group on each backbone, formal hypothesis tests are not meaningful. The analysis is descriptive and pattern-oriented, not inferential.

---

## 7. Freeze Conditions

Per the implementation spec, the following are frozen:

- The four metrics defined in §2 are the complete metric set. No additional metrics should be added during execution.
- The contrast panel defined in n13 §4 is the complete case set. No additional cases should be added.
- The interpretation framework in §4 is pre-registered. Outcomes should be classified before any post-hoc exploration.
