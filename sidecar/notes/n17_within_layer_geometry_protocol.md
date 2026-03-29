# Note: Within-Layer Geometry Protocol

## Metadata

- **Type:** protocol
- **Date:** 2026-03-26
- **Related notes:** n16 (collision subset), n15 (per-layer findings), n14 (per-layer protocol)
- **Project:** Within-Layer Collision Program, Stage B

---

## Purpose

This note defines the minimal within-layer geometry study. It specifies four metrics, the contrast panel, the layer selection strategy, and the interpretation framework. The central question:

> Inside the collision subset, what within-layer geometric condition separates catastrophic from safe or merely unstable outcomes?

---

## 1. Contrast Panel

### Group 1 — Catastrophic cases

| Case | Pair | Backbone | Alignment | Worst Δ | Role |
|------|------|----------|----------:|--------:|------|
| CA-01 | QNLI×MRPC | DistilBERT | 0.86 | 41.7% | Core collision catastrophe |
| CA-02 | QNLI×SST-2 | RoBERTa | 0.66 | 27.2% | Moderate-alignment catastrophe |

### Group 2 — Safe collision controls

| Case | Pair | Backbone | Alignment | Worst Δ | Role |
|------|------|----------|----------:|--------:|------|
| SC-QMRB | QNLI×MRPC | RoBERTa | 0.80 | 1.7% | Same pair as CA-01, collision but safe |
| SC-MSRB | MRPC×SST-2 | RoBERTa | 0.89 | 15.0% | Highest cross-task alignment, non-catastrophic |

### Group 3 — Non-collision contrasts

| Case | Pair | Backbone | Alignment | Worst Δ | Role |
|------|------|----------|----------:|--------:|------|
| NC-QSDB | QNLI×SST-2 | DistilBERT | 0.17 | 11.0% | Same pair as CA-02, no collision, not catastrophic |
| NC-RMRB | RTE×MRPC | RoBERTa | 0.63 | 8.3% | Low alignment, low severity |

### Key comparison

The highest-information comparison is CA-01 vs. SC-QMRB: QNLI×MRPC on DistilBERT (catastrophic, ρ=0.86) vs. QNLI×MRPC on RoBERTa (safe, ρ=0.80). Same task pair, both colliding, different outcome. If within-layer metrics differentiate these two cases, the evidence for within-layer incompatibility as the operative variable is strong.

---

## 2. Layer Selection Strategy

Not every layer is equally informative. The per-layer analysis (n15) showed that adapters concentrate their norm mass in specific layers. The within-layer study should focus on **critical layers** — those where both adapters in a pair carry high norm mass.

### Selection rule

For each pair, identify the top-*k* layers by combined norm mass: for each layer, compute `norm_mass_A(l) + norm_mass_B(l)`, then take the top layers that collectively account for ≥ 60% of total combined norm mass.

This focuses the analysis on the layers where collision actually occurs, rather than diluting the signal with low-mass layers where any interaction is negligible.

For reporting, compute all metrics on all layers, but highlight the critical-layer subset in the findings.

---

## 3. Four Within-Layer Metrics

For each layer, each adapter produces a LoRA perturbation: W = lora_B @ lora_A (shape: hidden_dim × hidden_dim). The within-layer metrics operate on these W matrices (or their SVD decompositions) for pairs of adapters at matched layers.

### 3.1 Principal Angle Spectrum

**What it measures:** The geometric relationship between two adapters' perturbation subspaces at a given layer.

**Definition:** Compute the SVD of each adapter's per-layer W: W_A = U_A Σ_A V_A^T. Take the top-*k* left singular vectors from each (where *k* is the number of singular values capturing ≥ 90% of energy). Compute the principal angles between the two subspaces:

```
cos(θ_i) = σ_i(U_A_k^T @ U_B_k)
```

where σ_i are the singular values of the inner product matrix.

**Output per layer:** A vector of cosine values in [0, 1]. Cosine near 1 means overlapping subspaces; near 0 means orthogonal.

**Summary statistics:** Report the mean and minimum principal cosine per layer. The minimum cosine indicates the most orthogonal direction — if this is large, the subspaces are tightly aligned.

### 3.2 Top Singular Direction Overlap

**What it measures:** Whether the dominant learned direction in one adapter's perturbation overlaps with the dominant direction in the other.

**Definition:** For each adapter's per-layer W, take the top-1 left singular vector (u_1). Compute:

```
top_overlap = |u_1_A · u_1_B|
```

**Output per layer:** A scalar in [0, 1]. Value of 1 means the dominant perturbation directions are identical (or opposite); 0 means orthogonal.

**Why it matters:** If catastrophic pairs show high top-direction overlap at critical layers while safe pairs show low overlap, that identifies a specific geometric condition for destructive interference: both adapters push hardest in the same direction, and linear merge averages two incompatible pushes.

### 3.3 Subspace Dimensionality Ratio

**What it measures:** Whether two adapters use similar or different numbers of effective dimensions at a given layer.

**Definition:** For each adapter's per-layer W, compute the effective rank:

```
eff_rank = exp(H(σ̃))
```

where σ̃ is the normalized singular value spectrum (σ_i / Σ σ_i) and H is Shannon entropy.

Then for a pair:

```
dim_ratio = min(eff_rank_A, eff_rank_B) / max(eff_rank_A, eff_rank_B)
```

**Output per layer:** A scalar in (0, 1]. Value of 1 means both adapters use the same number of effective dimensions; values near 0 mean one is much more concentrated than the other.

**Why it matters:** The n10 dossier synthesis noted that victims tend to have more concentrated features. If catastrophic pairs show large dimensionality asymmetry (one adapter narrow, one broad) at critical layers while safe pairs are more balanced, that provides a geometric account of victim/culprit dynamics.

### 3.4 Directional Conflict Score

**What it measures:** Whether two adapters push in *opposing* directions within their shared subspace.

**Definition:** Project both adapters' per-layer W into their shared subspace (defined by the top principal vectors). Within this shared subspace, compute the cosine similarity of the projected perturbations:

```
conflict = 1 - cos(W_A_proj, W_B_proj) / 2
```

Normalized to [0, 1] where 0 means perfectly aligned perturbations and 1 means perfectly opposed.

In practice, compute this as: flatten the projected matrices, compute cosine similarity, and transform.

**Output per layer:** A scalar in [0, 1].

**Why it matters:** Two adapters can occupy the same subspace (high principal cosines) while either reinforcing each other (same direction → safe merge) or contradicting each other (opposing directions → destructive merge). This metric distinguishes these two cases.

---

## 4. Computation Procedure

### Step 1 — Layer-level SVD

For each adapter, at each layer, for each attention module (query, key, value, output):

1. Load lora_A and lora_B.
2. Compute W = lora_B @ lora_A.
3. Compute SVD: W = U Σ V^T.
4. Store U, Σ, V^T.

For the per-layer aggregate: concatenate or average across modules. The simplest approach: concatenate the four W matrices column-wise into a single per-layer perturbation matrix, then SVD the aggregate.

### Step 2 — Per-layer pair metrics

For each pair in the contrast panel, at each layer:

1. Compute principal angle spectrum between the two adapters' subspaces.
2. Compute top singular direction overlap.
3. Compute effective rank for each adapter and the dimensionality ratio.
4. Compute directional conflict score.

### Step 3 — Critical layer identification

For each pair, identify critical layers (top layers by combined norm mass, ≥ 60% threshold).

### Step 4 — Group comparison

Aggregate metrics across critical layers for each group. The central comparisons:

- **Group 1 vs. Group 2 at critical layers:** Do catastrophic collision cases show different subspace geometry than safe collision cases?
- **Group 1 vs. Group 3:** Do catastrophic cases show different geometry than non-collision cases? (Less informative — these differ in alignment already.)
- **Within CA-01 seed variants:** Do the catastrophic seed variant (s42×s7, Δ=41.7%) and mild variant (s7×s7, Δ=12.7%) differ in within-layer metrics? This is the seed-sensitivity test.

---

## 5. Seed Variant Analysis

For CA-01, all four seed combinations are available. The per-layer analysis (n15) showed these have nearly identical layer-level alignment. If within-layer metrics differentiate the catastrophic variant (s42×s7) from the mild variant (s7×s7), that directly demonstrates within-layer geometry as the seed-dependent variable.

For CA-02, the sharp culprit (qnli_s42) should show different within-layer geometry when paired with sst2 adapters compared to the benign qnli_s7.

---

## 6. Interpretation Framework

### Positive outcome

At least one within-layer metric cleanly separates Group 1 from Group 2 at critical layers, OR within-layer metrics differentiate catastrophic from mild seed variants of the same pair.

A positive outcome directly supports thresholded interference inside aligned layer profiles.

### Mixed outcome

Metrics show trends but with overlap, OR different metrics favor different cases, OR the signal is present for CA-01 but absent for CA-02 (or vice versa).

### Negative outcome

No within-layer metric differentiates catastrophic from safe collision cases. This would indicate that the operative variable is not in the per-layer subspace geometry — the search would need to move to output-space or head-level analysis.

---

## 7. Scope Boundaries

- This is a **pilot study** with a small panel. No statistical testing.
- All computation is CPU-only using existing saved adapters.
- The module-level aggregate (concatenation of Q/K/V/O matrices) is the primary analysis. Per-module decomposition is a follow-up if the aggregate shows signal.
- No merged-model analysis. All metrics are computed from source adapters.

---

## 8. Freeze Conditions

- The four metrics in §3 are the complete metric set.
- The contrast panel in §1 is fixed.
- The interpretation framework in §6 is pre-registered.
