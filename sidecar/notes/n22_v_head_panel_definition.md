# Note: V-Module Head-Level Panel Definition

## Metadata

- **Type:** subset definition
- **Date:** 2026-03-26
- **Related notes:** n19 (per-module subset), n21 (per-module findings), n20 (per-module protocol)
- **Project:** V-Module Head-Level Program, Stage A

---

## Purpose

This note defines the analyzable head-level panel for the V-module head-level geometry program. The per-module analysis (n21) identified V-module dimensionality ratio as the strongest correlate of catastrophic threshold (d=3.36, zero range overlap), but CA-01 seed sensitivity remains unexplained at per-module resolution (all deltas < 0.07). This stage decomposes the V-module into per-head comparisons, testing whether the signal localizes further to specific attention heads.

**Central question:** Does the decisive difference localize further to specific attention heads inside the V module, especially in seed-sensitive catastrophic families like CA-01?

**Mechanistic hypothesis:** The V-module signal is still partly aggregated. Catastrophic cross-task interference may localize to a small number of attention heads whose geometry differs from safe collision controls and mild contrasts, even when whole-module summaries blur the distinction.

---

## 1. Head Architecture

Both backbones use 12 attention heads with a hidden size of 768, yielding 64 dimensions per head:

| Property | DistilBERT | RoBERTa |
|:---------|:-----------|:--------|
| Hidden size | 768 | 768 |
| Attention heads | 12 | 12 |
| Head dimension | 64 | 64 |
| V-module LoRA rank | 16 | 16 |
| V lora_A shape | (16, 768) | (16, 768) |
| V lora_B shape | (768, 16) | (768, 16) |
| V product W shape | (768, 768) | (768, 768) |
| W rows per head | 64 | 64 |

### Head slicing

The V-module product matrix W = lora_B @ lora_A has shape (768, 768). The row dimension corresponds to the output of the value projection, which is concatenated across heads before the output projection (O module). Each head's contribution is a contiguous 64-row block:

```
W_head_h = W[h*64 : (h+1)*64, :]    # shape: (64, 768)
```

This gives 12 head-level matrices per layer, each (64, 768). These are rectangular (not square), unlike the full-module (768, 768) matrices. SVD on a (64, 768) matrix yields at most 64 non-zero singular values.

**Important geometric note:** The head-level matrix W_head_h represents how head h's value projection is perturbed by the LoRA adapter. The full input space (768 dimensions) is shared across all heads, but each head reads a different 64-dimensional slice of the output. This means head-level effective rank is bounded by 64 (the row dimension), which is substantially lower than the module-level bound of 768.

---

## 2. Critical V Layers

Critical layers are selected by V-module norm mass, using the ≥60% combined norm mass threshold from the per-module analysis (n20). The critical layers are identified per backbone from the adapter norm profiles:

### DistilBERT (6 layers)

V-module norm mass is concentrated in layers 1, 2, and 4, with layers 1+2+4 carrying approximately 62% of combined V-module norm. Layer 3 contributes ~15%, bringing the four-layer set (1, 2, 3, 4) to ~77%.

**Critical V layers for DistilBERT: {1, 2, 3, 4}** (same as the per-module analysis critical layer set).

### RoBERTa (12 layers)

V-module norm mass peaks in the middle-to-late layers, with layers 4–8 carrying approximately 52–59% of combined V-module norm depending on adapter. Adding layer 3 and 9 brings coverage above 70%.

**Critical V layers for RoBERTa: {4, 5, 6, 7, 8}** (5 layers, covering ~52–59% of V-module norm mass; layers 3 and 9 are secondary).

For the head-level pilot, we restrict to these critical V layers to keep the computation tractable while focusing on the layers where the V-module signal is strongest.

---

## 3. Contrast Panel

The contrast panel extends the per-module panel (n19) with an explicit Group 4 (seed-sensitive variants) and restricts to V-module heads at critical layers.

### Group 1: Catastrophic Anchors

| Case ID | Pair | Backbone | Critical V Layers | Worst Δ | Instability |
|:--------|:-----|:---------|:------------------|--------:|:-----------:|
| CA-01 | QNLI×MRPC | DistilBERT | {1, 2, 3, 4} | 41.7% | 0.868 |
| CA-02 | QNLI×SST-2 | RoBERTa | {4, 5, 6, 7, 8} | 27.2% | 0.738 |

### Group 2: Safe Collision Controls

| Case ID | Pair | Backbone | Critical V Layers | Worst Δ | Instability |
|:--------|:-----|:---------|:------------------|--------:|:-----------:|
| SC-QMRB | QNLI×MRPC | RoBERTa | {4, 5, 6, 7, 8} | 1.7% | 0.868 |
| SC-MSRB | MRPC×SST-2 | RoBERTa | {4, 5, 6, 7, 8} | 15.0% | 0.211 |

### Group 3: Non-Collision Contrasts

| Case ID | Pair | Backbone | Critical V Layers | Worst Δ | Instability |
|:--------|:-----|:---------|:------------------|--------:|:-----------:|
| NC-QSDB | QNLI×SST-2 | DistilBERT | {1, 2, 3, 4} | 11.0% | 0.738 |
| NC-RMRB | RTE×MRPC | RoBERTa | {4, 5, 6, 7, 8} | 8.3% | 0.193 |

### Group 4: Seed-Sensitive Variants (within Groups 1–3)

The seed sensitivity analysis targets the two catastrophic anchor families:

**CA-01 (DistilBERT):** The pair QNLI×MRPC with different QNLI seeds.
- Worst variant: qnli_s42 × mrpc_s7 → Δ = 41.7%
- Mild variant: qnli_s7 × mrpc_s7 → Δ = 12.7%
- The 29-point severity gap is invisible at per-module resolution (all Δ < 0.07). Head-level decomposition is the primary test for whether this gap concentrates at specific heads.

**CA-02 (RoBERTa):** The pair QNLI×SST-2 with toxic vs. benign QNLI adapters.
- Toxic adapter: qnli_s42 pairs → Δ_worst = 27.2%
- Benign adapter: qnli_s7 pairs → Δ_worst = 1.0%
- Per-module analysis partly explained this (V Δcos = -0.15, O Δcos = -0.31). Head-level decomposition tests whether the V-module contribution concentrates at specific heads.

---

## 4. Preliminary Head-Level Reconnaissance

Before formalizing the full analysis protocol, preliminary head-level data was computed to confirm the signal is present and to guide the metric selection.

### 4.1 CA-01 Head-Level Dimensionality Ratio (DistilBERT, V module)

At layer 3 (where the largest module-level seed delta appeared for overlap), the head-level dimensionality ratio shows a Δ_DR range of [-0.23, +0.09] across heads. Head 6 shows Δ_DR = -0.229 (worst seed has lower dim ratio than mild seed), while the module-level aggregate was Δ = +0.006. This is the first evidence that the CA-01 seed sensitivity signal exists at head resolution but is washed out by averaging across heads with opposing signs.

At layers 1 and 2, individual heads show Δ_DR up to ±0.19, while layers 3 and 4 show a mix of larger and smaller deltas. The pattern is not uniform across layers, suggesting that the seed-sensitive mechanism may engage different heads at different layers.

### 4.2 CA-01 Head-Level Alignment (DistilBERT, V module, layer 3)

Top direction overlap shows Δ_OV up to -0.16 (head 10) and -0.11 (head 6). These are heads where the worst seed variant's V perturbation is substantially less aligned with MRPC's than the mild variant's — a pattern consistent with the hypothesis that specific heads become geometrically incompatible under one seed but not the other.

### 4.3 CA-02 Head-Level Dimensionality Ratio (RoBERTa, V module, layer 6)

The toxic adapter (qnli_s42) shows consistently lower dimensionality ratios with SST-2 across most heads (10 of 12 heads), with the largest gaps at heads 2, 8, and 9 (Δ_DR ~ -0.15). This is directionally consistent with the module-level finding (toxic adapter has lower V-module dim ratio) but now reveals that the signal is distributed across multiple heads rather than concentrated in one.

### 4.4 Head Norm Mass Distribution

Norm mass is approximately uniform across heads (~0.07–0.10 per head at most layers), confirming that no single head dominates the V-module perturbation. The signal, if it localizes, does so through geometry (alignment, dimensionality) rather than magnitude.

---

## 5. Head-Level Computation Budget

| Component | DistilBERT | RoBERTa | Total |
|:----------|:----------:|:-------:|:-----:|
| Critical V layers | 4 | 5 | 9 |
| Heads per layer | 12 | 12 | — |
| Head matrices per layer per adapter | 12 | 12 | — |
| Contrast panel cases | 2 | 4 | 6 |
| Seed combos per case | 4 | 4 | — |
| Variants | 8 | 16 | 24 |
| Head comparisons per variant per layer | 12 | 12 | — |
| **Total head comparisons** | 8×4×12 = 384 | 16×5×12 = 960 | **1,344** |
| Metrics per comparison | 4 | 4 | — |
| **Total metric computations** | 1,536 | 3,840 | **5,376** |

Each head comparison involves SVD of two (64, 768) matrices — fast on CPU. The entire computation should complete in under 5 minutes.

---

## 6. What Head-Level Decomposition Tests

### Primary predictions

**P1 (CA-01 head localization):** The 29-point severity gap in CA-01 concentrates at a subset of heads (≤4 of 12) where head-level dimensionality ratio or alignment shows deltas ≥ 0.10 between worst and mild seed variants. This is the main target: the signal that was invisible at per-module resolution.

**P2 (CA-02 head localization):** The CA-02 toxic/benign difference, already visible at per-module level, further concentrates at specific heads, with at least one head showing d > 2.0 on dimensionality ratio between toxic and benign variants.

**P3 (Head mismatch concentration):** In the backbone-controlled comparison (CA-02 vs. safe collision controls on RoBERTa), the V-module dimensionality mismatch signal (d = 3.36 at module level) persists or strengthens when computed at the "worst head" level — i.e., using the minimum or maximum head-level dim ratio rather than the module mean.

### Null result interpretation

If no head subset shows a clean CA-01 seed separation (all head-level Δ < 0.08), the seed sensitivity variable is below head resolution and the remaining candidates are: (a) individual weight directions within heads, (b) output-space interaction through the O module and classification head. In that case, escalate to output-space analysis (Stage D).

---

## 7. Structured Outputs

| File | Location |
|------|----------|
| This note | `sidecar/notes/n22_v_head_panel_definition.md` |
| Head panel table (JSON) | `sidecar/results/head_level_v/head_panel_table.json` |
| Head panel table (MD) | `sidecar/results/head_level_v/head_panel_table.md` |
