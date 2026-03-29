# Note: Per-Module Subset Definition

## Metadata

- **Type:** subset definition
- **Date:** 2026-03-26
- **Related notes:** n16 (collision subset), n17 (within-layer protocol), n18 (within-layer findings)
- **Project:** Per-Module Geometry Program, Stage A

---

## Purpose

This note defines the analyzable per-module panel for the per-module geometry program. The within-layer analysis (n18) operated on concatenated Q/K/V/O matrices per layer, which may have diluted module-specific signals. This stage decomposes the same contrast panel into per-module comparisons, computing metrics separately for each attention component.

**Central question:** If aggregate within-layer geometry is too coarse, does the decisive difference appear at the per-module level inside attention blocks, or does the explanation likely live in output-space / task-head incompatibility instead?

---

## 1. Cross-Backbone Module Correspondence

The four attention modules have different naming conventions across backbones but serve identical functional roles:

| Canonical Name | Function | DistilBERT Key | RoBERTa Key |
|:--------------:|:---------|:---------------|:------------|
| **Q** | Query projection | `q_lin` | `self.query` |
| **K** | Key projection | `k_lin` | `self.key` |
| **V** | Value projection | `v_lin` | `self.value` |
| **O** | Output projection | `out_lin` | `output.dense` |

All modules have rank r=16 LoRA adapters with shapes:
- DistilBERT: lora_A = (16, 768), lora_B = (768, 16) → W = B @ A is (768, 768)
- RoBERTa: lora_A = (16, 768), lora_B = (768, 16) → W = B @ A is (768, 768)

The per-module W matrices are square (768×768) and individually SVD-decomposable, unlike the concatenated (768×3072) matrices used in n17–n18.

---

## 2. Contrast Panel (Inherited from n16/n17)

The contrast panel is identical to the within-layer analysis, now decomposed per module:

### Group 1: Catastrophic Anchors

| Case ID | Pair | Backbone | Collision Class | Worst Δ | Instability |
|---------|------|----------|-----------------|--------:|:-----------:|
| CA-01 | QNLI×MRPC | DistilBERT | catastrophic_collision | 41.7% | 0.868 |
| CA-02 | QNLI×SST-2 | RoBERTa | moderate_alignment_catastrophic | 27.2% | 0.738 |

### Group 2: Safe Collision Controls

| Case ID | Pair | Backbone | Collision Class | Worst Δ | Instability |
|---------|------|----------|-----------------|--------:|:-----------:|
| SC-QMRB | QNLI×MRPC | RoBERTa | unstable_collision | 1.7% | 0.868 |
| SC-MSRB | MRPC×SST-2 | RoBERTa | non_catastrophic_collision | 15.0% | 0.211 |

### Group 3: Non-Collision Contrasts

| Case ID | Pair | Backbone | Collision Class | Worst Δ | Instability |
|---------|------|----------|-----------------|--------:|:-----------:|
| NC-QSDB | QNLI×SST-2 | DistilBERT | non_collision_cross_task | 11.0% | 0.738 |
| NC-RMRB | RTE×MRPC | RoBERTa | non_collision_cross_task | 8.3% | 0.193 |

### Group 4: Seed-Sensitive Variants (within Groups 1–3)

The seed sensitivity test is conducted within CA-01 and CA-02:
- **CA-01 seed variants:** s42×s7 (Δ=41.7%), s7×s7 (Δ=12.7%), s42×s42 and s7×s42 (intermediate)
- **CA-02 seed variants:** qnli_s42 pairs (toxic adapter, Δ=27.2% worst) vs. qnli_s7 pairs (benign adapter, Δ=1.0% worst)

---

## 3. What Per-Module Decomposition Tests

The aggregate analysis (n18) found that when backbone is controlled, catastrophic cases are indistinguishable from safe controls. Two possibilities:

1. **Module-concentrated signal:** The catastrophic difference is concentrated in one or two modules (e.g., value matrices) but is washed out when all four modules are concatenated. A per-module analysis would reveal this.

2. **Sub-module or output-space signal:** The catastrophic mechanism is below per-module resolution (e.g., specific attention heads, or in the classification head interaction). Per-module decomposition would also yield null results, pointing toward output-space escalation.

### Specific predictions under hypothesis (1):

- **Value-matrix hypothesis:** V modules may concentrate the catastrophic signal because value projections directly determine what information is passed to the output. If catastrophic pairs show distinctive V-module geometry while Q/K/O are indistinguishable, this localizes the mechanism.

- **Query-key hypothesis:** Q and K jointly determine attention patterns. If catastrophic pairs show distinctive Q×K interaction geometry, this points to attention routing as the mechanism.

- **Output-projection hypothesis:** O transforms the attention output back to the residual stream. If catastrophic pairs show distinctive O-module geometry, this points to residual-stream contamination.

### Null result interpretation:

If no module shows a clean catastrophic/safe separation, the threshold variable is below per-module resolution, and the program should escalate to output-space analysis (Workstream C).

---

## 4. Analysis Design

### 4.1 Metrics (same 4 from n17, applied per module)

1. **Principal angle spectrum** — subspace alignment between adapter A's module-W and adapter B's module-W
2. **Top direction overlap** — cosine between dominant singular vectors per module
3. **Dimensionality ratio** — effective rank comparison per module
4. **Directional conflict** — opposing perturbation directions in shared subspace per module

### 4.2 Layer selection

Same critical layer selection as n17: top layers by combined norm mass ≥ 60%. The critical layers are already computed and cached from the within-layer analysis.

### 4.3 Output structure

Per-module metrics are computed for each (case × seed combo × layer × module) tuple, then aggregated to:
- Per-module critical-layer means per variant
- Per-module group-level statistics
- Module-by-module group comparison (the key deliverable: a 4×4 matrix of module × metric, colored by group)

### 4.4 Decision criteria

**POSITIVE:** At least one module shows a clean group separation (Group 1 outside Group 2 range on at least one metric) that was not visible in the aggregate analysis.

**MIXED:** Some module-level differences exist but ranges overlap or the separation is driven by the backbone confound (CA-01 only).

**NEGATIVE:** No module shows better group separation than the aggregate analysis. Escalate to output-space.

---

## 5. Per-Module Coverage Verification

All 16 adapters (8 DistilBERT + 8 RoBERTa) have complete Q/K/V/O coverage:

| Backbone | Tasks | Seeds | Layers | Modules per layer | Total W matrices per adapter |
|----------|-------|-------|--------|-------------------|-----------------------------:|
| DistilBERT | QNLI, MRPC, SST-2, RTE | s42, s7 | 6 | 4 (q_lin, k_lin, v_lin, out_lin) | 24 |
| RoBERTa | QNLI, MRPC, SST-2, RTE | s42, s7 | 12 | 4 (self.query, self.key, self.value, output.dense) | 48 |

**Total per-module W matrices available:** 8 × 24 + 8 × 48 = 576

All adapters were verified present and loadable in n13. No coverage gaps.

---

## 6. Structured Outputs

| File | Location |
|------|----------|
| This note | `sidecar/notes/n19_per_module_subset_definition.md` |
| Per-module subset table (JSON) | `sidecar/results/per_module_geometry/per_module_subset_table.json` |
| Per-module subset table (MD) | `sidecar/results/per_module_geometry/per_module_subset_table.md` |
