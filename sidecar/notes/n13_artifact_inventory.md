# Note: Per-Layer Artifact Inventory

## Metadata

- **Type:** inventory
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related notes:** n12 (artifact mining inventory)
- **Project:** Phase 3, Project G — Per-Layer Structural Analysis

---

## Purpose

This note documents the complete inventory of saved LoRA adapter weights available for per-layer structural analysis. It confirms which adapters exist, verifies their internal structure, and establishes the analyzable contrast panel for the per-layer study.

---

## 1. Inventory Summary

All 16 source adapters are present and loadable. The adapters span two backbones, four tasks, and two seeds.

| Property | DistilBERT | RoBERTa |
|----------|-----------|---------|
| Base model | distilbert-base-uncased | roberta-base |
| Transformer layers | 6 | 12 |
| Attention type | Standard | Standard |
| LoRA rank | 16 | 16 |
| LoRA alpha | 16 | 16 |
| Hidden dimension | 768 | 768 |
| Target modules | q_lin, k_lin, v_lin, out_lin | query, key, value, dense |
| lora_A shape | (16, 768) | (16, 768) |
| lora_B shape | (768, 16) | (768, 16) |
| Adapters per backbone | 8 | 8 |
| LoRA keys per adapter | 48 | 96 |
| Adapter file size | ~4.6 MB | ~13 MB |

**Rank discrepancy note:** The S01 protocol specifies r=8, but all saved adapters use r=16. This does not affect the per-layer analysis — all metrics are computed from whatever rank the adapters actually have. The discrepancy should be noted in the study record but is not a blocker.

---

## 2. Adapter Manifest

### DistilBERT (8 adapters)

| Adapter ID | Task | Seed | Path | Status |
|-----------|------|------|------|--------|
| distilbert_qnli_s42 | QNLI | s42 | `results/cross_task_subtype_study_01/sources/qnli_s42/` | Present |
| distilbert_qnli_s7 | QNLI | s7 | `results/cross_task_subtype_study_01/sources/qnli_s7/` | Present |
| distilbert_mrpc_s42 | MRPC | s42 | `results/cross_task_subtype_study_01/sources/mrpc_s42/` | Present |
| distilbert_mrpc_s7 | MRPC | s7 | `results/cross_task_subtype_study_01/sources/mrpc_s7/` | Present |
| distilbert_rte_s42 | RTE | s42 | `results/cross_task_subtype_study_01/sources/rte_s42/` | Present |
| distilbert_rte_s7 | RTE | s7 | `results/cross_task_subtype_study_01/sources/rte_s7/` | Present |
| distilbert_sst2_s42 | SST-2 | s42 | `results/cross_task_subtype_study_01/sources/sst2_s42/` | Present |
| distilbert_sst2_s7 | SST-2 | s7 | `results/cross_task_subtype_study_01/sources/sst2_s7/` | Present |

### RoBERTa (8 adapters)

| Adapter ID | Task | Seed | Path | Status |
|-----------|------|------|------|--------|
| roberta_qnli_s42 | QNLI | s42 | `results/task_pair_severity_generalization_study_01/roberta/sources/qnli_s42/` | Present |
| roberta_qnli_s7 | QNLI | s7 | `results/task_pair_severity_generalization_study_01/roberta/sources/qnli_s7/` | Present |
| roberta_mrpc_s42 | MRPC | s42 | `results/task_pair_severity_generalization_study_01/roberta/sources/mrpc_s42/` | Present |
| roberta_mrpc_s7 | MRPC | s7 | `results/task_pair_severity_generalization_study_01/roberta/sources/mrpc_s7/` | Present |
| roberta_rte_s42 | RTE | s42 | `results/task_pair_severity_generalization_study_01/roberta/sources/rte_s42/` | Present |
| roberta_rte_s7 | RTE | s7 | `results/task_pair_severity_generalization_study_01/roberta/sources/rte_s7/` | Present |
| roberta_sst2_s42 | SST-2 | s42 | `results/task_pair_severity_generalization_study_01/roberta/sources/sst2_s42/` | Present |
| roberta_sst2_s7 | SST-2 | s7 | `results/task_pair_severity_generalization_study_01/roberta/sources/sst2_s7/` | Present |

---

## 3. Weight Key Structure

### DistilBERT key pattern

```
base_model.model.distilbert.transformer.layer.{0-5}.attention.{q_lin|k_lin|v_lin|out_lin}.lora_{A|B}.weight
```

6 layers × 4 modules × 2 matrices = 48 LoRA keys per adapter. Remaining keys are classifier weights (not relevant for per-layer analysis).

### RoBERTa key pattern

```
base_model.model.roberta.encoder.layer.{0-11}.attention.{self.query|self.key|self.value|output.dense}.lora_{A|B}.weight
```

12 layers × 4 modules × 2 matrices = 96 LoRA keys per adapter. Remaining keys are classifier weights.

### Structural alignment

Both backbones target the same four attention components (query, key, value, output) with identical LoRA rank (16) and hidden dimension (768). The per-layer metrics are directly comparable within a backbone. Cross-backbone comparison requires normalization by layer count (6 vs. 12).

---

## 4. Analyzable Contrast Panel

The per-layer study uses a three-group contrast design.

### Group A — Catastrophic Anchors

These are the cases where the instability program's central question is sharpest: what structural footprint, if any, distinguishes catastrophic outcomes?

| Case ID | Pair | Backbone | Worst Δ | Seed range | Instability |
|---------|------|----------|--------:|----------:|-----------:|
| CA-01 | QNLI × MRPC | DistilBERT | 41.7% | 28.9% | 0.87 |
| CA-02 | QNLI × SST-2 | RoBERTa | 27.2% | 26.2% | 0.74 |

**Adapters involved:**

- CA-01: distilbert_qnli_s42, distilbert_qnli_s7, distilbert_mrpc_s42, distilbert_mrpc_s7 (4 adapters, 4 seed combinations)
- CA-02: roberta_qnli_s42, roberta_qnli_s7, roberta_sst2_s42, roberta_sst2_s7 (4 adapters, 4 seed combinations)

**Sharpest within-group contrasts:**

- CA-01: worst variant (s42×s7, Δ=41.7%) vs. best variant (s7×s7, Δ=12.7%) — range of 28.9 points
- CA-02: worst variant (s42×s7, Δ=27.2%) vs. best variant (s7×s42, Δ=1.0%) — range of 26.2 points

### Group B — Same-Task Controls

These set the noise floor. Same-task merges should show minimal per-layer divergence; any per-layer signal that fires on same-task pairs is a false positive.

| Case ID | Pair | Backbone | Worst Δ |
|---------|------|----------|--------:|
| ST-DB-QNLI | QNLI s42 × QNLI s7 | DistilBERT | 2.2% |
| ST-DB-MRPC | MRPC s42 × MRPC s7 | DistilBERT | 0.8% |
| ST-RB-QNLI | QNLI s42 × QNLI s7 | RoBERTa | 0.0% |
| ST-RB-SST2 | SST-2 s42 × SST-2 s7 | RoBERTa | 0.0% |

These controls use adapters already in Group A, so no additional adapters are needed.

### Group C — Stable Cross-Task Contrasts

These are cross-task pairs with low instability. If per-layer metrics differentiate Group A from Group C, the signal is specific to catastrophic anchors, not merely to cross-task status.

| Case ID | Pair | Backbone | Worst Δ | Instability |
|---------|------|----------|--------:|-----------:|
| SC-01 | RTE × MRPC | DistilBERT | 7.1% | 0.19 |
| SC-02 | RTE × SST-2 | DistilBERT | 8.3% | 0.15 |
| SC-03 | RTE × MRPC | RoBERTa | 8.3% | 0.19 |
| SC-04 | RTE × SST-2 | RoBERTa | 12.6% | 0.15 |

**Adapters involved:** distilbert_rte_{s42,s7}, distilbert_mrpc_{s42,s7}, distilbert_sst2_{s42,s7}, roberta_rte_{s42,s7}, roberta_mrpc_{s42,s7}, roberta_sst2_{s42,s7}

### Panel summary

All 16 adapters are used. The panel produces:

- 2 catastrophic-anchor cases (Group A) with 4 seed variants each = 8 merge pairs
- 4 same-task controls (Group B) = 4 merge pairs
- 4 stable cross-task cases (Group C) with 4 seed variants each = 16 merge pairs

Total: 28 analyzable cases. But the per-layer analysis operates on *individual adapters and adapter pairs*, not on merged models. The key comparisons are between the structural profiles of adapters that participate in catastrophic outcomes versus those that participate in stable outcomes.

---

## 5. What This Inventory Establishes

1. **All 16 source adapters are present, loadable, and structurally consistent.** No data gaps block the per-layer analysis.

2. **The contrast panel is fully populated.** Groups A, B, and C all have the required adapters.

3. **The rank is 16, not 8.** This is higher than the S01 protocol specified but does not affect the analysis. If anything, higher rank provides richer per-layer structure to analyze.

4. **Cross-backbone per-layer comparison requires normalization.** DistilBERT has 6 layers; RoBERTa has 12. Direct layer-index comparison is not meaningful. Metrics should be expressed as distributions (e.g., fraction of total norm per layer) rather than absolute values.

---

## 6. Structured Output

The machine-readable inventory is at `sidecar/results/per_layer_analysis/artifact_inventory.json` (schema: `sidecar.artifact_inventory/v1`).
