# n30 — Output-Space Panel Definition

**Type:** panel definition
**Program:** Sidecar B — Output-Space Compatibility
**Stage:** A (CPU-only panel formalization)
**Date:** 2026-03-27

---

## 1. Purpose

Define the analyzable output-space panel for testing whether catastrophic
cross-task interference arises from readout incompatibility, decision-boundary
mismatch, or downstream amplification rather than from representation geometry
alone.

This note identifies which cases support output-space analysis given currently
available artifacts, and establishes the panel for Stages B and C.

---

## 2. Critical artifact discovery

A key finding during panel formalization: **all 16 source adapters include
full task-specific classifier head weights** in their safetensors files.

### DistilBERT adapters include:
- `pre_classifier.weight` — shape [768, 768] — intermediate projection
- `pre_classifier.bias` — shape [768]
- `classifier.weight` — shape [2, 768] — final task readout
- `classifier.bias` — shape [2]

### RoBERTa adapters include:
- `classifier.dense.weight` — shape [768, 768] — intermediate projection
- `classifier.dense.bias` — shape [768]
- `classifier.out_proj.weight` — shape [2, 768] — final task readout
- `classifier.out_proj.bias` — shape [2]

This means readout geometry is directly analyzable for all source adapters
without model inference. The final task-readout vectors (the 2×768 weight
matrices) define each adapter's decision boundary in representation space.

---

## 3. Available assets per case

For each case in the contrast panel, the following assets are available:

### Present and loadable
- Source adapter A LoRA weights (Q, K, V, O modules × all layers)
- Source adapter B LoRA weights (same)
- Source adapter A classifier/readout head weights
- Source adapter B classifier/readout head weights
- Source adapter A pre-classifier projection
- Source adapter B pre-classifier projection
- Aggregate eval results (accuracy, delta per task)
- Per-layer structural metrics (from Sidecar A: per_layer, per_module, head_level)

### Not available
- **Merged adapter checkpoints** — not saved; only eval results stored
- **Raw logits/predictions** — not saved; only aggregate accuracy stored
- **Per-example predictions** — not stored
- **Merged classifier heads** — would require re-merging (linear merge of
  classifier weights is feasible on CPU)

### Derivable on CPU
- **Merged readout weights** — linear interpolation of source classifier heads
  (standard LoRA merge: `W_merged = 0.5 * W_a + 0.5 * W_b`)
- **Readout geometry metrics** — cosine similarity, subspace overlap,
  directional relation of task-specific decision vectors
- **Readout–representation coupling** — how LoRA output-space directions
  relate to classifier readout directions
- **CPU inference** — DistilBERT (~66M params) can run small eval subsets
  on CPU if needed for logit analysis; RoBERTa (~125M params) is feasible
  but slower

---

## 4. Analyzable levels

Given the asset inventory, four levels of output-space analysis are possible:

| Level | What it requires | Feasibility |
|-------|-----------------|-------------|
| `head_weights_only` | Classifier head weights from safetensors | **Available now for all 16 adapters** |
| `merged_readout_geometry` | Linear interpolation of source readout weights | **Derivable on CPU for all pairs** |
| `predictions_and_logits` | CPU inference on eval subsets | **Feasible for DistilBERT, slow for RoBERTa** |
| `full_readout_panel` | All above combined | **Feasible for DistilBERT cases** |

---

## 5. Output-space panel

### Group 1 — Catastrophic anchors

| Case ID | Pair | Backbone | Seeds | Worst Δ | Instability | Classifier head | Analyzable level |
|---------|------|----------|-------|---------|-------------|-----------------|-----------------|
| CA-01 | QNLI × MRPC | DistilBERT | s42×s7 (worst), s7×s7 (mildest) | 41.7% | 0.87 | yes | full_readout_panel |
| CA-02 | QNLI × SST-2 | RoBERTa | s42×s7 (worst), s7×s42 (mildest) | 27.2% | 0.74 | yes | merged_readout_geometry |

**Notes:**
- CA-01 on DistilBERT is the strongest candidate for full_readout_panel
  because DistilBERT is small enough for CPU inference on eval subsets.
- CA-02 on RoBERTa can do merged_readout_geometry (weight analysis) on CPU.
  Full inference is feasible but slower.

### Group 2 — Safe collision controls

| Case ID | Pair | Backbone | Worst Δ | Instability | Classifier head | Analyzable level |
|---------|------|----------|---------|-------------|-----------------|-----------------|
| SC-QMRB | QNLI × MRPC | RoBERTa | 1.7% | 0.87 (same pair, different backbone) | yes | merged_readout_geometry |
| SC-MSRB | MRPC × SST-2 | RoBERTa | 4.8% | 0.12 | yes | merged_readout_geometry |

**Notes:**
- SC-QMRB is the crucial control: same pair as CA-01 but on RoBERTa where
  it is mild rather than catastrophic. Readout geometry comparison between
  CA-01 and SC-QMRB directly tests whether readout structure differs when
  severity differs.
- SC-MSRB: highest collision (ρ=0.89 per Sidecar A) but stable. Tests whether
  high collision + compatible readout → safe.

### Group 3 — Mild or stable cross-task contrasts

| Case ID | Pair | Backbone | Worst Δ | Instability | Classifier head | Analyzable level |
|---------|------|----------|---------|-------------|-----------------|-----------------|
| NC-RMDB | RTE × MRPC | DistilBERT | 7.1% | 0.19 | yes | full_readout_panel |
| NC-RSDB | RTE × SST-2 | DistilBERT | 8.3% | 0.15 | yes | full_readout_panel |
| NC-QSDB | QNLI × SST-2 | DistilBERT | 11.0% | 0.74 (but DistilBERT variant is moderate) | yes | full_readout_panel |

**Notes:**
- NC-RMDB and NC-RSDB are stable cross-task pairs — low instability,
  moderate degradation. They serve as controls where upstream risk is
  present but outcome is not catastrophic.
- NC-QSDB is the DistilBERT variant of the CA-02 pair — moderate rather
  than catastrophic on this backbone.

### Group 4 — Seed-sensitive within-family contrast

| Case ID | Variant | Backbone | Δ | Classifier head | Analyzable level |
|---------|---------|----------|---|-----------------|-----------------|
| CA-01-catastrophic | QNLI_s42 × MRPC_s7 | DistilBERT | 41.7% | yes | full_readout_panel |
| CA-01-mild | QNLI_s7 × MRPC_s7 | DistilBERT | 12.7% | yes | full_readout_panel |

**Notes:**
- Same pair, same backbone, same MRPC source. Only the QNLI seed differs.
- Readout weights for QNLI_s42 vs QNLI_s7 are both available.
- This contrast directly tests whether readout geometry explains the 29pp
  seed gap that module-level V geometry only partially resolves (per n24).

### Group 5 — Toxic vs benign adapter-linked contrast

| Case ID | Variant | Backbone | Δ | Classifier head | Analyzable level |
|---------|---------|----------|---|-----------------|-----------------|
| CA-02-toxic | QNLI_s42 × SST-2_s7 | RoBERTa | 27.2% | yes | merged_readout_geometry |
| CA-02-benign | QNLI_s7 × SST-2_s42 | RoBERTa | 1.0% | yes | merged_readout_geometry |

**Notes:**
- QNLI_s42 is the identified "toxic adapter" (per n09). Its classifier head
  is directly available.
- Compare readout geometry of QNLI_s42 vs QNLI_s7 to test whether the
  toxic adapter's catastrophic behavior is visible in readout space.

---

## 6. Panel summary

| Group | Cases | Analyzable level | CPU-feasible |
|-------|-------|-----------------|-------------|
| Catastrophic anchors | 2 | full (CA-01) / merged_readout (CA-02) | yes |
| Safe collision controls | 2 | merged_readout_geometry | yes |
| Mild cross-task | 3 | full_readout_panel | yes |
| Seed-sensitive (CA-01) | 2 variants | full_readout_panel | yes |
| Toxic vs benign (CA-02) | 2 variants | merged_readout_geometry | yes |
| **Total** | **11 case entries** | | |

---

## 7. Adapter inventory for readout analysis

All 16 source adapters across 4 tasks × 2 seeds × 2 backbones:

| Adapter | Task | Seed | Backbone | Path |
|---------|------|------|----------|------|
| qnli_s42 | QNLI | 42 | DistilBERT | `results/cross_task_subtype_study_01/sources/qnli_s42/` |
| qnli_s7 | QNLI | 7 | DistilBERT | `results/cross_task_subtype_study_01/sources/qnli_s7/` |
| mrpc_s42 | MRPC | 42 | DistilBERT | `results/cross_task_subtype_study_01/sources/mrpc_s42/` |
| mrpc_s7 | MRPC | 7 | DistilBERT | `results/cross_task_subtype_study_01/sources/mrpc_s7/` |
| rte_s42 | RTE | 42 | DistilBERT | `results/cross_task_subtype_study_01/sources/rte_s42/` |
| rte_s7 | RTE | 7 | DistilBERT | `results/cross_task_subtype_study_01/sources/rte_s7/` |
| sst2_s42 | SST-2 | 42 | DistilBERT | `results/cross_task_subtype_study_01/sources/sst2_s42/` |
| sst2_s7 | SST-2 | 7 | DistilBERT | `results/cross_task_subtype_study_01/sources/sst2_s7/` |
| qnli_s42 | QNLI | 42 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/qnli_s42/` |
| qnli_s7 | QNLI | 7 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/qnli_s7/` |
| mrpc_s42 | MRPC | 42 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/mrpc_s42/` |
| mrpc_s7 | MRPC | 7 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/mrpc_s7/` |
| rte_s42 | RTE | 42 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/rte_s42/` |
| rte_s7 | RTE | 7 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/rte_s7/` |
| sst2_s42 | SST-2 | 42 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/sst2_s42/` |
| sst2_s7 | SST-2 | 7 | RoBERTa | `results/task_pair_severity_generalization_study_01/roberta/sources/sst2_s7/` |

---

## 8. Classifier head architecture summary

### DistilBERT readout path
```
[CLS] hidden state (768) → pre_classifier (768→768, ReLU) → classifier (768→2)
```
Two-stage: projection + linear readout. Both layers saved per adapter.

### RoBERTa readout path
```
[CLS] hidden state (768) → classifier.dense (768→768, tanh) → classifier.out_proj (768→2)
```
Two-stage: projection + linear readout. Both layers saved per adapter.

The final readout layer in both cases is a 2×768 weight matrix whose rows
define the decision-boundary normal vectors in representation space.

---

## 9. What merged readout weights look like

For a standard linear merge with equal weighting:
```
W_merged_classifier = 0.5 * W_a_classifier + 0.5 * W_b_classifier
```

This produces a merged decision boundary that is the geometric average of
the two source boundaries. The central output-space hypothesis is that when
these source boundaries are incommensurable (e.g., one classifies
entailment/non-entailment while the other classifies positive/negative
sentiment), the merged boundary occupies a "neither-task" state that
satisfies neither classification well.

---

## 10. Success criteria for Stage A

- [x] Analyzable output-space panel defined with at least 2 catastrophic
  anchors and 2 safe controls
- [x] At least one catastrophic anchor supports full_readout_panel analysis
  (CA-01 on DistilBERT)
- [x] Within-family seed-sensitive contrast included (CA-01 catastrophic
  vs mild variants)
- [x] Toxic vs benign contrast included (CA-02 QNLI_s42 vs QNLI_s7)
- [x] Classifier head weights confirmed present in all adapter files
- [x] Merged readout geometry derivable on CPU for all panel entries

**Stage A is complete.** Proceed to Stage B.

---

## 11. Relationship to Sidecar A

This panel reuses the same case definitions as Sidecar A's contrast panels
(per n13, n19, n22) but extends the analysis from representation geometry
to readout-space geometry. The key novelty is that classifier head weights
are now known to be available, enabling direct analysis of how upstream
LoRA modifications interact with task-specific readout boundaries.

The multiscale mechanism ladder (n25) identified Rung 3 (downstream
amplification) as the leading open question. This panel is designed to
test that rung directly.
