# Panel: P01 — Catastrophic Anchor Panel

## Purpose

Defines the canonical set of task pairs, seed variants, and backbones for studying catastrophic cross-task merge interference. This panel provides the common reference frame for Workstream A (anchor replication) and Workstream B (layerwise conflict contrast).

## Panel Type

catastrophic-anchor

## Prior Evidence

This panel is grounded in two existing adjudication datasets:

- `results/cross_task_subtype_study_01/pairs/adjudication_results.json` — 29 pairs on distilbert-base-uncased
- `results/task_pair_severity_generalization_study_01/roberta/pairs/adjudication_results.json` — 29 pairs on roberta-base

The evidence base reveals that the identity of the catastrophic anchor **shifts across backbones**, which is itself a primary finding the sidecar must investigate.

## Anchor Classification

Anchors are classified by observed worst-case delta (max of delta_task_a, delta_task_b) across all seed variants on a given backbone. Thresholds:

- **Catastrophic:** worst-case delta > 15%
- **Severe:** worst-case delta 10–15%
- **Broad degradation:** worst-case delta 5–10%
- **Mild:** worst-case delta < 5%

These thresholds are operational definitions for this panel, not universal severity boundaries.

## Anchors

### DistilBERT-base-uncased

| Pair ID | Task A | Task B | Classification | Worst-case Δ | Seed variant | Notes |
|---------|--------|--------|----------------|---------------|--------------|-------|
| qnli × mrpc | QNLI | MRPC | **Catastrophic** | 41.7% (Δ_B) | s42 × s7 | Primary anchor. Both tasks collapse. |
| qnli × mrpc | QNLI | MRPC | Severe | 13.6% (Δ_A) | s7 × s42 | Same task pair, different seeds — still severe but ~3× less than worst seed combo. |
| mrpc × sst2 | MRPC | SST-2 | Severe | 12.7% (Δ_A) | s7 × s42 | MRPC degrades; SST-2 mostly protected. |
| qnli × sst2 | QNLI | SST-2 | Severe | 11.0% (Δ_A) | s42 × s7 | QNLI degrades; SST-2 protected. Asymmetric pattern. |
| qnli × sst2 | QNLI | SST-2 | Broad degradation | 10.4% (Δ_B) | s7 × s7 | SST-2 degrades on different seed combo. |
| rte × sst2 | RTE | SST-2 | Broad degradation | 8.3% (Δ_A) | s42 × s7 | RTE degrades; SST-2 protected. |
| qnli × rte | QNLI | RTE | Mild | 2.6% (Δ_A) | s42 × s7 | Surprisingly benign despite being cross-task. |
| rte × mrpc | RTE | MRPC | Mild | 7.1% (Δ_B) | s7 × s7 | Mild to moderate. NLI-family pair. |

### RoBERTa-base

| Pair ID | Task A | Task B | Classification | Worst-case Δ | Seed variant | Notes |
|---------|--------|--------|----------------|---------------|--------------|-------|
| qnli × sst2 | QNLI | SST-2 | **Catastrophic** | 27.2% (Δ_B) | s42 × s7 | **New catastrophic anchor on roberta.** SST-2 collapses (89.4% → 62.2%). |
| qnli × sst2 | QNLI | SST-2 | Severe | 17.6% (Δ_B) | s42 × s42 | Same task pair, different seeds — still severe. |
| mrpc × sst2 | MRPC | SST-2 | Severe | 15.0% (Δ_B) | s42 × s7 | SST-2 degrades. MRPC protected. |
| mrpc × sst2 | MRPC | SST-2 | Severe | 13.6% (Δ_B) | s7 × s7 | Consistent across seed variants. |
| rte × sst2 | RTE | SST-2 | Severe | 12.6% (Δ_A) | s42 × s42 | RTE degrades. |
| rte × sst2 | RTE | SST-2 | Severe | 11.2% (Δ_A) | s7 × s42 | Consistent. |
| rte × mrpc | RTE | MRPC | Broad degradation | 8.3% (Δ_A) | multiple | RTE degrades consistently. |
| qnli × mrpc | QNLI | MRPC | **Mild** | 1.7% (Δ_B) | s42 × s7 | **Was catastrophic on distilbert (41.7%). Near-harmless on roberta.** |

### Backbone Shift Summary

The most important finding encoded in this panel:

| Task pair | DistilBERT class | RoBERTa class | Shift |
|-----------|-----------------|---------------|-------|
| QNLI × MRPC | **Catastrophic** (41.7%) | Mild (1.7%) | Collapses to near-zero |
| QNLI × SST-2 | Severe (11.0%) | **Catastrophic** (27.2%) | Escalates ~2.5× |
| MRPC × SST-2 | Severe (12.7%) | Severe (15.0%) | Stable-ish |
| RTE × SST-2 | Broad degradation (8.3%) | Severe (12.6%) | Escalates modestly |
| QNLI × RTE | Mild (2.6%) | Broad degradation (8.3%) | Escalates |
| RTE × MRPC | Mild–moderate (7.1%) | Broad degradation (8.3%) | Stable |

## Conditions

### Backbones

**Existing data:**

1. `distilbert-base-uncased` — 6 transformer layers, 66M parameters
2. `roberta-base` — 12 transformer layers, 125M parameters

**Planned for replication (Study S01):**

3. `deberta-v3-base` — 12 transformer layers, 184M parameters (disentangled attention; architecturally distinct from both existing backbones)

### Seeds

Two seeds per adapter: **42** and **7**. For each cross-task pair, this produces up to 4 seed combinations (s42×s42, s42×s7, s7×s42, s7×s7).

Seed 123 was also used in some DistilBERT experiments but is not in the adjudication dataset. Standardize on {42, 7} for this panel.

### Training Configuration

LoRA fine-tuning:

- **Rank:** r=8 (default for encoder models)
- **Alpha:** α=16 (scaling factor = 2)
- **Target modules:** query, value projections
- **Learning rate:** 2e-4
- **Epochs:** 3
- **Batch size:** 16

(Exact configs to be confirmed against existing training scripts in `experiments/`.)

### Evaluation

Each adapter is evaluated on its own task's validation split. After linear merge (α=0.5/0.5), the merged model is evaluated on **both** source tasks.

Metrics:

- **Accuracy** on QNLI, SST-2, MRPC validation sets
- **F1** on RTE validation set (following GLUE conventions)

Delta = best_source_score − merged_score (positive = degradation).

## Metrics Collected

For each pair in the panel:

- `delta_task_a` — degradation on task A after merge
- `delta_task_b` — degradation on task B after merge
- `max_delta` — max(delta_task_a, delta_task_b)
- `pair_risk` — Gradience pair-risk label (low/medium/high)
- `dominant_issue` — Gradience dominant-issue classification
- `recon_error` — reconstruction error from merge audit
- `advisory` — whether task-relationship advisory fired
- `classification` — catastrophic / severe / broad / mild (per thresholds above)

For layerwise analysis (Workstream B):

- per-layer verdict from merge audit (SAFE / REDUNDANT / CONFLICTING / IMBALANCED)
- per-layer spectral overlap
- per-layer norm ratio

## Rerun Protocol

### Step 1 — Train source adapters

For each (task, seed, backbone) triple:

```bash
# Template — adjust paths and backbone
python experiments/train_glue_lora.py \
  --model_name {backbone} \
  --task_name {task} \
  --seed {seed} \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir sidecar/results/p01/{backbone}/{task}_s{seed}
```

### Step 2 — Run Gradience audit on each adapter

```bash
gradience audit \
  --adapter sidecar/results/p01/{backbone}/{task}_s{seed} \
  --emit-report sidecar/results/p01/{backbone}/{task}_s{seed}/qa_artifact.json
```

### Step 3 — Run pairwise merge audit

```bash
gradience merge-audit \
  --adapter-a sidecar/results/p01/{backbone}/{task_a}_s{seed_a} \
  --adapter-b sidecar/results/p01/{backbone}/{task_b}_s{seed_b} \
  --qa-a sidecar/results/p01/{backbone}/{task_a}_s{seed_a}/qa_artifact.json \
  --qa-b sidecar/results/p01/{backbone}/{task_b}_s{seed_b}/qa_artifact.json \
  --emit-report sidecar/results/p01/{backbone}/{pair_id}/merge_qa_report.json
```

### Step 4 — Merge and evaluate

```bash
# Linear merge (α=0.5/0.5), evaluate on both task validation sets
python experiments/merge_and_eval.py \
  --adapter-a sidecar/results/p01/{backbone}/{task_a}_s{seed_a} \
  --adapter-b sidecar/results/p01/{backbone}/{task_b}_s{seed_b} \
  --eval-tasks {task_a} {task_b} \
  --output sidecar/results/p01/{backbone}/{pair_id}/eval_results.json
```

### Step 5 — Compile adjudication table

```bash
python sidecar/benchmarks/compile_adjudication.py \
  --results-dir sidecar/results/p01/{backbone} \
  --output sidecar/results/p01/{backbone}/adjudication_results.json
```

## Used By

- **Study S01** — Catastrophic Anchor Replication (Workstream A)
- Study S02 — Layerwise Conflict Contrast (Workstream B, planned)
- Study S03 — Output-Space Incompatibility Probe (Workstream C, planned)
