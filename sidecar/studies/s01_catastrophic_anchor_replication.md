# Study: S01 — Catastrophic Anchor Replication

## Metadata

- **Program:** A — Cross-Task Severity Lab
- **Workstream:** A — Catastrophic anchor replication
- **Status:** active (two-backbone analysis complete; DeBERTa leg pending GPU compute)
- **Date opened:** 2026-03-26
- **Date closed:** —
- **Panels used:** P01 (catastrophic anchor panel)
- **Depends on:** None (founding study)

---

## Question

> Do catastrophic cross-task merge failures replicate across seeds, and do they replicate across backbones — or does the identity of the catastrophic anchor shift in ways that reveal backbone-dependent interference mechanisms?

## Motivation

The existing evidence base contains a striking asymmetry. On DistilBERT, QNLI × MRPC is the catastrophic anchor: the worst seed variant produces a 41.7% performance collapse on MRPC. On RoBERTa, that same task pair is nearly harmless (1.7% worst-case delta), while QNLI × SST-2 becomes the catastrophic anchor instead (27.2% collapse on SST-2).

This is not a marginal instability. It is a qualitative reversal: the pair that produces the worst failure on one backbone produces almost no failure on the other.

If this reversal replicates on a third backbone, it is strong evidence that catastrophic interference is not a property of task-pair identity alone — it is backbone-dependent in a way that current Gradience summary signals do not capture. If the reversal does not replicate (i.e., a third backbone converges with one of the two existing patterns), that is also informative: it would suggest a regime boundary between shallow and deeper encoders.

This study is the empirical foundation for the entire sidecar. Without it, the other workstreams lack a reliable anchor set.

## Design

### What is being compared

Three contrasts, in order of priority:

**Contrast 1 — Seed stability (within-backbone).** For each task pair on each backbone, compare the delta across all seed variants. This asks: is the catastrophic case specific to a particular seed combination, or does it recur across seeds for the same task pair?

**Contrast 2 — Backbone replication.** For each task pair, compare its severity classification across DistilBERT, RoBERTa, and DeBERTa-v3. This asks: does the same task pair produce the same severity class on a new backbone, or does the anchor identity shift again?

**Contrast 3 — Backbone-shift pattern.** If the anchor identity does shift, characterize the shift. Specifically: which pairs escalate, which collapse, and is there any structural property of the backbone (depth, attention mechanism, parameter count) that predicts the direction of the shift?

### Data

**Existing data (no new computation needed):**

- DistilBERT: 29 pairs from `results/cross_task_subtype_study_01/pairs/adjudication_results.json`
- RoBERTa: 29 pairs from `results/task_pair_severity_generalization_study_01/roberta/pairs/adjudication_results.json`

**New data (requires computation):**

- DeBERTa-v3-base: 29 pairs, following the same protocol as the existing studies. This is the primary empirical cost of S01.

DeBERTa-v3 is chosen because it is architecturally distinct from both existing backbones. It uses disentangled attention (content and position are separate), which could interact differently with LoRA weight composition during merge. If QNLI × MRPC is catastrophic on DeBERTa-v3 (as on DistilBERT), that would suggest the mechanism is task-level. If QNLI × SST-2 is catastrophic (as on RoBERTa), that would suggest the mechanism tracks model depth or capacity. If a third pattern emerges, that is the most interesting outcome.

### Adapters to train

Four GLUE tasks × two seeds × one new backbone = **8 new adapters** on DeBERTa-v3-base.

| Adapter | Backbone | Task | Seed |
|---------|----------|------|------|
| deberta_qnli_s42 | deberta-v3-base | QNLI | 42 |
| deberta_qnli_s7 | deberta-v3-base | QNLI | 7 |
| deberta_rte_s42 | deberta-v3-base | RTE | 42 |
| deberta_rte_s7 | deberta-v3-base | RTE | 7 |
| deberta_mrpc_s42 | deberta-v3-base | MRPC | 42 |
| deberta_mrpc_s7 | deberta-v3-base | MRPC | 7 |
| deberta_sst2_s42 | deberta-v3-base | SST-2 | 42 |
| deberta_sst2_s7 | deberta-v3-base | SST-2 | 7 |

### Pairs to evaluate

All 6 cross-task pairs × up to 4 seed combinations = **24 cross-task pairs**, plus 4 same-task controls = **28 total pairs** on DeBERTa-v3 (matching the existing study designs).

### Method

Follow the rerun protocol defined in Panel P01:

1. Train adapters (LoRA r=8, α=16, 3 epochs)
2. Run Gradience audit on each adapter
3. Run pairwise merge audit on all 28 pairs
4. Linear merge (α=0.5/0.5) and evaluate on both tasks
5. Compile adjudication table

All outputs go to `sidecar/results/s01/deberta/`.

### Controls

**Same-task pairs** serve as the primary control. On both existing backbones, same-task pairs show negligible deltas (< 2.2% on DistilBERT, < 1.0% on RoBERTa). If DeBERTa-v3 same-task pairs show significant degradation, that would indicate a training or evaluation problem rather than a merge interference finding.

**Seed variance** is itself a control. A pair classified as "catastrophic" must show worst-case delta > 15% on at least one seed variant. If a task pair is catastrophic on one seed combination but mild on all others, that is worth documenting separately — it may indicate that catastrophic interference is seed-fragile rather than structurally robust.

## Outputs

### Primary deliverables

| Output | Path | Format |
|--------|------|--------|
| DeBERTa adjudication table | `sidecar/results/s01/deberta/adjudication_results.json` | JSON (same schema as existing) |
| Three-backbone comparison table | `sidecar/results/s01/three_backbone_comparison.json` | JSON |
| Seed stability summary | `sidecar/results/s01/seed_stability.json` | JSON |
| Per-pair case dossiers | `sidecar/results/s01/deberta/{pair_id}/` | Directory per pair |

### Secondary deliverables

| Output | Path | Format |
|--------|------|--------|
| Backbone shift table | `sidecar/results/s01/backbone_shift_table.md` | Markdown |
| Anchor replication note | `sidecar/notes/n01_anchor_replication.md` | Note (interpretation) |

## Analysis Plan

### Step 1 — Compile DeBERTa results

Train, merge, evaluate. Produce `adjudication_results.json` in the same schema as the existing studies.

### Step 2 — Classify all DeBERTa pairs

Apply the P01 severity thresholds (catastrophic > 15%, severe 10–15%, broad 5–10%, mild < 5%) to the DeBERTa results. Record the classification for each pair.

### Step 3 — Build three-backbone comparison

For each of the 6 cross-task task pairs, produce a row:

| Task pair | DistilBERT class | DistilBERT worst Δ | RoBERTa class | RoBERTa worst Δ | DeBERTa class | DeBERTa worst Δ |
|-----------|-----------------|-------------------|--------------|----------------|--------------|----------------|

### Step 4 — Assess seed stability

For each (task pair, backbone), compute:

- Range of max_delta across seed variants
- Whether the classification changes across seeds (e.g., catastrophic on one seed combo, severe on another)
- Coefficient of variation of delta across seed combos

### Step 5 — Characterize the backbone shift

Three possible outcomes for each task pair:

- **Stable anchor:** Same severity class on all three backbones. The interference mechanism is task-pair-intrinsic.
- **Backbone-dependent anchor:** Severity class shifts across backbones. The interference mechanism interacts with architecture.
- **Fragile anchor:** Severity class is unstable even within a backbone (high seed variance). The pair is not a reliable anchor.

### Step 6 — Select refined anchor set

Based on steps 3–5, define a refined anchor set for use in subsequent studies:

- **Tier 1 anchors:** Catastrophic on at least 2 of 3 backbones, stable across seeds.
- **Tier 2 anchors:** Catastrophic on 1 backbone and severe on another, or catastrophic but seed-fragile.
- **Tier 3 (non-anchors):** Never catastrophic. May still be useful as contrast cases.

## Predictions

Before running DeBERTa, the study should record explicit predictions to be checked against the results. Based on the existing evidence and the three sidecar hypotheses, the following predictions are offered:

**Prediction 1 (from Hypothesis 3 — backbone dependence):** The catastrophic anchor identity will shift on DeBERTa-v3, rather than matching either existing pattern exactly. Basis: DeBERTa-v3's disentangled attention mechanism is architecturally distinct from both DistilBERT and RoBERTa, so task-pair × backbone interaction should produce a new severity profile.

**Prediction 2 (from Hypothesis 1 — discontinuity):** At least one task pair will show a large gap between its worst and second-worst seed variant — suggesting that catastrophic interference is not smoothly distributed across seeds but has a threshold character.

**Prediction 3 (null hypothesis):** SST-2-involving pairs will tend toward catastrophic on DeBERTa-v3 (as on RoBERTa), because DeBERTa-v3 is a deep 12-layer model like RoBERTa rather than a shallow 6-layer model like DistilBERT. This would support a depth-dependent account.

These predictions are not the study's thesis. They are pre-registered expectations that structure the interpretation of results.

## Preliminary Results (Two-Backbone Phase)

The two-backbone phase is complete. Full analysis in `sidecar/results/s01/` and interpretation in `sidecar/notes/n01_anchor_replication_preliminary.md`.

### Backbone Shift

| Task pair | DistilBERT worst Δ | Class | RoBERTa worst Δ | Class | Shift |
|-----------|--------------------:|-------|----------------:|-------|-------|
| QNLI × MRPC | 41.7% | **catastrophic** | 1.7% | mild | collapses |
| QNLI × SST-2 | 11.0% | severe | 27.2% | **catastrophic** | escalates |
| MRPC × SST-2 | 12.8% | severe | 15.0% | **catastrophic** | escalates |
| RTE × SST-2 | 8.3% | broad | 12.6% | severe | escalates |
| RTE × MRPC | 7.1% | broad | 8.3% | broad | stable |
| QNLI × RTE | 6.4% | broad | 8.3% | broad | stable |

### Seed Stability

The two catastrophic anchors (QNLI×MRPC on DistilBERT, QNLI×SST-2 on RoBERTa) have the highest seed-variant ranges: 28.9% and 26.2% respectively. This suggests catastrophic interference has a threshold character dependent on specific subspace properties, not just task identity.

### Key Preliminary Finding

No task pair is catastrophic on both backbones. The catastrophic anchor identity reverses completely between DistilBERT and RoBERTa. All SST-2-involving pairs escalate on the deeper backbone.

### Figures

- `sidecar/figures/s01_backbone_shift.svg` — Paired bar chart of worst-case deltas by backbone
- `sidecar/figures/s01_seed_stability.svg` — Range plot of seed-variant spread

## DeBERTa-v3 Success Criterion

The DeBERTa leg of S01 is not primarily a severity replication. The question is not "which pair is catastrophic on DeBERTa?" — that answer is expected to differ from both existing backbones. The question is whether **instability rankings are portable**: do the same pairs remain the most unstable, regardless of which pair happens to be catastrophic?

### Three testable predictions

**Prediction A (instability ranking preserved):** QNLI×MRPC and QNLI×SST-2 will have the highest seed ranges on DeBERTa-v3, whether or not either is catastrophic on this backbone. This is the core prediction. If it holds, instability is the first portable cross-backbone merge descriptor.

**Prediction B (stable cluster preserved):** The four stable-asymmetric pairs (MRPC×SST-2, RTE×SST-2, RTE×MRPC, QNLI×RTE) will remain in the low-instability regime on DeBERTa-v3 — seed ranges < 10%, no backbone-reversal behavior.

**Prediction C (gap preserved):** The instability gap between the two clusters will persist. On the two existing backbones, no pair occupies the 0.30–0.74 range. If DeBERTa produces a pair in this gap, the taxonomy needs refinement but the concept may still hold.

### Outcome interpretation

| Outcome | What it means |
|---------|---------------|
| A + B + C all hold | Instability is portable. Strongest possible result. Begin planning promotion path. |
| A holds, B or C fails | Instability is real but the composite score needs recalibration. The concept survives; the operationalization needs work. |
| A fails | Instability is backbone-dependent too. The sidecar's working concept needs fundamental revision. |

### What "holds" means operationally

Prediction A holds if: the two pairs with the highest seed ranges on DeBERTa are QNLI×MRPC and QNLI×SST-2 (in either order), with seed ranges at least 2× the median of the other four pairs.

Prediction B holds if: all four stable-asymmetric pairs have seed ranges below 10% and no pair reverses severity class relative to its two-backbone profile.

Prediction C holds if: no pair has an instability score between 0.30 and 0.70 (using the same composite formula).

## Conclusion

{Pending DeBERTa-v3 replication. See preliminary interpretation in `notes/n01_anchor_replication_preliminary.md`. DeBERTa success criteria defined above and in `notes/n05_instability_as_working_concept.md`.}

## Implication for Core

**Preliminary assessment (pre-DeBERTa):** These findings validate the current core design. Core Gradience stops at boundary detection and does not attempt severity grading — correctly so, because severity grading based on task-pair identity alone would produce catastrophic false confidence when transferred across backbones.

Nothing from the two-backbone phase is promotable to core. The entire finding is about instability of severity signals, which is the sidecar's reason for existing.

**Final assessment pending DeBERTa-v3 results.**

## Open Questions

1. **Why does QNLI × MRPC collapse on DistilBERT but not on RoBERTa?** This is the highest-priority follow-up question. Possible directions: layerwise conflict contrast (Study S02), output-space probe (Study S03).

2. **Is DeBERTa's disentangled attention a relevant architectural variable?** If DeBERTa shows a unique catastrophic pattern, the disentangled attention mechanism may be interacting with LoRA composition in a way that standard scaled dot-product attention does not.

3. **Is seed fragility informative?** If a pair is catastrophic on one seed combo but mild on others, that seed sensitivity itself is a signal — it may indicate that the interference depends on the precise learned subspace, not just the task identity.

4. **What happens at different LoRA ranks?** P01 fixes r=8. A follow-up panel could vary rank (r=4, r=16, r=32) to test whether catastrophic interference depends on the dimensionality of the adapter subspace.
