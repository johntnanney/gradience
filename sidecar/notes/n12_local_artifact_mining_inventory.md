# Note: Local Artifact Mining — Inventory and Preliminary Findings

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related studies:** S01
- **Project:** Phase 3, Project G — Local Artifact Mining

---

## Purpose

This note inventories the existing data assets that can be mined for structural patterns without new GPU compute, and records preliminary findings from the analysis already performed. The goal is to identify whether per-pair or per-backbone structural features contain information that the instability program has not yet exploited.

---

## 1. Available Data Assets

### 1.1 Cross-task adjudication data

| Asset | Path | Backbone | Content |
|-------|------|----------|---------|
| DistilBERT analysis | `results/s01/distilbert_analysis.json` | DistilBERT | 24 cross-task + 4 same-task pairs, per-variant deltas, pair_risk, dominant_issue |
| RoBERTa analysis | `results/s01/roberta_analysis.json` | RoBERTa | 24 cross-task + 4 same-task pairs, per-variant deltas |

**Available fields per variant:**
- `delta_task_a`, `delta_task_b`: per-task degradation
- `max_delta`: worst-case delta
- `classification`: severity class
- `pair_risk`: core Gradience pair-risk label (DistilBERT only)
- `dominant_issue`: core Gradience dominant-issue label (DistilBERT only)

**Not available:** per-layer weight norms, per-layer overlap scores, reconstruction error details. These would require re-running `gradience audit` and `gradience merge-audit` on the saved adapter weights.

### 1.2 Derived analysis outputs

| Asset | Path | Content |
|-------|------|---------|
| Instability profiles | `results/s01/instability_profiles.json` | Per-pair composite scores, per-backbone stats |
| Seed stability | `results/s01/seed_stability.json` | Per-pair seed-variant details |
| Three-backbone comparison | `results/s01/three_backbone_comparison.json` | Cross-backbone severity table (two-backbone) |
| Taxonomy | `results/s01/taxonomy.json` | Pair classification |
| Regime summaries | `results/s01/regime_summaries.json` | Per-regime statistics |
| Same-task contrast | `results/s01/same_task_contrast.json` | All same-task controls |
| Backbone shift table | `results/s01/backbone_shift_table.md` | Formatted comparison |

### 1.3 Saved adapter weights (potential for deeper analysis)

Per the evidence atlas (n04 §5): "Adapter weights (.safetensors) exist for all blind-spot studies and cross-task studies on DistilBERT and RoBERTa." If these weights are accessible, CPU-only layerwise analysis is possible:

- Per-layer LoRA weight norms
- Per-layer rank profiles (SVD of LoRA matrices)
- Per-layer subspace overlap between pair members
- Per-layer norm concentration (how many layers carry the bulk of the adaptation)

This would be the richest vein for Project G, but requires locating and loading the saved weights.

---

## 2. Preliminary Findings from Existing Data

### 2.1 Task-asymmetry in degradation

Across all cross-task pairs, the degradation is asymmetric: one task degrades more than the other. The degree of asymmetry can be quantified as the ratio of the two deltas:

**DistilBERT cross-task pairs (worst seed variant each):**

| Pair | Δ task A | Δ task B | Asymmetry ratio |
|------|--------:|--------:|----------------:|
| QNLI × MRPC | 21.6% | 41.7% | 1.93 |
| QNLI × SST-2 | 11.0% | 9.4% | 1.17 |
| MRPC × SST-2 | 12.8% | 3.0% | 4.27 |
| QNLI × RTE | 2.5% | 6.4% | 2.56 |
| RTE × MRPC | 7.1% | 3.7% | 1.92 |
| RTE × SST-2 | 2.2% | 8.3% | 3.77 |

The catastrophic pair (QNLI×MRPC) has a moderate asymmetry ratio (1.93). The highest asymmetry is MRPC×SST-2 (4.27) — MRPC degrades 4× more than SST-2. But MRPC×SST-2 is not catastrophic.

**Finding:** Asymmetry ratio does not predict catastrophic status. Catastrophic pairs are not the most asymmetric; they are the most *variable*.

### 2.2 Seed-variant correlation structure

For the two backbone-reversal pairs, examining whether the same seed variant is always worst:

**QNLI × MRPC on DistilBERT:**
- Worst: s42×s7 (41.7%) — qnli_s42 as adapter A, mrpc_s7 as adapter B
- Best: s7×s7 (12.7%)

**QNLI × SST-2 on RoBERTa:**
- Worst: s42×s7 (27.2%) — qnli_s42 as adapter A, sst2_s7 as adapter B
- Best: s7×s42 (1.0%)

**Observation:** qnli_s42 appears in the worst variant of *both* catastrophic anchors (on different backbones). This could be coincidence — s42 is just one of two seeds, so it has a 50% chance of appearing in any given worst variant. But combined with the sharp culprit identification in CA-02, it suggests qnli_s42 may have systematically more "interfering" subspace geometry across backbones.

**Testable on DeBERTa:** If qnli_s42 variants are consistently the worst on DeBERTa as well, the "toxic adapter" hypothesis gains significant support.

### 2.3 Core signal uniformity on DistilBERT

On DistilBERT, where Gradience core signals are available:

| Pair | pair_risk | dominant_issue |
|------|-----------|---------------|
| QNLI × MRPC | medium | partial_redundancy |
| QNLI × SST-2 | medium | partial_redundancy |
| MRPC × SST-2 | medium | partial_redundancy |
| QNLI × RTE | medium | partial_redundancy |
| RTE × MRPC | medium | partial_redundancy |
| RTE × SST-2 | medium | partial_redundancy |

**Every cross-task pair receives identical core signals.** The pair_risk and dominant_issue labels are completely non-discriminating within the cross-task regime on DistilBERT. This is not a failure of the signals but a confirmation that they were designed for boundary detection (which they do perfectly) rather than within-regime grading.

### 2.4 Same-task regime: backbone-dependent tightness

| Backbone | Mean same-task Δ | Max same-task Δ | N pairs at Δ = 0 |
|----------|-----------------:|----------------:|------------------:|
| DistilBERT | 0.8% | 2.2% | 0 |
| RoBERTa | 0.1% | 1.0% | 3 |

RoBERTa's same-task merges are essentially lossless (3 of 4 pairs at exactly 0% delta). DistilBERT's are safe but measurably noisy. This suggests that merge noise floor scales inversely with model capacity.

**Implication for DeBERTa:** Expect DeBERTa same-task controls to be similar to RoBERTa (near-zero delta).

---

## 3. Artifacts That Could Be Mined With More Work

### 3.1 Per-layer weight analysis (requires saved adapter files)

If saved `.safetensors` files are locatable, the following can be computed on CPU:

- **Norm concentration:** For each adapter, compute the L2 norm of LoRA_A and LoRA_B matrices per layer. Determine whether the adaptation is concentrated in a few layers or distributed. Hypothesis: unstable pairs may involve adapters with *concentrated* adaptation (most weight in a few layers), making them more vulnerable to seed-dependent interference.

- **Subspace overlap:** For each pair, compute the principal angle (or similar) between the two adapters' LoRA subspaces at each layer. Hypothesis: catastrophic seed variants will show high overlap (subspace collision) in specific layers that mild variants do not.

- **Effective rank profile:** Compute the effective rank (e.g., rank-90 energy) of each adapter per layer. Compare catastrophic-pair adapters to stable-pair adapters. Hypothesis: unstable pairs may involve adapters with very different effective ranks, creating an "imbalanced merge" in specific layers.

### 3.2 Reconstruction error disaggregation

Reconstruction error (available on DistilBERT) is 0.207 for the catastrophic pair — within the normal range. But this is a global summary. If the raw per-layer reconstruction errors are stored, they could reveal whether the catastrophic pair has an unusual *distribution* of error across layers, even if the total is unremarkable.

### 3.3 Cross-task delta decomposition

For each pair, decompose the worst-case delta into:
- **Contribution from task A's degradation** vs. **task B's degradation**
- **Contribution from each layer** (requires per-layer merge analysis, which is more expensive)

This would test whether catastrophic failures concentrate in specific layers or are distributed.

---

## 4. Mining Priority Ranking

| Analysis | CPU-feasible? | Requires | Expected yield | Priority |
|----------|:----------:|----------|---------------|----------|
| qnli_s42 cross-anchor pattern | Yes (from existing data) | Nothing new | Medium — confirms or denies cross-anchor culprit pattern | **Done** (§2.2) |
| Core signal uniformity | Yes (from existing data) | Nothing new | Low — confirms known limitation | **Done** (§2.3) |
| Task asymmetry analysis | Yes (from existing data) | Nothing new | Low — asymmetry does not predict catastrophe | **Done** (§2.1) |
| Per-layer norm concentration | Yes (if weights accessible) | Saved adapter files | High — could reveal structural predictor | **Top priority** |
| Per-layer subspace overlap | Yes (if weights accessible) | Saved adapter files | High — direct test of thresholded interference | **Top priority** |
| Effective rank profiles | Yes (if weights accessible) | Saved adapter files | Medium — may reveal imbalance pattern | Second priority |
| Per-layer reconstruction error | Partially (DistilBERT only) | Saved audit outputs | Medium — may reveal local error concentration | Second priority |

---

## 5. What This Note Establishes

The existing data has been mined for the patterns extractable without new compute. Three findings are useful:

1. **Task-asymmetry does not predict catastrophe** — ruling out a possible confound.
2. **qnli_s42 appears in both catastrophic anchors' worst variants** — strengthening the toxic-adapter hypothesis.
3. **Core signals are completely non-discriminating within the cross-task regime** — confirming the target-variable problem.

The highest-value next step for Project G is locating saved adapter weight files and performing per-layer structural analysis. This would directly test the thresholded subspace interference hypothesis and could yield the first structural predictor of instability class.

---

## Next Steps

1. Locate saved adapter `.safetensors` files for DistilBERT and RoBERTa studies
2. If found: write a CPU-only analysis script to compute per-layer norms, effective ranks, and subspace overlaps
3. Compare structural profiles of catastrophic vs. stable seed variants (the CA-02 qnli_s42 vs. qnli_s7 contrast is the sharpest test case)
4. Document findings in a follow-up mining note
