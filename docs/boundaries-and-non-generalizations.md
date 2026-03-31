# Boundaries and Non-Generalizations

**Last updated:** 2026-03-31

What didn't work, what doesn't generalize, and what you should not assume.

This document exists to prevent overclaiming. Every finding here was tested with real experiments and rejected or narrowed based on evidence. The eliminations are as important as the positive findings — they define the shape of what is actually true.

---

## The three big non-generalizations

### 1. No structural metric is fully portable across artifact classes

**Tested:** LoRA, LoHa (shimmed), checkpoint delta (summary representation). Nine-case panel across all three artifact classes.

**Result:** Zero structural metrics are fully portable.

| Signal | LoRA | LoHa | Checkpoint delta |
|--------|------|------|-----------------|
| V-module dimensionality ratio (d=3.36) | Yes | Yes (shimmed) | No — requires factor geometry |
| Subspace overlap / principal angles | Yes | Yes (shimmed) | No |
| Compatibility score | Yes (0.47 same-task) | Yes (different scale) | Yes (0.89 same-task) — **not comparable** |
| Pair risk labels (low/medium/high) | Yes | Yes | Yes — **different thresholds** |
| Merge strategy recommendation | Yes | Yes | No — merge out of scope |

The V-module signal — the strongest catastrophe discriminator in the entire research program — cannot be computed from summary-based representations. Any claim that "Gradience detects catastrophic risk across artifact classes" would be false at the structural-metric level.

**What is portable:** Evidence gating, conservative narrowing, task-relationship ordering. All workflow-level, not metric-level.

**Source:** `sidecar/notes/n78_representation_local_signal_audit.md`, `sidecar/results/cross_artifact_portability/local_signal_table.json`

### 2. Severity is not portable across backbones

**Tested:** Same task pairs evaluated on DistilBERT and RoBERTa.

**Result:** Severity rankings reverse completely.

| Pair | DistilBERT | RoBERTa |
|------|-----------|---------|
| QNLI x MRPC | 41.7% degradation (catastrophic) | 1.7% (mild) |
| QNLI x SST-2 | 0.3% (safe) | 30.3% (catastrophic) |

No task pair is catastrophic on both tested backbones. The unit of catastrophe is not the task pair but the (task pair x backbone x seed) triple.

Six candidate severity signals were tested (task-pair identity, core-space shared basis, pair-risk label, format similarity, source-strength gap, reconstruction error). All failed to predict severity portably.

**What replaced it:** Instability — the variability of severity across conditions. Instability rankings are consistent across backbones, with a clean gap between unstable and stable clusters.

**Source:** `sidecar/notes/n68_ruled_out_mechanisms.md` (hypotheses 1–2)

### 3. Context and aggregation override structure

**Tested:** Four aggregation families (worst-case, distributional, QA-dominant, hybrid) applied to the same 12-case structural panel.

**Result:** Only 2/12 cases produce the same judgment across all aggregation rules. The remaining 10 change.

- Worst-case collapses the routing gradient (confusable / moderate / separable) to one label
- QA-dominant blocks the highest-compatibility pair (0.892) because both sources lack evidence
- Distributional preserves gradients that worst-case erases

The same geometric fact becomes operative or inoperative depending on whether the user is merging, routing, or triaging — and whether evidence exists.

**Source:** `sidecar/notes/n85_aggregation_sensitive_operational_implications.md`

---

## Specific negative results

### Low-rank approximation for checkpoint deltas: too lossy

**Tested:** Truncated SVD at CPU-feasible ranks k=4, 8, 16 on distilbert-base-uncased checkpoint deltas.

| Rank | Mean retention | Mean reconstruction error |
|------|---------------|--------------------------|
| k=4 | 0.486 | 0.692 |
| k=8 | 0.558 | 0.641 |
| k=16 | 0.638 | 0.578 |

Disposition: **rejected.** Even at k=16, nearly half the signal is lost. Ring 2 proceeded with layerwise summary statistics (Representation C), accepting reduced operationality for CPU-feasible stability.

**Source:** `experiments/ring2_checkpoint_delta/stage_a_representation_results.json`, `docs/design/ring2_stage_d_assessment_memo.md`

### Readout orthogonality as risk marker: falsified

**Tested:** Whether near-orthogonal classifier decision axes predict dangerous merge outcomes.

**Evidence against:** 5 of 14 same-task seed pairs show orthogonal readout (cos ~ 0), yet all merge safely (max degradation 2.2%). A standalone readout-cosine metric would false-alarm on ~40% of same-task merges.

**Independent falsifier:** QNLI x MRPC on RoBERTa has identical readout geometry to the catastrophic CA-01 case on DistilBERT (~89 degrees, ~0.70 margin proxy) but is safe (1.7% degradation).

Readout orthogonality is common, bimodally distributed, and harmless in isolation.

**Source:** `sidecar/notes/n68_ruled_out_mechanisms.md` (hypotheses 4–5)

### Aggregate Q/K/V/O geometry: dilutes the signal

**Tested:** Concatenated attention-module geometry (all four modules together) as a catastrophe threshold.

**Result:** When backbone is controlled, catastrophic cases are indistinguishable from safe cases on all four metrics (principal angles, top-direction overlap, dimensionality ratio, directional conflict). The separation visible in raw data was a backbone confound.

**Why it failed:** Concatenation averages V-module signal (which carries catastrophe discrimination, d=3.36) with Q and O noise. Per-module decomposition recovered the signal that aggregate analysis had erased.

**Source:** `sidecar/notes/n68_ruled_out_mechanisms.md` (hypothesis 3)

### Feature plurality as universal attractor origin: partially falsified

**Tested:** Whether multi-attractor readout structure arises because different seeds lock onto different pretrained feature sets.

**Result:** Two qualitatively different mechanisms, not one:
- **Rotational degeneracy** (all DistilBERT cases) — seeds use the same principal components but combine them in orthogonal angular orientations. Same features, different combinations.
- **Feature-set switching** (only QNLI on RoBERTa) — seeds genuinely use different PCs (shared top-3 = 0, energy overlap = 0.255). Different features entirely.

The simple feature-plurality hypothesis does not apply universally.

**Source:** `sidecar/notes/n68_ruled_out_mechanisms.md` (hypothesis 6)

### Routing-confusability: no merge-visible behavioral signature

**Tested:** Whether routing-confusable pairs (high overlap, similar accuracy, same task) produce confusion-like behavior when merged.

**Result:** NM-01 (the routing-confusable case) is behaviorally indistinguishable from safe merges on all discriminating metrics: neither-source <2%, zero confidence collapse, zero high-confidence wrong.

**Interpretation:** Routing-confusability is a structural property that likely only manifests as confusion in actual routing scenarios, not in merge. Whether it does in a routing setting is untested.

**Source:** `sidecar/notes/n88_behavioral_route2_findings.md` (H4)

---

## Scope limits on positive findings

These are real findings with real boundaries. Don't extend them past their evidence base.

### Numeric thresholds are descriptive, not calibrated

The following thresholds describe the current panel. They should not be hardcoded:

| Threshold | Source | Panel size |
|-----------|--------|-----------|
| <2% neither-source (safe tier) | 8 LoRA cases, 4000 examples | Small |
| ~14% neither-source (pathology tier) | 8 LoRA cases, 4000 examples | Small |
| d=3.36 V-module separation | 2 backbones, LoRA only | Small |
| 5 aggregation-sensitive patterns | 12-case panel | Small |
| 3 behavioral tiers | 8-case panel | Small |

### Evidence bootstrap is a prerequisite, not enrichment

Without behavioral evidence, Gradience produces nothing useful. Pilot 1 ran 4 adapters through the full pipeline without evaluation scores. Every adapter was classified as `unknown_no_behavioral_eval`. The structural analysis was technically correct and operationally useless.

**Source:** `docs/product-validation.md`

### Spectral analysis is underexercised at rank 1

Most field trial adapters are r=1 TransferGraph models. At rank 1, energy-rank profiles, utilization patterns, and multi-rank interactions are invisible. The spectral layer's distinctive contribution is clearest on same-task pairs with rank >= 8.

**Source:** `docs/product-validation.md`

### Near-miss is validated for LoRA only

Near-miss as a behavioral category (indistinguishable from safe) is well-validated in LoRA (7 pairs, 3 backbones, avg degradation -0.006). It has not been observed in LoHa or checkpoint delta — those panels were too small and homogeneous to produce middle states. The absence is inconclusive, not falsifying.

**Source:** `sidecar/notes/n80_cross_artifact_product_relevance.md`

### Merge execution remains LoRA-specific

Ring 1 (LoHa) validated audit and triage but did not test merge execution. Ring 2 (checkpoint delta) explicitly states merge execution is out of scope. The claim is: "a LoRA-native merge tool with a PEFT-general audit and triage substrate."

**Source:** `docs/strategy/ring1_peft_generalization_results.md`, `docs/design/ring2_stage_d_assessment_memo.md`

### Norm imbalance loses discriminative power at scale

In field trials, 75–100% of pairs showed norm imbalance as the dominant structural issue, driven by rank heterogeneity in public adapters (r=1 vs r=4–16). When the majority share the same label, the label stops being informative.

**Source:** `docs/product-validation.md`

---

## Summary table

| Finding | Status | What to say | What not to say |
|---------|--------|------------|----------------|
| Cross-artifact structural metrics | Not portable | "Workflow-level signals transfer" | "Gradience detects risk across artifact classes" |
| Severity across backbones | Not portable | "Instability is consistent" | "This pair is catastrophic" |
| Aggregation | Not invariant | "Depends on decision context" | "The compatibility score is X" |
| Low-rank checkpoint deltas | Rejected | "Summary-based representation" | "Factor analysis works on checkpoints" |
| Readout orthogonality | Harmless alone | "Readout is a background condition" | "Orthogonal readout means danger" |
| Routing-confusability behavior | No merge signature | "Structural property" | "Confusable pairs behave differently when merged" |
| Numeric thresholds | Panel-specific | "In our panel, we observed..." | "The boundary is at 2% / 14%" |
| Near-miss cross-artifact | Unvalidated | "Validated for LoRA" | "Near-miss is safe across PEFT methods" |
| Evidence bootstrap | Prerequisite | "Required for any output" | "Optional enrichment" |
| Merge execution | LoRA-only | "Triage generalizes; merge doesn't" | "Gradience merges any PEFT adapter" |

---

## Where the full negative results live

- **Ten ruled-out hypotheses:** `sidecar/notes/n68_ruled_out_mechanisms.md` (primary) and `sidecar/packet/03_ruled_out.md` (packet copy)
- **Cross-artifact non-portability:** `sidecar/results/cross_artifact_portability/local_signal_table.json`
- **Aggregation non-invariance:** `sidecar/results/aggregation_sensitive_compatibility/pattern_taxonomy.json`
- **Ring 2 rejection data:** `experiments/ring2_checkpoint_delta/stage_a_representation_results.json`
- **Product limitations:** `docs/product-validation.md` (sections 4 and 6)
- **Route 2 boundaries:** `sidecar/notes/n93_route2_synthesis.md` (section 7)
