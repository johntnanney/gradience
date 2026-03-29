# Note: Catastrophic Anchor Case Dossiers

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related panels:** P01

---

## Summary

Two catastrophic anchors exist in the current evidence base. They are each confined to a single backbone, and they swap roles between backbones. This note provides a concise dossier for each.

---

## Dossier 1 — QNLI × MRPC on DistilBERT

**Pair identity:** QNLI (question NLI) × MRPC (paraphrase detection)
**Backbone:** distilbert-base-uncased (6 transformer layers)
**Taxonomy:** backbone reversal (catastrophic on DistilBERT, mild on RoBERTa)

### Severity profile

| Seed variant | Δ task A (QNLI) | Δ task B (MRPC) | Max Δ | Class |
|--------------|----------------:|----------------:|------:|-------|
| s42 × s7 | 21.6% | **41.7%** | 41.7% | **catastrophic** |
| s7 × s42 | **13.6%** | 4.9% | 13.6% | severe |
| s42 × s42 | 13.2% | 11.0% | 13.2% | severe |
| s7 × s7 | 8.8% | 12.7% | 12.7% | severe |

### Key observations

**Seed sensitivity:** The range across seeds is 28.9 percentage points (CV = 0.61). The worst variant (s42×s7) is 3× worse than the best (s7×s7). This is the highest seed variance of any pair on either backbone.

**Asymmetry:** On the worst variant, MRPC collapses much more than QNLI (41.7% vs 21.6%). MRPC drops from 73.5% to 31.9% — near chance. On other seed variants, QNLI degrades more than MRPC.

**Structural signals:** pair_risk = medium, dominant_issue = partial_redundancy. These are the same labels applied to many non-catastrophic pairs. None of Gradience's current signals distinguished this pair as catastrophic in advance.

**Reconstruction error:** 0.207 — indistinguishable from other pairs (range 0.12–0.28 across all pairs).

### Backbone reversal

On RoBERTa, this same task pair shows worst-case Δ = 1.7% (mild). The entire 41.7% signal disappears. QNLI×MRPC on RoBERTa is one of the most benign cross-task pairs in the evidence base.

### What this rules out

Task-pair identity alone cannot predict catastrophic interference. The pair that is most dangerous on one backbone is among the safest on another. Any severity feature based on "QNLI × MRPC is a bad combination" would be catastrophically wrong on RoBERTa.

### Open questions

1. What changes between the s42×s7 and s7×s7 seed variants that produces a 29% difference? The task pair is the same; only the learned LoRA subspaces differ.
2. Why is MRPC the primary victim on the worst variant, when MRPC has the higher source score (73.5% vs 69.0%)?
3. Does the shallow architecture of DistilBERT (6 layers) create conditions where QNLI and MRPC subspaces are forced to compete in fewer layers?

---

## Dossier 2 — QNLI × SST-2 on RoBERTa

**Pair identity:** QNLI (question NLI) × SST-2 (sentiment classification)
**Backbone:** roberta-base (12 transformer layers)
**Taxonomy:** backbone reversal (catastrophic on RoBERTa, severe on DistilBERT)

### Severity profile

| Seed variant | Δ task A (QNLI) | Δ task B (SST-2) | Max Δ | Class |
|--------------|----------------:|-----------------:|------:|-------|
| s42 × s7 | 8.4% | **27.2%** | 27.2% | **catastrophic** |
| s42 × s42 | 8.2% | **17.6%** | 17.6% | catastrophic |
| s7 × s7 | −4.8% | 2.8% | 2.8% | mild |
| s7 × s42 | 0.2% | 1.0% | 1.0% | mild |

### Key observations

**Extreme seed sensitivity:** The range across seeds is 26.2 percentage points (CV = 0.89 — the highest CV of any pair on either backbone). Variants involving qnli_s42 are catastrophic; variants involving qnli_s7 are mild. The catastrophic behavior appears to be driven by a property of qnli_s42's learned subspace.

**SST-2 is the victim:** In both catastrophic variants, SST-2 collapses (89.4% → 62.2%, 89.6% → 72.0%) while QNLI shows moderate degradation. SST-2 is the higher-scoring source, so the higher-quality adapter is the one destroyed.

**QNLI_s42 is the culprit:** The seed pattern strongly implicates qnli_s42 as the interfering source. When qnli_s7 is used instead, the pair is benign regardless of SST-2's seed.

**Structural signals:** pair_risk not available in RoBERTa adjudication data; standard merge audit signals would not have flagged this pair specifically.

### Backbone escalation

On DistilBERT, QNLI×SST-2 shows worst-case Δ = 11.0% (severe, not catastrophic). The pair escalates from severe to catastrophic on the deeper backbone. This is part of a systematic pattern: all four SST-2-involving pairs escalate on RoBERTa.

### What this suggests

SST-2's representational footprint may expand on deeper models, creating a larger "target surface" for cross-task interference to disrupt. The fact that qnli_s42 (but not qnli_s7) triggers this disruption suggests the mechanism depends on the specific learned subspace, not just the task.

### Open questions

1. What is structurally different about qnli_s42 vs qnli_s7 on RoBERTa? Both are trained on the same task with the same config. Only the random seed differs.
2. Does the catastrophic interference concentrate in specific layers? This is a direct question for Workstream B.
3. Does SST-2's single-sentence format make it more vulnerable — perhaps because its decision boundary is simpler and therefore more fragile when disrupted?

---

## Cross-Dossier Patterns

### What the two anchors share

1. **Extreme seed sensitivity.** Both anchors have seed ranges > 25 percentage points. This is 5–10× larger than any stable pair's seed range.
2. **Asymmetric victim pattern.** In both cases, one task collapses while the other shows moderate degradation. The victim is the higher-scoring source in both cases.
3. **Current signals did not predict them.** Neither pair_risk, dominant_issue, reconstruction_error, nor core-space shared-basis distinguished these pairs from non-catastrophic ones.

### What differs

1. **Backbone.** The two anchors live on different backbones. Neither is a general catastrophic pair.
2. **Task roles.** In Dossier 1, MRPC is the victim. In Dossier 2, SST-2 is the victim. There is no single "vulnerable task" across backbones.
3. **Culprit specificity.** In Dossier 2, a single source adapter (qnli_s42) is clearly implicated. In Dossier 1, the seed sensitivity is more diffuse.

### The strongest inference

Catastrophic cross-task interference is not a property of task pairs. It is a property of specific (task pair × backbone × seed) triples. This makes it fundamentally harder to predict from summary-level features and explains why no severity signal has generalized.

The sidecar's next steps should focus on what makes these specific triples catastrophic: layerwise analysis (Workstream B) and output-space probes (Workstream C).
