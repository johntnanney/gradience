# Catastrophic Anchor Dossier: QNLI × SST-2 on RoBERTa

## Metadata

- **Anchor ID:** CA-02
- **Date:** 2026-03-26
- **Related studies:** S01
- **Instability program role:** Second catastrophic anchor. Demonstrates the backbone-reversal phenomenon from the opposite direction: catastrophic on RoBERTa, severe (not catastrophic) on DistilBERT. Contains the strongest single-adapter culprit identification in the evidence base.

---

## Identity

- **Task pair:** QNLI (question NLI) × SST-2 (sentiment classification)
- **Backbone:** roberta-base (12 transformer layers, standard attention)
- **Taxonomy class:** backbone reversal
- **Instability score:** 0.74 (rank 2 of 6)

---

## Severity Profile

| Seed variant | Δ QNLI | Δ SST-2 | Max Δ | Severity class |
|--------------|-------:|--------:|------:|----------------|
| s42 × s7 | 8.4% | **27.2%** | 27.2% | **catastrophic** |
| s42 × s42 | 8.2% | **17.6%** | 17.6% | catastrophic |
| s7 × s7 | −4.8% | 2.8% | 2.8% | mild |
| s7 × s42 | 0.2% | 1.0% | 1.0% | mild |

**Seed range:** 26.2 percentage points (27.2% − 1.0%)
**CV:** 0.89 (highest of any pair on either backbone)
**Worst variant:** s42 × s7 (qnli_s42 merged with sst2_s7)
**Best variant:** s7 × s42

---

## What Core Signals Said

| Signal | Value | Would it have predicted catastrophic? |
|--------|-------|--------------------------------------|
| pair_risk | not available in RoBERTa data | N/A |
| dominant_issue | not available | N/A |
| reconstruction_error | not available | N/A |
| task_relationship_advisory | present (different tasks) | **Yes,** but only binary. |
| source QA eligibility (QNLI) | eligible | N/A |
| source QA eligibility (SST-2) | eligible | N/A |

**Verdict:** Core signals were not fully available for the RoBERTa adjudication data (different experimental setup). The advisory would have correctly flagged this as cross-task but could not have distinguished it from any other cross-task pair.

---

## Cross-Backbone Behavior

| Backbone | Worst Δ | Class | Seed range |
|----------|--------:|-------|----------:|
| DistilBERT | 11.0% | severe | 2.4% |
| RoBERTa | 27.2% | **catastrophic** | 26.2% |
| DeBERTa | — | — | — |

**Backbone shift:** 16.2 percentage points
**Reversal?** Yes — escalation from severe to catastrophic, with massive seed-range amplification.

---

## Mechanistic Observations

### Victim pattern

SST-2 is the primary victim in both catastrophic variants. In the worst case (s42 × s7), SST-2 drops from 89.4% to 62.2% — a 27.2-point collapse. QNLI degrades more moderately (8.4%). In the s42 × s42 variant, SST-2 drops from 89.6% to 72.0%.

SST-2 is the higher-scoring source (89.4–89.6% vs. QNLI's ~79–82%). As with CA-01, the stronger adapter is the victim. The pattern reinforces the hypothesis that concentrated, high-quality representations are more vulnerable to disruption.

SST-2 is a single-sentence binary classification task. Its decision boundary may be simpler and therefore more fragile — a narrow feature subspace that, when contaminated by QNLI's broader multi-sentence reasoning features, loses discriminative power.

### Culprit pattern

**This is the clearest culprit identification in the evidence base.** The catastrophic behavior is cleanly driven by **qnli_s42**:

- Variants involving qnli_s42 (s42×s7, s42×s42): catastrophic (27.2%, 17.6%)
- Variants involving qnli_s7 (s7×s7, s7×s42): mild (2.8%, 1.0%)

The SST-2 seed does not matter much: switching SST-2's seed while keeping qnli_s42 changes the delta from 27.2% to 17.6% (still catastrophic). Switching QNLI's seed while keeping SST-2 fixed changes the delta from 27.2% to 1.0% (catastrophe → benign).

**qnli_s42 on RoBERTa is a "toxic adapter"** — its learned subspace has properties that destroy SST-2's signal when merged. qnli_s7, trained on the same task with the same configuration, does not. The entire catastrophe is attributable to the random initialization of a single adapter.

### Architectural hypotheses

**SST-2 escalation pattern.** All four SST-2-involving pairs escalate on RoBERTa relative to DistilBERT. The deeper backbone may allocate more representational capacity to sentiment features, creating a larger "surface area" for cross-task interference. Alternatively, RoBERTa's larger model may learn more task-specific (less shared) representations, making cross-task merging more disruptive.

**Disentangled attention interaction (DeBERTa).** DeBERTa uses separate content and position representations. If QNLI's learned features are more position-dependent (sentence-pair reasoning often involves positional structure) while SST-2's are more content-dependent, the disentangled architecture could either insulate the two tasks (reducing interference) or create novel interaction modes.

---

## Open Questions

1. **What is structurally different about qnli_s42 vs. qnli_s7 on RoBERTa?** Both are trained on the same task with identical configuration. Only the random seed differs. Yet one destroys SST-2 and the other is harmless. This is the single sharpest question in the sidecar — answering it would be a direct test of the thresholded subspace interference hypothesis. Workstream B should prioritize this contrast.

2. **Does the catastrophic interference concentrate in specific layers?** If qnli_s42's "toxic" property is localized to a few layers, a future structural predictor could detect it without running the merge. If it is distributed, the problem is harder.

3. **Does SST-2's single-sentence format make it structurally more vulnerable?** SST-2's task involves classifying individual sentences — a simpler input structure than QNLI's question-passage pairs. If the vulnerability correlates with input-structure simplicity, that would be an architectural explanation for why SST-2 is the victim.

4. **Will the culprit specificity replicate on DeBERTa?** If qnli_s42 is again the catastrophic adapter on DeBERTa while qnli_s7 is benign, the "toxic adapter" phenomenon is backbone-portable. That would be a major finding — it would mean specific learned subspaces, not just task identities, determine catastrophic risk.

---

## DeBERTa Prediction

**Expected behavior:** The SST-2 escalation pattern and the qnli_s42 culprit pattern together predict that this pair will show elevated severity on DeBERTa, potentially catastrophic. The seed range should remain high, concentrated in variants involving qnli_s42. This is the pair most likely to produce a DeBERTa catastrophe.

**Confidence:** Medium. The escalation pattern is strong (all SST-2 pairs escalate on the deeper backbone), but DeBERTa's disentangled attention is genuinely novel and could disrupt the pattern.

**What would be surprising:** If QNLI×SST-2 is mild on DeBERTa across all seed variants, the SST-2 escalation pattern would be specific to standard attention and would not generalize to disentangled architectures. That would be informative: it would isolate the attention mechanism as a variable in cross-task interference.
