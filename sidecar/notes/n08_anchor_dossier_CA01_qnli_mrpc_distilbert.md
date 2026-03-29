# Catastrophic Anchor Dossier: QNLI × MRPC on DistilBERT

## Metadata

- **Anchor ID:** CA-01
- **Date:** 2026-03-26
- **Related studies:** S01
- **Instability program role:** Primary catastrophic anchor. The single most extreme merge failure in the evidence base. Backbone-reversal exemplar — catastrophic on DistilBERT, mild on RoBERTa.

---

## Identity

- **Task pair:** QNLI (question NLI) × MRPC (paraphrase detection)
- **Backbone:** distilbert-base-uncased (6 transformer layers, standard attention)
- **Taxonomy class:** backbone reversal
- **Instability score:** 0.87 (rank 1 of 6)

---

## Severity Profile

| Seed variant | Δ QNLI | Δ MRPC | Max Δ | Severity class |
|--------------|-------:|-------:|------:|----------------|
| s42 × s7 | 21.6% | **41.7%** | 41.7% | **catastrophic** |
| s7 × s42 | **13.6%** | 4.9% | 13.6% | severe |
| s42 × s42 | 13.2% | 11.0% | 13.2% | severe |
| s7 × s7 | 8.8% | 12.7% | 12.7% | severe |

**Seed range:** 28.9 percentage points (41.7% − 12.7%)
**CV:** 0.61
**Worst variant:** s42 × s7 (qnli_s42 merged with mrpc_s7)
**Best variant:** s7 × s7

---

## What Core Signals Said

| Signal | Value | Would it have predicted catastrophic? |
|--------|-------|--------------------------------------|
| pair_risk | medium | **No.** Same label as many non-catastrophic pairs. |
| dominant_issue | partial_redundancy | **No.** Same label as benign pairs. |
| reconstruction_error | 0.207 | **No.** Within normal range (0.12–0.28). |
| task_relationship_advisory | present (different tasks) | **Yes,** but only binary — would not have distinguished this from any other cross-task pair. |
| source QA eligibility (QNLI) | eligible | N/A (source is fine) |
| source QA eligibility (MRPC) | eligible | N/A (source is fine) |

**Verdict:** No core signal distinguishes CA-01 from non-catastrophic cross-task pairs. The advisory correctly flags it as cross-task (caution zone), but every cross-task pair gets the same advisory. The core signals that should differentiate *within* the cross-task regime — pair_risk, dominant_issue, reconstruction_error — all give ordinary values for this catastrophic anchor.

---

## Cross-Backbone Behavior

| Backbone | Worst Δ | Class | Seed range |
|----------|--------:|-------|----------:|
| DistilBERT | 41.7% | **catastrophic** | 28.9% |
| RoBERTa | 1.7% | mild | 1.2% |
| DeBERTa | — | — | — |

**Backbone shift:** 40.0 percentage points (largest of any pair)
**Reversal?** Yes — full qualitative reversal. Catastrophic → mild.

---

## Mechanistic Observations

### Victim pattern

MRPC is the primary victim on the worst variant (s42 × s7): it collapses from 73.5% to 31.9%, dropping 41.7 points — near chance for a binary classification task. On other seed variants, the victim is less clearly MRPC; QNLI also degrades substantially (13.2–13.6%).

The victim (MRPC) is the *higher*-scoring source (73.5% vs. QNLI's 69.0%). This is counterintuitive: the stronger adapter is the one destroyed. The same pattern appears in CA-02 (SST-2 is the victim and the higher scorer). This may reflect a general principle: the adapter with a "cleaner" decision boundary is more vulnerable to disruption, because its signal is more concentrated and therefore more fragile when mixed with competing representations.

### Culprit pattern

Less specific than CA-02. The worst variant involves qnli_s42, but switching to qnli_s7 does not eliminate degradation — it reduces it from 41.7% to 12.7% (still severe). This contrasts with CA-02, where switching from qnli_s42 to qnli_s7 eliminates the catastrophe entirely. The culprit identification is diffuse for CA-01.

### Architectural hypotheses

DistilBERT has 6 layers versus RoBERTa's 12. The "forced competition" hypothesis: with fewer layers, QNLI and MRPC subspaces are forced to overlap more, creating conditions for destructive interference. On RoBERTa, the additional capacity allows both tasks' learned features to coexist without catastrophic conflict. This hypothesis predicts that DeBERTa (12 layers) will behave more like RoBERTa — mild, not catastrophic — for this pair. But the disentangled attention mechanism adds a confound.

---

## Open Questions

1. **What changes between the s42×s7 and s7×s7 variants?** The task pair is the same; only the learned LoRA subspaces differ. Is the 29% swing driven by a specific layer's alignment pattern, or is it distributed across all layers? This is Workstream B's primary question for this anchor.

2. **Why is MRPC the victim on the worst variant, when it has the higher source score?** Is it because MRPC's paraphrase detection signal is more "concentrated" in a few layers that get disrupted, while QNLI's broader signal survives partial interference?

3. **Does the 6-layer compression force a conflict that 12 layers can accommodate?** If so, CA-01 on DistilBERT may be a *capacity-limited catastrophe* — a failure mode specific to shallow architectures. DeBERTa will test this: if QNLI×MRPC is mild on DeBERTa (12 layers), the depth hypothesis is supported.

4. **Is there a detectable structural difference between the catastrophic seed variant and the severe variants?** All four variants are severe or worse — this pair never merges cleanly on DistilBERT. But the 3× gap between worst and best suggests something qualitative happens at the s42×s7 configuration.

---

## DeBERTa Prediction

**Expected behavior:** Mild or broad degradation — similar to RoBERTa. The depth hypothesis predicts that the 12-layer architecture will accommodate both tasks without catastrophic conflict. Seed range should still be elevated (>10%) because this pair's instability may be intrinsic, but the worst-case delta should be below the catastrophic threshold.

**Confidence:** Medium. The depth hypothesis is the simplest explanation, but DeBERTa's disentangled attention is a genuine architectural novelty that could interact with MRPC's paraphrase-detection features in unexpected ways.

**What would be surprising:** If QNLI×MRPC is catastrophic on DeBERTa, that would challenge the depth hypothesis and suggest the interference mechanism is task-intrinsic rather than capacity-limited. It would also be the strongest evidence yet for the instability program — the pair would be unstable across all three backbones.
