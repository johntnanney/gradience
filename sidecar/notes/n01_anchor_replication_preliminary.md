# Note: Preliminary Anchor Replication Findings (Two Backbones)

## Metadata

- **Type:** implication
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related panels:** P01

---

## Summary

The two-backbone analysis phase of Study S01 is complete. Every finding strengthens the case for backbone-dependent catastrophic interference and validates the sidecar's core research question. No task pair is a stable catastrophic anchor across both existing backbones, and the two catastrophic anchors that exist are each confined to a single backbone.

## Findings

### Finding 1 — The anchor identity reversal is real and large

QNLI × MRPC produces a 41.7% worst-case collapse on DistilBERT and a 1.7% worst-case delta on RoBERTa. QNLI × SST-2 produces an 11.0% worst-case on DistilBERT and a 27.2% collapse on RoBERTa. This is not a marginal instability — it is a qualitative reversal of which pair is catastrophic.

The ratio of RoBERTa-to-DistilBERT worst-case deltas for these two pairs is 0.04 and 2.47, respectively. No summary-level signal in the current Gradience toolkit predicted this reversal.

### Finding 2 — SST-2-involving pairs escalate systematically on the deeper backbone

All four cross-task pairs involving SST-2 show higher worst-case deltas on RoBERTa than on DistilBERT. The two non-SST-2 pairs (QNLI×RTE, RTE×MRPC) are stable between backbones. This pattern is consistent with an account in which SST-2's single-sentence sentiment classification task interacts differently with the deeper model's representational structure — perhaps because the additional layers of RoBERTa allocate more representational capacity to sentiment features, creating a larger "target" for cross-task interference to disrupt.

### Finding 3 — Seed fragility is concentrated in the catastrophic anchors

The two pairs with the largest seed-variant ranges are exactly the two catastrophic anchors: QNLI×MRPC on DistilBERT (28.9% range, CV=0.61) and QNLI×SST-2 on RoBERTa (26.2% range, CV=0.89). This is notable because it suggests that catastrophic interference has a threshold character — it depends on specific properties of the learned subspace (which vary with seed), not just on task identity.

By contrast, pairs that are merely "broad degradation" (5–10% delta) show much smaller ranges (2–5%) and lower CVs. They degrade consistently but mildly. The catastrophic pairs degrade inconsistently but sometimes severely.

This supports Hypothesis 1 from the strategy memo: catastrophic failures may not be the tail of a smooth severity curve. They may reflect a qualitatively different interference mode that requires specific subspace alignment conditions to trigger.

### Finding 4 — No stable catastrophic anchor exists in the current evidence

No task pair crosses the 15% catastrophic threshold on both DistilBERT and RoBERTa. The closest candidate is MRPC×SST-2 (12.8% on DistilBERT, 15.0% on RoBERTa — crossing the threshold only on RoBERTa).

This means the DeBERTa-v3 replication leg of S01 cannot be designed around "confirming a known catastrophic pair." It must instead ask: does a third pattern emerge, or does DeBERTa converge with one of the existing patterns?

### Finding 5 — Same-task controls remain clean

Same-task pairs show negligible deltas on both backbones (max 2.2% on DistilBERT, max 1.0% on RoBERTa). The cross-task boundary remains robust. This is not a finding about the sidecar's research question — it is a standing confirmation that core Gradience's boundary detection is well-calibrated.

## Implications for Core Gradience

**These findings validate the current core design.** Core Gradience stops at boundary detection (same-task vs. cross-task) and does not attempt severity grading within the cross-task regime. The two-backbone analysis shows that severity grading based on task-pair identity alone would have a catastrophic false-confidence problem: the "worst" pair on one backbone is benign on another. A user who trusted a task-pair severity ranking derived from DistilBERT experiments would be misled on RoBERTa.

Nothing in these findings is promotable to core. The entire finding is about instability of the severity signal, which is precisely the kind of result that should remain in the sidecar until a deeper mechanism is understood.

## Implications for the Sidecar

### What the DeBERTa leg should test

The DeBERTa-v3 replication is now the highest-priority next step. Three specific outcomes would be informative:

1. **DeBERTa matches the RoBERTa pattern** (SST-2 pairs escalate, QNLI×MRPC is benign). This would support a depth-dependent account: shallow models (DistilBERT) and deeper models (RoBERTa, DeBERTa) produce different interference profiles.

2. **DeBERTa matches the DistilBERT pattern** (QNLI×MRPC is catastrophic). This would be surprising and would point toward the disentangled attention mechanism or some other architectural variable as a differentiator.

3. **DeBERTa shows a third pattern.** This would be the most informative outcome for the sidecar, because it would rule out a simple depth explanation and force investigation of architecture-specific mechanisms.

### What Workstream B should focus on

The seed fragility finding (Finding 3) gives Workstream B (layerwise conflict contrast) a sharp question: for the catastrophic seed variants of QNLI×MRPC on DistilBERT and QNLI×SST-2 on RoBERTa, is the conflict localized to specific layers? And does the non-catastrophic seed variant of the same pair show a different layerwise pattern?

This would directly test Hypothesis 2 (localized conflict signatures) while taking advantage of the natural experiment that seed variation provides.

## Decision or Recommendation

No promotion decision yet. The two-backbone analysis is an intermediate deliverable. The DeBERTa leg must complete before S01 can reach a conclusion.

**Recommended next actions:**
1. Execute DeBERTa-v3 training and evaluation (requires GPU compute)
2. Begin Workstream B design using the catastrophic-vs-mild seed variant contrast
3. Update P01 with DeBERTa results when available
