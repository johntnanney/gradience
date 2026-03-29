# n64 — Example Dossier: Safe vs Fragile Merges

**Type:** dossier
**Date:** 2026-03-28
**Depends on:** n61 (behavior findings), n63 (taxonomy findings)
**Status:** Complete.

---

## Purpose

This dossier makes the behavioral contrast between safe and fragile merges concrete at example level. Rather than reporting aggregate rates, it walks through representative examples from the flip catalog to show what preservation, breakage, and neither-source behavior actually look like in individual predictions. The goal is interpretive: a reader should come away understanding not just *that* safe and fragile merges differ, but *how* the difference manifests in the model's per-example decision-making.

---

## Case profiles

### SR-01 — Safe retained baseline (DistilBERT / irony, binary)

**Pair:** irony_JB173 × neibla
**Sources:** A = 0.632, B = 0.620
**Merged:** 0.626 (Δ = -0.006)

Two irony adapters of comparable strength, same backbone, same task. The merge barely changes aggregate accuracy. This is the canonical "safe merge" — two models that learned similar discriminative rules, and linear averaging preserves them.

**Taxonomy composition:** A (preserved consensus) dominates at 60.6%. The remaining errors are overwhelmingly X (shared failure, 35.8%) — examples both sources already got wrong. Merge-caused categories are negligible: D (neither-source) = 0.8%, C (better-source loss) = 1.0%.

**Behavioral signature:** Near-perfect consensus preservation (97.5% of source-correct examples survive). Zero confidence collapse. The merge's confidence tracks the sources closely — mean merged confidence (0.697) nearly equals source A confidence (0.699). The merge is not uncertain about anything it decides.

### FR-01 — Fragile case (BERT / emotion, 4-class)

**Pair:** bert_emotion_TGbase × fabriceyhc
**Sources:** A = 0.752 (strong), B = 0.204 (near-chance on 4 classes)
**Merged:** 0.664 (Δ = -0.088)

One strong emotion adapter merged with a deeply weak partner. The merge loses 8.8 percentage points — not catastrophic, but structurally damaging. Source B is so weak (0.204 on a 4-class task, where chance is 0.25) that it has barely learned a discriminative rule at all.

**Taxonomy composition:** E (benign absorption) dominates at 49.4% — the merge correctly handles most source disagreements. But D (neither-source) is elevated at 12.4%, and C (better-source loss) adds 6.0%. Together with X (shared failure, 17.6%), only 14.6% of examples show preserved consensus — the merge has little consensus to preserve because the sources rarely agree.

**Behavioral signature:** 30 confidence collapse events (source A confident, merge uncertain). Neither-source rate of 14.6%. Zero high-confidence wrong predictions. The merge knows it is broken — it fails with low confidence, not misplaced certainty.

### FR-02 — Fragile case, severe variant (BERT / emotion, 4-class)

**Pair:** bert_emo_TGbase × hatexplain_NM
**Sources:** A = 0.752 (strong), B = 0.136 (deeply below chance)
**Merged:** 0.664 (Δ = -0.088)

Same strong source as FR-01, but the weak partner is even weaker (0.136, well below the 0.25 chance baseline). Same aggregate degradation (-0.088), but the example-level profile is more severe.

**Taxonomy composition:** E (benign absorption) dominates at 59.2%. D (neither-source) is 13.4%. A (preserved consensus) collapses to 5.0% — the sources almost never agree because B is too weak. The consensus-breakage rate is 34.2% (13 of 38 jointly-correct examples broken), the highest in the panel.

**Behavioral signature:** 28 confidence collapse events. Neither-source rate of 15.4%. The weaker the partner, the more violent the breakage on the rare examples where consensus exists.

### CT-01 — Cross-task control (BERT / ag_news × hate)

**Pair:** bert_cross_agnews × aviator_hate
**Sources:** A = 0.922 (ag_news, very strong), B = not evaluable (hate adapter, wrong label space)
**Merged:** 0.826 (Δ = -0.096, largest in panel)
**Evaluation task:** ag_news (4-class)

A strong ag_news adapter contaminated by a hate adapter that cannot even be evaluated on the ag_news task. The cross-task merge preserves 87.4% of source-correct examples but breaks 12.6% — and produces neither-source predictions on 14.4% of all examples.

**Taxonomy composition:** A (preserved consensus) dominates at 80.6% because source A alone defines "consensus" here. D (neither-source) is 13.6%. No C or E categories because there is no meaningful second source to disagree with.

**Behavioral signature:** High-confidence wrong predictions (23 cases). Only 3 confidence collapse events. Unlike the fragile merges, the cross-task control fails with *confidence* — it has inherited a wrong rule from the hate adapter and applies it decisively. This is qualitatively different from fragile failure.

---

## Comparative example walk-throughs

### Walk-through 1: Preserved consensus (Category A)

**SR-01, idx 0:** Label = 0, pred_a = 0, pred_b = 0, pred_m = 0. Merged confidence 0.787, source A confidence 0.796. Margin gap 0.574.

Both sources correctly classify this irony example. The merge preserves the prediction at nearly the same confidence level. The margin gap (0.574) is comfortable — the merge is not near a decision boundary. This is the modal example in a safe merge: uneventful, confident, correct.

**FR-01, idx 2:** Label = 3, pred_a = 3, pred_b = 3, pred_m = 3. Merged confidence 0.633, source A confidence 0.795. Margin gap 0.362.

Even in a fragile merge, some consensus is preserved — but notice the cost. Source A was confident (0.795), the merge is less so (0.633). The margin gap has compressed from wherever source A started to 0.362. The merge preserved the answer but degraded the confidence. If the task had a harder decision boundary, this example might have flipped.

**Key contrast:** In safe merges, preservation comes with confidence stability. In fragile merges, even preserved examples show confidence erosion — the merge is preserving the output but damaging the underlying representation.

### Walk-through 2: Neither-source behavior (Category D)

**FR-01, idx 4:** Label = 1, pred_a = 1, pred_b = 2, pred_m = **3**. Merged confidence 0.521, source A confidence 0.592. Margin gap 0.271.

Source A correctly predicts class 1 (sadness). Source B predicts class 2 (anger). The merge predicts class **3** (joy) — a label that neither source selected. The merged confidence (0.521) is barely above uniform (0.25 for 4-class). The margin gap (0.271) puts the merge very close to the decision boundary.

This is the canonical neither-source failure. The merge averaged the internal representations of "sadness" and "anger" and landed in the representation space corresponding to "joy." The result is a novel prediction that reflects neither source's learned discriminative rule, produced at near-chance confidence.

**FR-01, idx 5:** Label = 0, pred_a = 0, pred_b = 2, pred_m = **3**. Merged confidence 0.511, source A confidence 0.596. Margin gap 0.119.

Same pattern: source A says class 0, source B says class 2, merge says class 3. Margin gap 0.119 — the merge is essentially guessing. This is a 4-class problem, and the merge has been pushed so close to uniform confidence that it landed on the third option by a razor-thin margin.

**SR-01 for contrast:** Neither-source behavior occurs in 0.8% of SR-01 examples. When it does occur (idx 57: label = 1, pred_a = 1, pred_b = 1, pred_m = 0, confidence 0.520, margin gap 0.040), it happens at the absolute margin of the decision boundary. The merge was essentially at 50/50 and fell on the wrong side of the coin flip. This is noise, not structural pathology.

**Key contrast:** In fragile merges, neither-source predictions are systematic — they recur at 14–15% rate and reflect genuine representational compromise. In safe merges, the rare neither-source predictions are decision-boundary noise at vanishing margins.

### Walk-through 3: Confidence collapse vs high-confidence wrong

**FR-01, confidence collapse pattern:** Across 30 examples with confidence collapse (source A > 0.6, merge < 0.4), the merge has lost the source's discriminative signal. The mean merged confidence in these cases is approximately 0.35 on a 4-class problem — barely above uniform. The merge is not just wrong; it has lost its ability to distinguish between classes. The softmax distribution has flattened.

**CT-01, high-confidence wrong pattern (idx 26):** Label = 1, pred_a = 1, pred_m = **0**. Merged confidence 0.554, source A confidence **0.992**. Source A was nearly certain (99.2% confident, correct), the merge flipped to a wrong prediction at moderate confidence.

More strikingly, CT-01 shows 23 examples where the merge is confidently wrong (confidence > 0.8 on an incorrect prediction). This means the cross-task adapter injected a coherent but wrong rule: the merge doesn't flatten to uncertainty, it redirects to a different decision with conviction.

**Key contrast:** Fragile merges fail by losing signal (confidence collapse → near-uniform softmax). Cross-task merges fail by substituting signal (high confidence → wrong class). This maps directly to the mechanism ladder distinction between V-module pathology (fragile, representation averaging destroys discriminative directions) and readout contamination (control, a foreign task's readout rule is applied through the gate).

### Walk-through 4: Better-source loss (Category C)

**FR-01, idx 8:** Label = 3, pred_a = 1, pred_b = 3, pred_m = 1. Merged confidence 0.522, source A confidence 0.603.

Source A is wrong (predicts 1 instead of 3), source B happens to be correct (predicts 3). The merge follows source A's wrong prediction. This is a case where the stronger source's error is preserved by the merge — source B's correct answer is discarded.

In a same-task merge, this is the expected cost of averaging: the merge follows the more confident source, and when that source is wrong, the merge inherits the error. The confidence signature (0.522 merged) shows moderate uncertainty — the merge felt the pull of source B but not enough to flip.

**NM-01, idx 1:** Label = 1, pred_a = 0, pred_b = 1, pred_m = 0. Merged confidence 0.642, source A confidence 0.648.

Near-miss better-source loss looks almost identical to the fragile case in structure: source A wrong, source B right, merge follows A. But the confidence signature is different — merged confidence (0.642) nearly matches source A (0.648), and the margin gap (0.285) is comfortable. The merge is not uncertain or conflicted; it simply followed the wrong source on this example. This is averaging cost, not pathology.

**Key contrast:** Better-source loss in fragile merges carries the shadow of confidence erosion. In near-miss merges, it occurs at stable confidence — the merge is wrong, but it is making a clear decision, not a compromised one.

---

## Mechanism ladder implications

The example-level evidence maps onto the mechanism ladder with notable clarity:

**V-module pathology (Rung 1).** The neither-source predictions in FR-01 and FR-02 are the example-level footprint of V-module averaging. When two adapters encode different discriminative directions in the value subspace, linear averaging produces a compromise direction that corresponds to neither source's learned rule. At the example level, this manifests as predictions that no source made, at near-chance confidence. The 14–15% neither-source rate in fragile cases quantifies the fraction of examples where V-module pathology reaches the output.

**Readout gating (Rung 3).** The difference between fragile failure (confidence collapse) and control failure (high-confidence wrong) reflects the readout gate's role. In fragile same-task merges, both sources share the readout mapping, so V-module pathology transmits directly to the output as uncertainty — the readout gate is open, but there is no coherent signal to transmit. In cross-task merges (CT-01), the merged readout has inherited a foreign task's decision rule, and that rule is internally coherent — it classifies confidently, just wrongly. The gate transmits a clear signal, but the signal comes from the wrong source.

**Conjunctive interaction.** The safe merges (SR-01, SR-02) show that when V-module compatibility is high (similar tasks, comparable strength), the readout gate transmits a healthy signal. The near-miss merges (n65) confirm that mild V-module pathology does not breach the readout gate. The fragile merges demonstrate the threshold: when V-module pathology exceeds a critical level, the readout gate cannot compensate, and neither-source behavior emerges at the output.

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This dossier | `sidecar/notes/n64_example_dossier_safe_vs_fragile.md` |
| Flip catalog (source data) | `sidecar/results/example_semantics/example_flip_catalog.json` |
| Taxonomy findings (context) | `sidecar/notes/n63_failure_taxonomy_findings.md` |
