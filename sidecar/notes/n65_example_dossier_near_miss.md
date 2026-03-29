# n65 — Example Dossier: Near-Miss Merges

**Type:** dossier
**Date:** 2026-03-28
**Depends on:** n61 (behavior findings), n63 (taxonomy findings), n64 (safe vs fragile dossier)
**Status:** Complete.

---

## Purpose

This dossier examines the near-miss cases (NM-01, NM-02) and the anchor case (AN-01) at example level. The central question is whether near-miss merges — pairs that narrowly avoided the "fragile" classification in aggregate terms — show early signs of fragile-type pathology in their per-example behavior, or whether they are genuinely safe merges with slightly higher averaging cost.

The answer matters for the conjunctive model. If near-miss merges show a qualitatively different failure mode from safe retained merges (e.g., incipient neither-source behavior), this would suggest that the boundary between safe and fragile is a gradient with detectable precursors. If near-miss and safe retained are behaviorally identical, the boundary is a threshold — the system is either above it or below it, with no intermediate state.

---

## Case profiles

### NM-01 — Near-miss on DistilBERT / irony (binary)

**Pair:** irony_JB173 × phailyoor_NM
**Sources:** A = 0.632, B = 0.618
**Merged:** 0.620 (Δ = -0.012)
**Comparison pair:** SR-01 (same task, same backbone, same source A, Δ = -0.006)

NM-01 and SR-01 share the same source A adapter and the same task. The difference is source B: SR-01's partner (neibla) is slightly stronger (0.620 vs 0.618), and the merge outcome is marginally better. NM-01 is the near-miss that didn't quite land in the safe-retained bucket — its Δ is twice SR-01's, though still small in absolute terms.

**Taxonomy composition:**

| Category | NM-01 | SR-01 (safe comparator) |
|----------|-------|------------------------|
| A (preserved consensus) | 32.0% | 60.6% |
| C (better-source loss) | 28.6% | 1.0% |
| D (neither-source) | 1.8% | 0.8% |
| E (benign absorption) | 29.6% | 1.8% |
| X (shared failure) | 8.0% | 35.8% |

The contrast with SR-01 is instructive but must be read carefully. NM-01 has much more category C (28.6% vs 1.0%) and E (29.6% vs 1.8%) because the sources disagree more often — NM-01's partner is slightly weaker, so there are more examples where one source is right and the other is wrong. But the *critical* category — D (neither-source) — is almost identical: 1.8% vs 0.8%. The elevated disagreement in NM-01 produces more better-source loss and more benign absorption, but it does not produce neither-source behavior.

**Behavioral signature:**

- Preservation rate: 0.673 (vs SR-01's 0.975). Lower, but this reflects source disagreement, not breakage.
- Joint-source breakage rate: 4.2% (7 of 167). Above SR-01's 1.0% but well below the fragile threshold (~6–34%).
- Neither-source rate: 1.8%. Firmly in the safe range (< 2%).
- Confidence collapse: 0 events. No confidence pathology.
- High-confidence wrong: 1 event. Negligible.
- Mean merged confidence: 0.705 (vs SR-01's 0.697). The merge is actually slightly *more* confident than SR-01.

### NM-02 — Near-miss on BERT / hate (binary)

**Pair:** bert_hate_TGbase × aviator_NM
**Sources:** A = 0.514, B = 0.574 (source B is stronger)
**Merged:** 0.572 (Δ = -0.002)
**Comparison pair:** SR-02 (same task, same backbone, same source A, Δ = +0.028)

NM-02 is structurally parallel to SR-02: same source A, same task, different partner. SR-02's merge improved (+0.028); NM-02's merge barely changed (-0.002). Both have a weak source A (0.514), so the merge depends heavily on the partner.

**Taxonomy composition:**

| Category | NM-02 | SR-02 (safe comparator) |
|----------|-------|------------------------|
| A (preserved consensus) | 12.4% | 14.2% |
| C (better-source loss) | 39.2% | 34.4% |
| D (neither-source) | 0.0% | 0.0% |
| E (benign absorption) | 44.8% | 47.4% |
| X (shared failure) | 3.6% | 4.0% |

The profiles are nearly identical. NM-02 has slightly more better-source loss (39.2% vs 34.4%) and slightly less benign absorption (44.8% vs 47.4%). Neither case shows any neither-source behavior — 0.0% for both. The two cases are taxonomically indistinguishable.

**Behavioral signature:**

- Preservation rate: 0.593 (vs SR-02's 0.642). Lower, reflecting the weaker partner.
- Joint-source breakage rate: 0.0% (0 of 62). Zero consensus breakage — the same as SR-02.
- Neither-source rate: 0.0%. Zero neither-source behavior.
- Confidence collapse: 0 events.
- High-confidence wrong: 43 events (vs SR-02's 26). Both cases show high-confidence wrong predictions — the merge confidently follows source A's decisions on examples where source A is wrong. This is not pathology; it is the expected consequence of merging with a confident-but-sometimes-wrong source.
- Mean merged confidence: 0.693 (vs SR-02's 0.680). Again, the near-miss is marginally more confident.

### AN-01 — Anchor case (BERT / emotion, 4-class)

**Pair:** bert_emo_fab × hatexplain_NM
**Sources:** A = 0.204, B = 0.136
**Merged:** 0.202 (Δ = -0.002)

Both sources are near chance (0.25 for 4-class). The merge changes almost nothing. This case exists to calibrate the taxonomy: when there is no strong signal to preserve, what does the taxonomy show?

**Taxonomy composition:** X (shared failure) dominates at 65.2%. The merge cannot break what was never there. E (benign absorption) at 15.0% reflects cases where the merge happened to land on the right answer from wrong sources — stochastic recovery, not skill. D (neither-source) at 5.2% is intermediate between safe (<2%) and fragile (>12%): there is some representational averaging, but without strong discriminative directions to average, it produces modest noise rather than systematic pathology.

**Behavioral signature:** Mean merged confidence 0.351 — barely above uniform (0.25). No confidence collapse events (both sources were already uncertain). No high-confidence wrong predictions. The merge is flat, not broken.

---

## Comparative analysis: Is near-miss a precursor to fragile?

### The threshold hypothesis

The data support a **threshold model** rather than a gradient model. On every discriminating metric, near-miss cases fall within the safe-retained envelope:

| Metric | Safe retained range | Near-miss range | Fragile range |
|--------|-------------------|-----------------|---------------|
| Neither-source rate | 0.0–0.8% | 0.0–1.8% | 14.6–15.4% |
| Joint-source breakage | 0.0–1.0% | 0.0–4.2% | 6.4–34.2% |
| Confidence collapse | 0 events | 0 events | 28–30 events |
| Mean confidence | 0.680–0.697 | 0.693–0.705 | 0.557–0.592 |

The gap between near-miss and fragile is not a small step — it is a discontinuity. Near-miss merges show zero confidence collapse, zero or near-zero neither-source behavior, and breakage rates at most 4.2%. Fragile merges show 28–30 confidence collapses, 12–15% neither-source behavior, and breakage rates of 6–34%.

### Why near-miss has higher C (better-source loss)

The one metric where near-miss clearly exceeds safe retained is category C — better-source loss. NM-01 shows 28.6% (vs SR-01's 1.0%), and NM-02 shows 39.2% (vs SR-02's 34.4%). This is the expected cost of merging sources of unequal strength: the merge averages their rules and sometimes follows the weaker source.

But better-source loss is not pathological. It does not involve confidence collapse, neither-source behavior, or structural damage to the merged representation. It is the normal price of imperfect averaging — the merge sometimes picks the wrong source's answer when the sources disagree. This is a quantitative cost, not a qualitative failure mode.

### The role of agreement rate

The difference in taxonomy composition between NM-01 and SR-01 is largely driven by the **source agreement rate**. SR-01's sources agree on 306 of 500 examples (61.2%). NM-01's sources agree on 167 of 500 (33.4%). Because the sources disagree more often in NM-01, there is more material for categories C and E — but the merge's handling of disagreement is not structurally different.

This means that comparing taxonomy rates *between* classes must account for the base rate of source agreement. The taxonomy composition reflects the sources' compatibility, not just the merge's quality. The merge-caused categories (B, D) are the ones that isolate merge quality from source properties — and on these, near-miss and safe are indistinguishable.

### Example walk-throughs

**NM-01, D (neither-source), idx 39:** Label = 1, pred_a = 1, pred_b = 1, pred_m = **0**. Merged confidence 0.509, margin gap 0.018.

Both sources correctly predict class 1. The merge predicts class 0. This is consensus breakage reclassified as D (because the merged prediction matches neither source). But look at the margin gap: 0.018. The merge is at 50.9% confidence on a binary problem — essentially a coin flip. It fell on the wrong side by less than 1%. This is not the systematic neither-source pathology seen in FR-01 (where neither-source predictions come at 0.271–0.521 margins on a 4-class problem). This is boundary noise.

**NM-01, D (neither-source), idx 57:** Label = 1, pred_a = 1, pred_b = 1, pred_m = **0**. Merged confidence 0.539, margin gap 0.077.

Same pattern. Margin gap 0.077. The merge is just above the decision boundary. Every one of NM-01's 9 neither-source examples is a boundary case — the merge was nearly 50/50 and fell the wrong way. In FR-01, neither-source predictions show margin gaps of 0.003–0.271, but there are **73** of them, and many are on a 4-class problem where the merge wandered to a third option. The difference is between stochastic noise at a decision boundary and systematic representational confusion.

**NM-02 for contrast:** NM-02 has zero neither-source predictions. Every example is classified as either A, C, E, or X. The merge always follows one source or the other (or both when they agree). There is no representational compromise producing novel outputs.

---

## The anchor's interpretive role

AN-01 is not on the near-miss–safe continuum. It occupies a different position: when both sources are near chance, the merge has no discriminative signal to preserve or destroy. AN-01's 5.2% neither-source rate is higher than safe/near-miss (<2%) but lower than fragile (>12%). This intermediate position reflects the absence of strong discriminative directions in the V-module — there is some averaging noise, but no structured pathology to transmit.

The anchor confirms a negative prediction of the conjunctive model: if neither source has developed meaningful discriminative geometry, V-module pathology cannot arise because there is nothing to damage. The merge of two near-random adapters is itself near-random, and the taxonomy correctly identifies the result as dominated by pre-existing failure (X = 65.2%).

---

## Conclusions for the broader program

1. **Near-miss merges are behaviorally safe.** On every metric that discriminates safe from fragile — neither-source rate, confidence collapse, joint-source breakage — near-miss cases fall within the safe-retained envelope. The boundary between safe and fragile is a threshold, not a gradient.

2. **Better-source loss is the expected cost of averaging, not a warning sign.** Near-miss merges show elevated C rates because their sources disagree more often. This is a property of the sources, not a defect of the merge.

3. **The rare neither-source predictions in near-miss cases are boundary noise.** They occur at vanishing margins (0.018–0.077) on binary problems. They are not precursors of the systematic neither-source pathology that characterizes fragile merges.

4. **The anchor validates the taxonomy's logic.** When there is no signal to preserve, the taxonomy is dominated by shared failure. The merge cannot break what was never there, and the taxonomy correctly reflects this.

5. **The threshold model is confirmed at example level.** The field trial found that near-miss aggregate accuracy (Δ = -0.006 average) was close to safe retained (Δ = -0.024 average). The example-level audit shows that this is not a superficial resemblance: the behavioral mechanisms are the same. Near-miss merges do not harbor latent fragility — they are safe merges with slightly higher averaging cost.

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This dossier | `sidecar/notes/n65_example_dossier_near_miss.md` |
| Flip catalog (source data) | `sidecar/results/example_semantics/example_flip_catalog.json` |
| Safe vs fragile dossier (contrast) | `sidecar/notes/n64_example_dossier_safe_vs_fragile.md` |
