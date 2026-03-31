# n87 -- Behavioral Route 2 Comparison Protocol

**Type:** protocol note
**Date:** 2026-03-31
**Program:** Behavioral Route 2 Bridge
**Stage:** B
**Depends on:** n86 (panel), n60-n63 (original example semantics protocol and taxonomy)
**Status:** complete

---

## Question

What metrics and categories should be used to test whether broadened Route 2 compatibility profiles have distinct behavioral signatures?

---

## Design rationale

The n59-n66 example semantics program produced a validated 5-metric, 5-category behavioral framework for merge-failure analysis. This protocol adapts that framework to the broadened Route 2 context rather than reinventing it.

The adaptation requires two changes:

1. **Reframe metrics for decision contexts beyond merge.** The original metrics assume a merge scenario (source A + source B → merged). For routing and triage contexts, the "merged" model is still the comparison target, but the interpretation differs: a routing-confusable case should show confusion patterns, not collapse patterns; a QA-dominant review case should show under-evidenced stasis, not structural failure.

2. **Add Route 2-specific behavioral categories.** The original 5-category taxonomy (A: preserved consensus, C: better-source loss, D: neither-source behavior, E: benign absorption, X: shared failure) captures merge failure modes. Route 2 adds categories for optional-but-safe behavior, confusion behavior, and QA-override uncertainty.

---

## Metrics

### Metric 1 — Preservation rate

**Definition:** Among examples where at least one source is correct, what fraction does the target (merged/compared) model get correct?

**Computation:** `preserved_count / either_correct_count`

**Route 2 interpretation:** High preservation (>0.95) = aggregation-invariant safe behavior. Moderate preservation (0.6-0.95) = optional or marginal. Low preservation (<0.6) = structural failure or cross-task contamination.

### Metric 2 — Joint breakage rate

**Definition:** Among examples where both sources are correct (the joint-correct set), what fraction does the target model get wrong?

**Computation:** `joint_breakage_count / both_correct_count`

**Route 2 interpretation:** Near-zero joint breakage distinguishes aggregation-invariant from worst-case-collapse profiles. High joint breakage in a same-task case signals worst-case-collapse behavior. For cross-task cases, the both-correct set is typically small or absent, making this metric less informative.

### Metric 3 — Better-source loss rate

**Definition:** Among examples where one source is correct and the other is wrong, how often does the target model fail to preserve the correct source's answer?

**Computation:** `better_source_loss / (better_source_preserved + better_source_loss)`

**Route 2 interpretation:** High better-source loss in optional cases would undermine the "safe-but-under-evidenced" interpretation. Low better-source loss supports H5 (QA-dominant cases resemble safe rather than fragile).

### Metric 4 — Confidence collapse rate

**Definition:** Count of examples where merged confidence drops below 0.4 while the stronger source had confidence above 0.6.

**Route 2 interpretation:** Confidence collapse is the signature of fragile/worst-case-collapse profiles. Its absence in optional and QA-dominant profiles supports the profile distinction. Its presence in cross-task cases would suggest contamination rather than collapse.

### Metric 5 — Neither-source behavior rate

**Definition:** Fraction of examples where the target model produces a prediction that neither source made.

**Computation:** `neither_source_count / num_examples`

**Route 2 interpretation:** This is the cleanest discriminator from n63. In Route 2 terms: aggregation-invariant safe < 2%, worst-case-collapse ~14%, cross-task separable ~14%. The question is whether same-family optional cases fall below the threshold (supporting H5) and whether QA-dominant review cases show a distinct pattern (under-evidenced stasis rather than novel behavior).

### Metric 6 — High-confidence wrong rate

**Definition:** Count of examples where the target model predicts with confidence >0.7 and is wrong.

**Route 2 interpretation:** High-confidence wrong is the signature of cross-task contamination (CT-01 in the original analysis). If routing-confusable cases also show high-confidence wrong behavior, that would distinguish them from merge-fragile (which shows confidence collapse instead). If they don't, the behavioral distinction between routing-confusability and merge-fragility may not be as clean as the structural distinction.

---

## Behavioral categories

### Category A — Preserved stable behavior

**Definition:** Examples correctly handled by the target model that were also correct in at least one source. The merge preserved or improved on the source behavior.

**Includes:** preserved_consensus + better_source_preserved + merge_recovery (from original taxonomy)

**Route 2 significance:** The base rate for this category should be highest in aggregation-invariant safe profiles and lowest in worst-case-collapse.

### Category B — Optional-but-safe behavior

**Definition:** Examples where the target model's behavior is not strictly preserved but the error is benign — disagreement absorption, marginal accuracy changes, or different-but-harmless prediction shifts.

**Includes:** benign disagreement absorption (E from original taxonomy)

**Route 2 significance:** This category should dominate in same-family optional profiles. Its prevalence in optional cases would support H5 (QA-dominant review cases look like under-evidenced safety). It should be present but not dominant in aggregation-invariant safe profiles.

### Category C — Confusion behavior

**Definition:** Examples where the target model appears to mix or confuse source behaviors — low-margin predictions, unstable category assignments, or predictions that partially reflect both sources without clearly following either.

**Includes:** better-source loss cases where confidence is near-chance, neither-source cases where confidence is low

**Route 2 significance:** This is the hypothesized signature of routing-confusable cases (H4). If present, it would distinguish routing-confusability from merge-collapse (which shows concentrated breakage rather than diffuse confusion).

### Category D — Collapse behavior

**Definition:** Examples where the target model fails on cases that both sources handled correctly, with accompanying confidence collapse.

**Includes:** consensus_breakage + joint_breakage cases with confidence collapse

**Route 2 significance:** This is the hypothesized signature of worst-case-collapse profiles (H3). It should be concentrated (not diffuse) — a few examples failing sharply rather than many examples degrading mildly.

### Category E — QA-override uncertainty

**Definition:** Examples where both sources are near chance and the merge reflects this — shared failure, low confidence, no meaningful behavioral content to preserve or break.

**Includes:** shared_failure cases where both source confidences are below 0.5

**Route 2 significance:** This is the hypothesized signature of QA-dominant review profiles (H5). The behavioral pattern is stasis rather than failure — nothing to preserve, nothing to break, the merge is a formality.

### Category F — Confident misassignment / contamination

**Definition:** Examples where the target model makes a high-confidence wrong prediction, especially when the prediction reflects a different task's decision boundary.

**Includes:** high-confidence wrong cases, source_a_loss cases with high merged confidence

**Route 2 significance:** This is the hypothesized signature of cross-task separable cases. The merge doesn't produce uncertainty — it produces confident wrong answers because the cross-task adapter's learned features interfere with the primary task's decision structure.

---

## Application protocol

For each case in the panel:

1. Load the 500-example prediction file
2. Compute all 6 metrics
3. Classify each example into the dominant behavioral category (A-F)
4. Record the category distribution
5. Compare the case's behavioral signature to its Route 2 profile assignment
6. Note where the profile is behaviorally supported, ambiguous, or contradicted

Then compare across profiles:
- aggregation-invariant (SR-01, SR-02) vs worst-case-collapse (FR-01, FR-02)
- same-family optional (NM-01, NM-02) vs cross-task separable (CT-01)
- routing-confusable (NM-01) vs merge-fragile (FR-01)
- QA-dominant review (AN-01) vs worst-case-collapse (FR-01)

---

## Success criteria

- The metric set is small (6 metrics) and interpretable
- The categories are broad enough for Route 2 but not sprawling (6 categories)
- The protocol can be applied consistently across all 8 panel cases
- The categories adapt the validated n63 taxonomy rather than replacing it

---

## Output artifacts

- `sidecar/notes/n87_behavioral_route2_protocol.md` (this note)
- `sidecar/results/behavioral_route2_bridge/protocol_schema.json`
