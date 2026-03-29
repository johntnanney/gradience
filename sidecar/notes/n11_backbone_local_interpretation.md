# Note: Backbone-Local Interpretation — DistilBERT vs. RoBERTa

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related studies:** S01
- **Project:** Phase 3, Project I — Backbone-Local Interpretation Notes

---

## Purpose

This note separates what appears regular *within* each backbone from what holds *across* backbones. The instability program's central claim is about cross-backbone portability, but even if instability is portable, individual backbones have local regularities that are scientifically informative and practically useful. This note catalogs them.

---

## 1. DistilBERT-Local Regularities

### 1.1 QNLI×MRPC is uniquely catastrophic

On DistilBERT, QNLI×MRPC is the only catastrophic pair (worst Δ = 41.7%). No other pair exceeds 12.8%. The gap between CA-01 (41.7%) and the next-worst pair (MRPC×SST-2, 12.8%) is 28.9 percentage points — larger than any other pair's *total* worst delta.

**Local interpretation:** On a 6-layer backbone, QNLI and MRPC compete catastrophically for limited representational capacity. This competition is seed-modulated (range = 28.9%) but always present — even the best seed variant shows 12.7% degradation (severe).

**What this does not tell us:** Whether QNLI×MRPC would be catastrophic on any other 6-layer architecture. The finding is specific to DistilBERT.

### 1.2 Same-task control is tighter on DistilBERT

DistilBERT same-task pairs: mean Δ = 0.8%, max Δ = 2.2%.
RoBERTa same-task pairs: mean Δ = 0.1%, max Δ = 1.0%.

RoBERTa's same-task control is tighter (max 1.0% vs. 2.2%). This may reflect RoBERTa's greater capacity: with 12 layers, two same-task adapters have more room to coexist without interference. On DistilBERT, even same-task merges produce small but measurable friction.

### 1.3 Cross-task severity distribution is bimodal on DistilBERT

On DistilBERT, the six cross-task pairs split into:
- 1 catastrophic (41.7%)
- 3 severe (11.0–13.6%)
- 2 broad (6.4–8.3%)

There is a gap between the catastrophic pair and the severe cluster (41.7% → 13.6%). Within the severe cluster, the three pairs are tightly bunched (11.0%, 12.8%, 13.6%). This bimodal structure supports the thresholded subspace interference hypothesis locally on DistilBERT.

### 1.4 Dominant issue uniformity

On DistilBERT, all cross-task pairs with available Gradience audit data show `dominant_issue = partial_redundancy` and `pair_risk = medium`. The core signals do not differentiate within the cross-task regime on this backbone. This is a local confirmation that the signals were designed for boundary detection, not within-regime grading.

---

## 2. RoBERTa-Local Regularities

### 2.1 SST-2 escalation

All four SST-2-involving cross-task pairs show higher worst-case deltas on RoBERTa than on DistilBERT:

| Pair | DistilBERT worst Δ | RoBERTa worst Δ | Change |
|------|--------------------:|----------------:|-------:|
| QNLI × SST-2 | 11.0% | 27.2% | +16.2 |
| MRPC × SST-2 | 12.8% | 15.0% | +2.2 |
| RTE × SST-2 | 8.3% | 12.6% | +4.3 |
| QNLI × MRPC | 41.7% | 1.7% | −40.0 |
| RTE × MRPC | 7.1% | 8.3% | +1.2 |
| QNLI × RTE | 6.4% | 8.3% | +1.9 |

The three SST-2 pairs escalate (+2.2 to +16.2 points). The two non-SST-2 pairs are approximately stable (+1.2, +1.9). QNLI×MRPC collapses (−40.0) — but this is the backbone reversal, not a local regularity.

**Local interpretation:** SST-2's single-sentence sentiment classification features may occupy a larger or more vulnerable subspace on the deeper RoBERTa backbone. The escalation is systematic across all SST-2 pairs, regardless of the other task, suggesting it is a property of SST-2's interaction with RoBERTa's architecture rather than a property of specific task pairings.

**Cross-backbone status:** This is a local regularity, not a portable one. It predicts behavior on other 12-layer backbones (DeBERTa) but this is extrapolation, not confirmed.

### 2.2 qnli_s42 as toxic adapter

On RoBERTa, the catastrophic behavior of QNLI×SST-2 is entirely driven by qnli_s42 (see CA-02 dossier). Variants with qnli_s7 are benign (1.0–2.8%). This is the sharpest single-adapter culprit identification in the evidence base.

**Local interpretation:** Something about qnli_s42's learned subspace on RoBERTa creates destructive interference with SST-2. The same adapter (qnli_s42) on DistilBERT participates in the CA-01 catastrophe but less specifically — on DistilBERT, the culprit is diffuse.

This suggests the "toxic adapter" phenomenon may be backbone-dependent: qnli_s42 is specifically toxic on RoBERTa but only generically bad on DistilBERT.

### 2.3 Same-task control is near-perfect

RoBERTa same-task pairs: 3 of 4 show exactly 0.0% delta. MRPC shows 0.98%. The same-task merge is almost lossless on RoBERTa. This sets a very clean baseline: any cross-task degradation on RoBERTa is unambiguously attributable to cross-task interference, not to merge noise.

### 2.4 RTE × MRPC has zero seed sensitivity on RoBERTa

All four seed variants of RTE×MRPC on RoBERTa produce exactly the same worst-case delta (8.3%). The seed range is 0.0%. This is unique in the dataset — no other pair on either backbone has exactly zero seed sensitivity. RTE×MRPC on RoBERTa is the most deterministic cross-task pair in the evidence base.

---

## 3. Cross-Backbone Contrasts

### What transfers across backbones

1. **Boundary detection.** Same-task pairs are safe on both backbones. Cross-task pairs degrade on both. The binary boundary is backbone-portable. (Core claim, confirmed.)

2. **Instability ranking.** The two most unstable pairs are the same on both backbones (QNLI×MRPC, QNLI×SST-2). The four most stable pairs are the same on both backbones. (Sidecar claim, awaiting DeBERTa confirmation.)

3. **Severity is non-trivial for all cross-task pairs.** Every cross-task pair shows >5% worst-case delta on at least one backbone. The "stable mild" taxonomy class remains empty in the cross-task regime.

### What does NOT transfer

1. **Severity rankings.** QNLI×MRPC: 41.7% → 1.7% across backbones. Rankings are completely reversed.

2. **Catastrophic anchor identity.** Different pairs are catastrophic on each backbone.

3. **Culprit specificity.** qnli_s42 is a sharp culprit on RoBERTa but a diffuse contributor on DistilBERT.

4. **SST-2 escalation.** Specific to the DistilBERT → RoBERTa comparison. May or may not generalize to DeBERTa.

5. **Same-task control magnitude.** RoBERTa is near-perfect (max 1.0%); DistilBERT is still safe but measurably noisier (max 2.2%).

---

## 4. What This Means for DeBERTa

The backbone-local analysis generates three specific DeBERTa expectations:

**Expectation 1 (from SST-2 escalation):** If SST-2 escalation is a depth phenomenon (6 layers → 12 layers), DeBERTa (12 layers) should show similar escalation. If it is a standard-attention phenomenon, DeBERTa's disentangled attention may block it.

**Expectation 2 (from same-task control):** DeBERTa same-task control should be tighter than DistilBERT and comparable to RoBERTa, given similar model capacity.

**Expectation 3 (from QNLI×MRPC collapse):** QNLI×MRPC should be mild on DeBERTa (as on RoBERTa), because the catastrophe appears to be a shallow-backbone phenomenon. But this is the lowest-confidence expectation — if QNLI×MRPC is catastrophic on DeBERTa, the depth hypothesis fails.

---

## 5. What This Note Does Not Claim

This note describes backbone-local regularities. It does not claim these regularities are portable. The SST-2 escalation pattern is a local finding until DeBERTa confirms or denies it. The qnli_s42 toxic-adapter phenomenon is a local finding until a structural analysis explains what makes it toxic.

The instability program's strength is that its central claim (instability ranking portability) is testable on DeBERTa. The backbone-local regularities documented here are supplementary: useful for interpretation and prediction, but not the program's decisive bet.
