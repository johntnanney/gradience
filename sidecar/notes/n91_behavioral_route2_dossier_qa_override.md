# n91 -- Dossier: QA-Override Stasis vs Structural Collapse

**Type:** case dossier
**Date:** 2026-04-01
**Program:** Behavioral Route 2 Bridge
**Stage:** D
**Depends on:** n86 (panel), n87 (protocol), n88 (findings)
**Status:** complete

---

## Why this comparison matters

The aggregation-sensitive compatibility program (n81-n85) found that QA-dominant aggregation can override the strongest structural signal — a pair with 0.892 compatibility was blocked because both sources lacked behavioral evidence. The question raised but not answered there: is QA-dominant aggregation *correct* to do this, or is it over-conservative?

This dossier answers that question behaviorally. If a QA-blocked case looks like stasis (nothing to preserve, nothing to break), then QA is correctly identifying evidence absence. If it looks like structural failure, QA might be miscategorizing the problem.

---

## Cases compared

| | AN-01 (QA review) | FR-01 (collapse) |
|---|---|---|
| **Backbone** | BERT | BERT |
| **Task** | emotion (4-class) | emotion (4-class) |
| **Source A** | 0.204 | 0.752 |
| **Source B** | 0.136 | 0.204 |
| **Merged** | 0.202 | 0.664 |
| **Δ vs best** | -0.002 | -0.088 |

Same backbone, same task, same number of classes — the only difference is source quality. AN-01 has both sources near chance; FR-01 has one strong source and one weak source.

---

## Metric comparison

| Metric | AN-01 (QA review) | FR-01 (collapse) |
|--------|-----|------|
| Preservation rate | 0.541 | 0.800 |
| Joint breakage rate | **0.000** | **0.064** |
| Neither-source rate | **0.070** | **0.146** |
| Confidence collapse | **0** | **30** |
| High-conf wrong | **0** | **0** |
| Mean merged confidence | **0.351** | **0.557** |

### What the numbers show

The two cases share some surface similarity — both involve weak sources on the same task. But the behavioral profiles are categorically different:

**AN-01 is not failing.** Zero joint breakage, zero confidence collapse, zero high-confidence wrong. The low preservation rate (54%) reflects the small number of correct examples to preserve (either_correct = 159/500), not pathological merge behavior. The merged model gets roughly the same examples right as the sources because all three are near chance.

**FR-01 is failing specifically.** 30 confidence collapses, 5 joint breakage examples, 73 neither-source predictions. The strong source provides a behavioral foundation that the merge partially destroys on specific examples.

The critical comparison is mean confidence: AN-01 merged confidence is 0.351 (near chance for 4-class = 0.25), FR-01 merged confidence is 0.557. AN-01's low confidence is not pathological — it is the natural confidence of a near-chance model. FR-01's moderate confidence conceals localized collapses.

---

## Category distribution

| Category | AN-01 | FR-01 |
|----------|-------|-------|
| A: Preserved stable | 92 (18%) | 321 (64%) |
| E: QA-override uncertainty | **326 (65%)** | 0 |
| Better-source loss | 73 (15%) | 75 (15%) |
| D: Collapse | 0 | 5 (1%) |
| C: Confusion | 9 (2%) | 11 (2%) |
| Shared failure | — | 88 (18%) |

The signature difference: **65% of AN-01's examples fall into Category E (QA-override uncertainty)** — shared failures where both sources are near chance and the merge reflects this stasis. FR-01 has zero Category E examples; instead, 64% are Category A (preserved stable) because the strong source provides genuine behavioral content to preserve.

---

## The behavioral meaning of QA-dominant aggregation

AN-01 is what "evidence-absent" looks like behaviorally: **stasis.** There is no signal to aggregate, so there is no aggregation-sensitive behavior. The structural measurement exists (you can compute compatibility scores between two near-chance adapters) but it means nothing operationally because the adapters themselves carry no meaningful behavioral content.

QA-dominant aggregation correctly identifies this state. When it blocks a case like this, it is not being over-conservative — it is recognizing that the structural measurement has no behavioral substrate. You can merge two near-chance adapters without catastrophe, but the result is another near-chance model. The "safety" of the merge is vacuous.

FR-01, by contrast, has genuine behavioral content (one strong source at 0.752) and genuine pathology when that content is corrupted by the weak source. Here, structural measurement matters: the compatibility score, the worst-case layer analysis, the distributional profile — all carry operational meaning because there is something to preserve or destroy.

---

## What this means for Route 2

**QA-dominant aggregation targets a real behavioral distinction.** The AN-01/FR-01 contrast shows that "blocked by QA" and "structurally fragile" are not two labels for the same thing — they correspond to qualitatively different behavioral states (stasis vs localized pathology).

**The evidence dimension is genuinely independent.** The aggregation-sensitive program (n83) showed that a pair with 0.892 structural compatibility can be QA-blocked. This dossier shows why that's correct: high structural compatibility in a near-chance regime is structurally meaningful but behaviorally empty. The independence of the evidence dimension and the structural dimension, documented architecturally in the aggregation programs, is confirmed behaviorally here.

**The three-tier behavioral model has operational content.** Tier 1 (no pathology, safe/optional) = proceed or QA-gate based on evidence. Tier 2 (localized pathology, collapse/contamination) = aggregation strategy matters, worst-case or distributional depending on decision context. Tier 3 (stasis, QA review) = evidence constraint is the only relevant signal. These tiers correspond to different operational actions, and the correspondence is grounded in behavior, not just architecture.

---

## Output artifacts

- `sidecar/notes/n91_behavioral_route2_dossier_qa_override.md` (this note)
