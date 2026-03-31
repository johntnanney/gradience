# n89 -- Dossier: Optional vs Fragile

**Type:** case dossier
**Date:** 2026-04-01
**Program:** Behavioral Route 2 Bridge
**Stage:** D
**Depends on:** n86 (panel), n87 (protocol), n88 (findings)
**Status:** complete

---

## Why this comparison matters

The same-family optional profile (near-miss) and the worst-case-collapse profile (fragile) are the two Route 2 categories most at risk of conflation. Both involve same-task pairs where one source is weaker than the other. Both produce some information loss. The structural distinction is clear (near-miss has marginal degradation, fragile has substantial degradation), but the behavioral distinction could in principle be blurry — a gradient from "a little bad" to "very bad."

The finding here is that the distinction is not a gradient. It is a clean threshold on three metrics, separating two qualitatively different behavioral regimes.

---

## Cases compared

| | NM-01 (optional) | FR-01 (collapse) |
|---|---|---|
| **Backbone** | DistilBERT | BERT |
| **Task** | irony (binary) | emotion (4-class) |
| **Source A** | 0.632 | 0.752 |
| **Source B** | 0.618 | 0.204 |
| **Merged** | 0.620 | 0.664 |
| **Δ vs best** | -0.012 | -0.088 |

---

## Metric comparison

| Metric | NM-01 | FR-01 | Ratio |
|--------|-------|-------|-------|
| Preservation rate | 0.673 | 0.800 | 0.84x |
| Joint breakage rate | 0.042 | 0.064 | 0.66x |
| Neither-source rate | **0.018** | **0.146** | **0.12x** |
| Confidence collapse | **0** | **30** | **0 vs 30** |
| High-conf wrong | 1 | 0 | — |
| Better-source loss rate | 0.491 | 0.233 | 2.1x |

### What the numbers show

The preservation rate is actually *higher* in FR-01 (0.80 vs 0.67). This is counterintuitive until you notice that FR-01's strong source (0.752) dominates most examples — the merge preserves the strong source's behavior on most of the evaluation set. The damage is concentrated on specific examples.

The discriminating metrics are neither-source rate and confidence collapse:
- **Neither-source:** NM-01 has 9 examples (1.8%) where the merge produces a prediction neither source made. FR-01 has 73 examples (14.6%). This is an 8x difference.
- **Confidence collapse:** NM-01 has zero. FR-01 has 30 examples where merged confidence drops below 0.4 while the stronger source had confidence above 0.6.

These are not different points on a gradient — they are different behavioral regimes. NM-01 has negligible novel failure; FR-01 has substantial novel failure with accompanying uncertainty.

---

## Category distribution

| Category | NM-01 | FR-01 |
|----------|-------|-------|
| A: Preserved stable | 310 (62%) | 321 (64%) |
| Better-source loss | 143 (29%) | 75 (15%) |
| D: Collapse | 7 (1.4%) | 5 (1.0%) |
| Shared failure | 40 (8%) | 88 (18%) |
| C: Confusion | 0 | 11 (2.2%) |
| F: Confident misassignment | 1 | 0 |

NM-01's dominant non-preserved category is better-source loss (29%) — the merge doesn't always keep the slightly better source's answer, but it never introduces new failure modes. FR-01's dominant non-preserved category is shared failure (18%) — both sources and the merge all fail, plus an additional 11 confusion examples and 73 neither-source examples that create novel failure.

---

## Representative examples

### NM-01 — typical optional behavior
The merge produces a different prediction from both sources on 9/500 examples (1.8%). In most of these, the merge's confidence is moderate (0.5-0.6), and the "wrong" prediction is the other binary class — a marginal boundary shift, not a catastrophic failure. The model hesitates at the decision boundary but doesn't produce pathological output.

### FR-01 — typical collapse behavior
The merge produces 73 neither-source predictions. On 30 of these, the merge confidence drops below 0.4 (near chance for 4-class). The model becomes genuinely uncertain — spread across multiple classes rather than committing to one. This is the V-module pathology signature at the example level: the weak source's incompatible dimensionality structure bleeds through on specific examples, creating conflicting gradients that cancel to uncertainty.

---

## What this means for Route 2

**The optional/fragile distinction is behaviorally real.** It is not a gradient — it is a threshold. On the discriminating metrics, NM-01 clusters with the safe cases (SR-01: 0.8% neither-source, 0 confidence collapses), not with the collapse cases (FR-01: 14.6% neither-source, 30 confidence collapses). There is nothing in between.

**QA-dominant aggregation is behaviorally justified for optional cases.** When QA flags NM-01 as "uncertain" or "needs review," it is not blocking a case that would collapse — it is correctly identifying a case that is safe-but-under-evidenced. The behavioral content of the optional case is safe-like; the evidence concern is the right concern.

**Worst-case aggregation is behaviorally justified for fragile cases.** FR-01's pathology is driven by the single weak source. Worst-case aggregation (which reduces the pair to its worst component) correctly identifies this risk. Distributional aggregation might underweight the concentrated pathology because the *average* behavior is decent (80% preservation).

---

## Output artifacts

- `sidecar/notes/n89_behavioral_route2_dossier_optional_vs_fragile.md` (this note)
