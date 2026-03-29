# Note: Catastrophic Anchor Dossier Synthesis

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related notes:** n08 (CA-01), n09 (CA-02), n02 (original dossiers)
- **Project:** Phase 3, Project H — Catastrophic Anchor Dossiers
- **Supersedes:** n02 (catastrophic anchor dossiers) — which contained the original compact dossiers; now expanded into individual dossier documents (n08, n09) plus this synthesis

---

## Purpose

This note synthesizes the two catastrophic anchor dossiers (CA-01: QNLI×MRPC on DistilBERT; CA-02: QNLI×SST-2 on RoBERTa) and extracts the cross-cutting patterns that inform the instability research program. It replaces n02 as the primary reference for catastrophic anchor interpretation while preserving n02 for historical reference.

---

## Cross-Dossier Patterns

### Pattern 1 — The victim is always the stronger source

In CA-01, MRPC (73.5%) collapses while QNLI (69.0%) degrades moderately. In CA-02, SST-2 (89.4%) collapses while QNLI (~80%) degrades moderately. In both cases, the adapter with the higher single-task accuracy is the one destroyed by the merge.

**Interpretation:** This is not a coincidence but may reflect a structural regularity. A "stronger" adapter likely has more concentrated, task-specific features — a cleaner signal in a lower-dimensional subspace. Per-layer analysis (n15) shows that catastrophic pairs' adapters load the *same* layers (collision pattern). When two adapters collide in the same layers and the stronger adapter's precise, concentrated features are averaged with the other's, the precise signal is destroyed — it is more fragile under linear interpolation than a broader, noisier adaptation. The "weaker" adapter's diffuse features survive partial collision; the "stronger" adapter's precise features do not.

**Testable implication:** If a structural predictor of instability exists, it may involve the *relative concentration* of the two adapters' feature subspaces *within their shared high-norm layers* — something like a ratio of effective dimensionality at the collision site, where pairs with one highly concentrated and one diffuse adapter loading the same layers are more catastrophe-prone.

### Pattern 2 — QNLI is always involved

Both catastrophic anchors involve QNLI. No pair that excludes QNLI crosses the catastrophic threshold on either backbone. This could be coincidental (only 4 tasks in the panel, and QNLI appears in 3 of 6 cross-task pairs), but it may also reflect something about QNLI's learned representations:

- QNLI is a question-passage NLI task, requiring multi-sentence reasoning
- The LoRA adaptations for QNLI may learn features that are particularly disruptive to simpler tasks (MRPC, SST-2) when merged
- QNLI adapters may have more variable subspace geometry across seeds, making them more likely to land in catastrophic configurations

**Testable implication:** If QNLI is intrinsically more "dangerous" as a merge partner, DeBERTa should show the same pattern — QNLI-involving pairs will have higher instability than non-QNLI pairs. If QNLI's role disappears on DeBERTa, the pattern is backbone-specific.

### Pattern 3 — Culprit specificity varies

CA-01 has a diffuse culprit: all four seed variants produce severe or catastrophic results, and no single adapter is cleanly implicated. CA-02 has a sharp culprit: qnli_s42 is cleanly responsible, and replacing it with qnli_s7 eliminates the catastrophe entirely.

**Interpretation:** These represent two modes of catastrophic interference:

- **Diffuse mode (CA-01):** The task pair is broadly incompatible on this backbone. All seed combinations produce significant degradation; the catastrophic variant is the worst of a uniformly bad set.
- **Sharp mode (CA-02):** The task pair is not intrinsically catastrophic — most seed combinations are benign. A specific adapter's learned geometry triggers the catastrophe.

The sharp mode is more consistent with the thresholded subspace interference hypothesis: shared-layer loading provides the collision precondition (present in both modes — n15), and per-module analysis (n21) now shows that qnli_s42's geometric toxicity is concentrated in the O-module (Δcos = -0.31 vs. qnli_s7) and V-module (Δcos = -0.15), consistent with V-module dimensionality mismatch as the threshold variable (see glossary). The diffuse mode (CA-01) may reflect a case where shared-layer loading on a shallow backbone (6 layers) is so extreme that *any* within-layer configuration produces significant interference — notably, CA-01's seed sensitivity remains invisible at per-module resolution (n21 §3.1), suggesting the threshold in diffuse mode operates below the Q/K/V/O level.

### Pattern 4 — Seed ranges separate the clusters

The two catastrophic anchors have seed ranges of 28.9% and 26.2%. The four stable-asymmetric pairs have seed ranges of 0–8%. The ratio between the worst unstable and best stable seed range is approximately 4:1. This separation is the empirical foundation of the instability concept.

**Testable implication:** If DeBERTa preserves this separation — high seed ranges for the backbone-reversal pairs, low for the stable pairs — the instability ranking is portable. If DeBERTa collapses the separation, the two-backbone gap was an artifact of the specific architectures tested.

---

## What the Dossiers Reveal About Core Gradience

Core's design is correct: it does not attempt to distinguish CA-01 from a benign cross-task pair, because no available signal makes that distinction reliably. The pair_risk, dominant_issue, and reconstruction_error values for CA-01 are indistinguishable from non-catastrophic pairs. The only signal that fires (task_relationship_advisory) is binary and fires on all cross-task pairs.

**This is not a failure of core's signals. It is a failure of the target variable.** The signals were designed to predict severity, and severity is not a stable property. Core's correct response to this finding is exactly what it does: flag the boundary, avoid severity grading, let the user budget evaluation accordingly. The sidecar's contribution, if instability proves portable, would be to add a second tier: "this is not just cross-task, it is *unstably* cross-task."

---

## What the Dossiers Contribute to the Instability Program

1. **Concrete exemplars.** Abstract claims about instability and thresholded interference become tangible when grounded in specific pairs with specific numbers. CA-01 and CA-02 are the program's reference cases.

2. **Testable predictions.** Each dossier includes DeBERTa predictions that can be checked after the adjudication leg. If the predictions hold, the dossiers demonstrate that the program can make useful forecasts. If they fail, the dossiers document exactly what went wrong.

3. **Mechanistic entry points.** The culprit specificity in CA-02 (qnli_s42 vs. qnli_s7) is the single sharpest question for Workstream B. Per-layer analysis (n15) showed that the collision precondition (shared-layer loading) is present for *all* seed variants of CA-02, so the toxicity of qnli_s42 is not a layer-level phenomenon — it must reside in within-layer subspace geometry. Computing principal angles between qnli_s42 and sst2 adapters at their shared high-norm layers, and comparing with the benign qnli_s7 pairing, would be the most direct test of whether within-layer subspace incompatibility is the operative threshold variable.

4. **Contrastive structure.** The two anchors differ in culprit specificity (diffuse vs. sharp), victim identity (MRPC vs. SST-2), and backbone (shallow vs. deep). These differences make them complementary: any explanation that accounts for one but not the other is incomplete.

---

## Stable-Pair Contrast Cases

The dossier set is not complete without documenting what the *non*-catastrophic pairs look like. Two contrast dossiers are noted here (full dossiers are lower priority but can be written if needed):

### RTE × MRPC (instability = 0.19, rank 5)

This is the most consistently benign cross-task pair. On DistilBERT: worst Δ = 7.1%, seed range = 4.7%. On RoBERTa: worst Δ = 8.3%, seed range = 0.0%. The pair degrades modestly and identically across all seed variants on RoBERTa — zero seed sensitivity. It is the anti-CA-02: a pair where the seed simply does not matter.

### RTE × SST-2 (instability = 0.15, rank 6)

The lowest-instability cross-task pair. Shows the SST-2 escalation pattern (8.3% on DistilBERT → 12.6% on RoBERTa) but with contained seed range (2.2% and 4.3%). Demonstrates that SST-2 escalation alone is not sufficient for instability — the pair also needs high seed sensitivity.

---

## Future Dossiers

If DeBERTa produces new catastrophic anchors (possible), they should receive full dossiers following the template. The most likely candidates:

1. **QNLI × SST-2 on DeBERTa** — if the SST-2 escalation pattern continues
2. **MRPC × SST-2 on DeBERTa** — already borderline catastrophic on RoBERTa (15.0%)
3. **Any surprise pair** — the most scientifically informative outcome

If a previously stable pair becomes catastrophic on DeBERTa, its dossier should specifically document what changed and whether the instability program predicted the shift.
