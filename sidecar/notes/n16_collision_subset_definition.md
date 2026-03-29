# Note: Collision Subset Definition

## Metadata

- **Type:** analysis
- **Date:** 2026-03-26
- **Related notes:** n15 (per-layer findings), n14 (per-layer protocol), n06 (program statement)
- **Project:** Within-Layer Collision Program, Stage A

---

## Purpose

This note formalizes the "collision pattern" result from n15 into an explicit subset definition for subsequent within-layer geometry work. It classifies all analyzed pairs on both backbones into collision categories and identifies the specific contrasts that will inform Stage B.

---

## 1. What "Collision" Means

The per-layer analysis (n15) found that catastrophic anchor CA-01 (QNLI×MRPC on DistilBERT) shows notably high per-layer alignment — both adapters concentrate their LoRA norm mass in the same layers. This note extends the analysis to all 12 cross-task pairs across both backbones (plus 8 same-task controls) and classifies each case by whether it sits in a collision regime and what its behavioral outcome is.

A pair is in the **collision regime** if its mean per-layer alignment (Spearman ρ between norm mass vectors) is ≥ 0.70. This threshold is descriptive and data-driven — it sits above the cross-task median and separates pairs whose adapters clearly load the same layers from those that distribute differently. It is not a theoretically motivated hard boundary.

---

## 2. Classification Scheme

| Class | Definition | Count |
|-------|-----------|------:|
| **safe_collision** | Same-task control. High alignment, always safe. | 8 |
| **catastrophic_collision** | High alignment AND catastrophic anchor on this backbone. | 1 |
| **unstable_collision** | High alignment for a backbone-reversal pair, but not catastrophic on this backbone. | 1 |
| **non_catastrophic_collision** | High alignment for a stable-asymmetric pair. Collision without catastrophe. | 3 |
| **moderate_alignment_catastrophic** | Moderate alignment (0.55–0.70) but catastrophic. Edge case. | 1 |
| **non_collision_cross_task** | Lower alignment. Adapters distribute differently across layers. | 6 |

---

## 3. Key Cases

### 3.1 The core collision case: CA-01

QNLI×MRPC on DistilBERT (alignment ρ = 0.86, instability = 0.87, worst Δ = 41.7%).

This is the cleanest collision case. Both QNLI and MRPC adapters produce rising norm mass profiles across DistilBERT's 6 layers, creating strong same-layer loading. The collision is so tight that all four seed variants produce significant degradation (diffuse culprit mode — see n08). On the shallow 6-layer backbone, the adapters have no room to avoid each other.

### 3.2 The collision-without-catastrophe contrast: QNLI×MRPC on RoBERTa

Same pair, different backbone: alignment ρ = 0.80, but worst Δ = 1.7% (mild).

This is the single most important contrast case for Stage B. The collision precondition is present — these adapters still load the same layers on RoBERTa — but no catastrophe occurs. If within-layer geometry distinguishes this case from CA-01, that is direct evidence for within-layer incompatibility as the sufficient condition.

### 3.3 The non-collision catastrophe: CA-02

QNLI×SST-2 on RoBERTa (alignment ρ = 0.66, instability = 0.74, worst Δ = 27.2%).

CA-02 falls *below* the collision threshold. Its alignment (0.66) is above the Group C average from n15 (0.62) but meaningfully below CA-01's (0.86). This is an important edge case. Either:

- The collision threshold is softer than 0.70 and CA-02 is weakly collisional, or
- CA-02's catastrophe operates through a partly different mechanism (within-layer incompatibility strong enough to cause catastrophe even without extreme same-layer loading), or
- CA-02 represents a different mode of catastrophic interference where moderate alignment plus particularly incompatible subspace geometry suffices.

The within-layer analysis (Stage B) should treat CA-02 as a case to characterize, not to force into the collision model.

### 3.4 The same pair, no collision: QNLI×SST-2 on DistilBERT

Alignment ρ = 0.17, worst Δ = 11.0% (severe but not catastrophic).

The same pair that is catastrophic on RoBERTa (CA-02) shows *very low* alignment on DistilBERT and is not catastrophic. This is consistent with the collision model: different backbone geometry produces different layer-level allocation, and when the adapters don't collide, the outcome is merely severe rather than catastrophic.

### 3.5 High-alignment stable contrasts

Three cross-task pairs show alignment ≥ 0.70 without catastrophic behavior:

- **MRPC×SST-2 on RoBERTa** (ρ = 0.89, worst Δ = 15.0%). The highest alignment of any cross-task pair — higher than CA-01 — but a stable-asymmetric pair (instability = 0.21). Its worst outcome (15%) is at the conventional catastrophic boundary but is consistent across seeds (range = 8%). This is a crucial case: maximal collision, non-catastrophic stable behavior.

- **QNLI×RTE on RoBERTa** (ρ = 0.85, worst Δ = 8.3%). High alignment, mild outcome.

- **RTE×SST-2 on DistilBERT** (ρ = 0.71, worst Δ = 8.3%). Borderline alignment, mild outcome.

These three cases demonstrate that collision is not sufficient for catastrophe. Something *additional* must be present in CA-01 that is absent in these safe-but-colliding pairs.

---

## 4. The Collision Landscape

Sorting all 20 entries by alignment reveals the full picture:

```
ρ ≈ 0.99  MRPC×MRPC (RoBERTa), SST-2×SST-2 (RoBERTa)     → safe_collision
ρ ≈ 0.95  QNLI×QNLI (RoBERTa)                              → safe_collision
ρ ≈ 0.94  MRPC×MRPC (DistilBERT), SST-2×SST-2 (DistilBERT) → safe_collision
ρ ≈ 0.90  RTE×RTE (both backbones)                          → safe_collision
ρ = 0.89  MRPC×SST-2 (RoBERTa)                              → non_catastrophic_collision
ρ = 0.86  QNLI×MRPC (DistilBERT)                            → catastrophic_collision ★
ρ = 0.85  QNLI×RTE (RoBERTa)                                → non_catastrophic_collision
ρ = 0.80  QNLI×MRPC (RoBERTa)                               → unstable_collision
ρ = 0.71  RTE×SST-2 (DistilBERT), QNLI×QNLI (DistilBERT)   → non_catastrophic / safe
─── collision threshold (ρ = 0.70) ──────────────────────────────────────────
ρ = 0.66  QNLI×SST-2 (RoBERTa)                              → moderate_alignment_catastrophic ★
ρ = 0.63  RTE×MRPC (RoBERTa)                                 → non_collision_cross_task
ρ = 0.57  RTE×MRPC (DistilBERT)                              → non_collision_cross_task
ρ = 0.56  RTE×SST-2 (RoBERTa)                                → non_collision_cross_task
ρ = 0.53  QNLI×RTE (DistilBERT)                              → non_collision_cross_task
ρ = 0.21  MRPC×SST-2 (DistilBERT)                            → non_collision_cross_task
ρ = 0.17  QNLI×SST-2 (DistilBERT)                            → non_collision_cross_task
```

The two catastrophic anchors (★) sit in different parts of this landscape: CA-01 is in the collision regime, CA-02 is on the boundary. This means the within-layer analysis will need to explain *both* cases, not just the collision-model-friendly one.

---

## 5. What This Tells Us About Collision and Catastrophe

### 5.1 Collision is neither necessary nor sufficient

- **Not sufficient:** MRPC×SST-2 on RoBERTa has the highest cross-task alignment (ρ = 0.89) but is stable. Multiple high-alignment pairs are non-catastrophic.
- **Not strictly necessary:** CA-02 (ρ = 0.66) is catastrophic despite moderate alignment.

### 5.2 But collision matters

- The only strongly colliding catastrophic case (CA-01, ρ = 0.86) has the highest instability in the dataset (0.87) and the most extreme severity (41.7%).
- Backbone-reversal pairs (instability > 0.7) tend toward higher alignment than stable-asymmetric pairs on their catastrophic backbone.
- The overall trend identified in n15 — catastrophic anchors show higher alignment than stable cross-task pairs — holds when additional pairs are included.

### 5.3 The refined picture

Collision is a **risk amplifier**, not a deterministic trigger. It creates the precondition for destructive interference by ensuring that merged weight perturbations land in the same layers. Whether that precondition becomes catastrophic depends on within-layer geometry — the subspace angles, directional overlap, and effective-rank interaction at the collision site. This is exactly what Stage B should test.

---

## 6. Stage B Contrast Panel (Recommended)

The collision subset supports a three-group contrast for within-layer analysis:

### Group 1 — Catastrophic collision

- **CA-01:** QNLI×MRPC on DistilBERT (ρ = 0.86, Δ = 41.7%)
- **CA-02:** QNLI×SST-2 on RoBERTa (ρ = 0.66, Δ = 27.2%) — edge case, included despite moderate alignment

### Group 2 — Safe collision controls

- **QNLI×MRPC on RoBERTa** (ρ = 0.80, Δ = 1.7%) — same pair as CA-01, different backbone. The highest-information contrast.
- **MRPC×SST-2 on RoBERTa** (ρ = 0.89, Δ = 15.0%) — highest cross-task alignment, non-catastrophic.
- **Same-task controls** as needed (ρ ≈ 0.90–0.99, Δ ≈ 0–2%)

### Group 3 — Non-collision cross-task

- **QNLI×SST-2 on DistilBERT** (ρ = 0.17, Δ = 11.0%) — same pair as CA-02, no collision, not catastrophic.
- **RTE×MRPC on RoBERTa** (ρ = 0.63, Δ = 8.3%) — low alignment, low severity.

### The key contrasts

The highest-information comparison is within the QNLI×MRPC pair across backbones:

```
DistilBERT: ρ = 0.86, catastrophic (41.7%)   → collision + catastrophe
RoBERTa:    ρ = 0.80, mild (1.7%)            → collision + safe
```

If within-layer metrics differentiate these two cases — same task pair, both colliding, but only one catastrophic — that is strong evidence that within-layer geometry is the operative threshold variable.

---

## 7. Decision

Stage A succeeds: the collision subset is meaningful, includes both catastrophic and safe cases, and creates a clear contrast for Stage B. The subset also reveals an important refinement: collision is a risk amplifier rather than a deterministic trigger, and the CA-02 edge case ensures that Stage B does not over-commit to a pure collision model.

Proceed to Stage B.

---

## 8. Structured Outputs

| File | Location |
|------|----------|
| Collision subset JSON | `sidecar/results/collision_subset/collision_subset_table.json` |
| Collision subset table | `sidecar/results/collision_subset/collision_subset_table.md` |
