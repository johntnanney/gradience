# Visual Packet — Curated Figures

**Last updated:** 2026-03-31

Eight figures that carry the full story. Grouped by theme, ordered for narrative flow. Use these for talks, posts, collaborator onboarding, or re-entry after time away.

All figures live in `sidecar/figures/`. SVG is primary; PNG where available.

---

## Theme 1: The Founding Observation

### Figure 1 — Severity reverses; instability doesn't
**File:** `s01_summary_panel.png` / `.svg`

**What it shows:** Per-pair merge severity on DistilBERT vs RoBERTa. The same task pairs swap positions across backbones — QNLI x MRPC is catastrophic on DistilBERT (41.7% degradation) but mild on RoBERTa (1.7%), and vice versa for QNLI x SST-2. But the pairs that are *variable* (high instability) are the same on both backbones.

**Why it matters:** This is the founding observation. It kills the idea of a portable severity score and establishes instability (variability, not magnitude) as the stable descriptor. Every subsequent finding builds on this reframing.

---

## Theme 2: The Mechanism

### Figure 2 — V-module dimensionality ratio: the strongest signal
**File:** `per_module_v_spotlight.png` / `.svg`

**What it shows:** V-module dimensionality ratio for catastrophic vs safe-collision pairs. Cohen's d = 3.36 with zero range overlap. The two groups are perfectly separable on this single metric.

**Why it matters:** This is the sharpest finding in the entire evidence base. After testing Q, K, V, and O modules separately, only V carries the catastrophe-discriminating signal. The aggregate analysis (all modules concatenated) had averaged this away — per-module decomposition recovered it.

### Figure 3 — Only V discriminates
**File:** `per_module_discrimination.png` / `.svg`

**What it shows:** Cohen's d per attention module: V = 3.36, K = 1.39, Q ~ 0, O ~ 0. A bar chart showing that the discriminative power is concentrated in V, not distributed across modules.

**Why it matters:** Companion to Figure 2. Explains why the aggregate within-layer analysis failed (hypothesis 3 in the ruled-out list) and why module-specific decomposition was the methodological breakthrough.

### Figure 4 — Readout orthogonality is benign
**File:** `output_space_readout_alignment.png` / `.svg`

**What it shows:** Readout alignment (decision-axis cosine) grouped by merge outcome. Incompatible readout appears in both catastrophic and safe groups. The SC-QMRB falsifier — same readout geometry as catastrophic CA-01, but safe (1.7% degradation) — is highlighted.

**Why it matters:** This is the key falsifier. It proves that readout incompatibility alone is harmless, establishing the conjunctive model: catastrophe requires V-module pathology AND readout incompatibility. Either alone is benign. 40% of same-task pairs are orthogonal and all merge safely.

---

## Theme 3: Behavioral Grounding

### Figure 5 — Behavioral signatures across the merge panel
**File:** `example_semantics_preservation_breakage.png`

**What it shows:** Preservation rate and breakage rate across the 8-case behavioral panel, grouped by merge quality class (safe, near-miss, fragile, control). Safe and near-miss cases cluster together with high preservation. Fragile and control cases show distinct breakage patterns.

**Why it matters:** The first figure that connects structural compatibility to observable model behavior on real examples. It shows that near-miss is behaviorally indistinguishable from safe — the boundary between them is an evidence gap, not a behavioral threshold.

### Figure 6 — Neither-source rate as threshold discriminator
**File:** `example_semantics_taxonomy_composition.png`

**What it shows:** Five-category failure taxonomy composition by case. Category D (neither-source behavior — the model produces output matching neither parent adapter) jumps from <2% in safe/near-miss to >12% in fragile/control. This is the cleanest single behavioral discriminator.

**Why it matters:** Establishes that catastrophic merge failure produces qualitatively novel behavior, not just degraded versions of the source outputs. The <2% / >12% threshold is the behavioral boundary between "safe" and "pathological" tiers.

---

## Theme 4: Route 2 Broadening

### Figure 7 — Aggregation is computational, not presentational
**File:** `decision_dependent_aggregation_matrix.svg`

**What it shows:** A 12-case panel evaluated under four aggregation families (worst-case, distributional, QA-dominant, hybrid). Each cell shows the operational label produced by that family for that case. Only 2/12 cases are aggregation-invariant (both cross-task with clear QA). The remaining 10 change label depending on the aggregation rule.

**Why it matters:** This kills the idea that aggregation is a presentation choice. Different rules produce genuinely different operational judgments from the same structural evidence. Merge needs worst-case. Routing needs distributional. Triage needs QA-dominant. General-purpose needs the hybrid. The figure makes the divergence visually immediate.

### Figure 8 — Route 2 profiles have behavioral reality
**File:** `behavioral_route2_profile_matrix.svg`

**What it shows:** Three behavioral metrics (neither-source %, confidence collapse count, high-confidence wrong count) plotted for each of the five Route 2 compatibility profiles. The profiles separate into three tiers: no pathology (safe, optional), localized pathology (collapse, cross-task), and stasis (QA review).

**Why it matters:** Grounds the Route 2 framework in observable behavior. The most important finding it reveals: worst-case collapse and cross-task contamination produce the same ~14% neither-source rate but through opposite channels — collapse (model knows it doesn't know: 28-30 confidence collapses, 0 high-confidence wrong) vs contamination (model doesn't know it doesn't know: 3 confidence collapses, 23 high-confidence wrong).

---

## Using this packet

### For a 5-minute overview
Show Figures 1, 2, and 5. The founding observation, the strongest signal, and the behavioral grounding. Three figures, one story: severity is unstable, V-module explains why, and the structural findings track real model behavior.

### For a 15-minute presentation
Show all 8 in order. The narrative arc: founding observation (1) → mechanism (2–4) → behavioral confirmation (5–6) → broadened framework (7–8).

### For a written document
Figures 2 and 4 are the two you'd include in a methods section (the signal and its falsifier). Figures 5 and 6 go in results (behavioral grounding). Figures 7 and 8 go in discussion (generalization and decision-context dependence).

### For collaborator onboarding
Pair this packet with the [demo paths](demo-paths.md). Path A readers need Figures 5–6. Path B readers need Figures 7–8. Path C readers need all 8.

---

## What's not in this packet

40 figures total exist in `sidecar/figures/`. This packet selects 8. The full registry with tier markings (T1/T2/T3) is in [`sidecar/figures/README.md`](../sidecar/figures/README.md). Key omissions:

- **Head-level V analysis** (T2) — shows how module-level signal decomposes into per-head cancellation. Important for mechanism but not for the top-level story.
- **Seed-readout coupling** (T2) — shows that V-module pathology and readout incompatibility are independently determined. Strengthens the conjunctive model but not needed for first exposure.
- **Attractor mapping** (T2/T3) — shows the multi-attractor landscape across task families. Interesting but specialized.
- **Confidence distributions** (T2) — the double dissociation between fragile (confidence collapse) and control (high-confidence wrong). Compelling but covered by the description of Figure 8.
