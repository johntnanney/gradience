# Note: Evidence Atlas — What Gradience Knows (March 2026)

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-27
- **Related studies:** S01 and all prior studies
- **Related panels:** P01
- **See also:** n49 (mechanism-to-commensurability synthesis — commensurability v3), n45 (conjunctive model with mechanisms), n46 (mechanism classification audit), n48 (mechanism determinant findings), n41 (attractor mapping findings), n42 (readout topology), n44 (decision-axis analysis)

---

## Summary

This atlas maps the current state of empirical knowledge across the Gradience project: what has been tested, what is resolved, what remains open, and what has been ruled out.

### Three Conceptual Levels (descriptive)

The evidence base supports a clear distinction between three levels of merge-risk description, listed in order of decreasing reliability:

1. **Boundary** (core concept, stable). The same-task / cross-task boundary is reliably detected with 0 false positives across 2 backbones. This is a binary classification: same-task pairs are safe; cross-task pairs require caution. Boundary detection is the foundation of core Gradience and is not in question.

2. **Severity** (backbone-local, not portable). The magnitude of performance degradation after a cross-task merge. Severity rankings reverse across backbones — the most severe pair on DistilBERT is among the mildest on RoBERTa. Severity is a valid measurement within a single (backbone × seed) condition but is not a property of the task pair itself. No signal tested to date predicts severity portably.

3. **Instability** (first promising portable descriptor). The variability of a pair's severity across seeds and backbones. Instability rankings are consistent across both tested backbones: the same two pairs are most unstable regardless of which backbone's perspective is adopted. The instability gap (0.74 → 0.30) is the cleanest separation in the evidence base. Pending DeBERTa-v3 confirmation before any promotion claim.

These levels are not competing. They are nested: boundary is settled, severity is useful but local, instability may be the first cross-condition descriptor. See n05 for the full argument.

### Multiscale Mechanism Ladder (explanatory)

Orthogonal to the three descriptive levels, the sidecar now has a three-rung mechanistic ladder explaining *why* catastrophic failures occur, organized by the spatial scale at which each condition operates (see glossary, n25):

1. **Module-level risk** (n21). V-module dimensionality ratio discriminates catastrophic from safe collision pairs (d=3.36, zero overlap). This is a group-level signal: it identifies *which pairs* are at risk. Averaging across attention heads strengthens the signal.

2. **Head-level modulation** (n24). The same module-level dim ratio can produce catastrophe or not, depending on how the mismatch distributes across attention heads. Different seeds reconfigure head-level geometry in opposite directions, cancelling at module level but producing different functional outcomes. This explains *seed sensitivity* — specifically, CA-01's 29-point severity gap that was invisible at all prior resolution levels.

3. **Readout gating** (n32, n36, n37, n44, n45; resolved — bifurcated by backbone). The classifier readout layer functions as a **gate**, not an amplifier. Rung 3 now bifurcates into two distinct mechanisms:

   - **Rung 3a — Rotational degeneracy** (DistilBERT multi-attractor families). Seeds find orthogonal orientations within a *shared* low-rank PC subspace. Energy overlap is high (> 0.93); the seeds use the same features but combine them differently. Gate is open via rotational freedom.
   - **Rung 3b — Feature-set switching** (QNLI on RoBERTa). Seeds lock onto *different* principal components of the pretrained representation. Energy overlap is low (0.26). One QNLI/RoBERTa seed uses a decision direction aligned with RTE (cos = 0.86) — cross-task feature exploitation. Gate is open via structural divergence.

   The conjunctive model remains: **V-module pathology × readout incompatibility → catastrophic failure**. Decision-axis cosine is bimodal (~0 or ~1) and modulated by task identity, backbone architecture, and training convergence. Readout orthogonality is common and normally harmless (5/14 same-task seed pairs, all safe). The two conditions are independently determined (n36). See n45 for the complete updated model.

The ladder is nested: Rung 1 sets the precondition, Rung 2 modulates the outcome, Rung 3 gates whether upstream pathology manifests as catastrophic classification errors. The Rung 3 mechanism is backbone-dependent. Each rung is independently testable on DeBERTa.

4. **Mechanism determinants** (n46–n49; new). The Rung 3 bifurcation is governed by a structured hierarchy: task identity (primary, determines attractor count) → backbone architecture (secondary, selects mechanism) → training convergence (tertiary, gates attractor count) → domain structure (weak). All 5 multi-attractor conditions classified: 4 rotational degeneracy (all DistilBERT), 1 feature-set switching (QNLI/RoBERTa). Critical backbone confound: mechanism and backbone are perfectly correlated in the current panel. Commensurability refined to version 3 (n49): readout incompatibility decomposes by mechanism class, with different computational checks and different failure semantics.

---

## 1. Resolved — High Confidence

### Same-task regime: CLOSED / SAFE

**Evidence:** 4 same-task pairs on DistilBERT (max Δ = 2.2%), 4 on RoBERTa (max Δ = 1.0%).

**Verdict:** Same-task pairs are safe to merge across seeds. No actionable blind spot found.

**Tested dimensions:** seed variation, training-style variation (rank, alpha, dropout, schedule), source-strength variation (strong/medium/weak), domain shift within task family (SST-2/Yelp/Amazon sentiment).

**Regime summary:**

| Study | N pairs | Max Δ | Verdict |
|-------|--------:|------:|---------|
| Same-task cross-study (DistilBERT) | 4 | 2.2% | safe |
| Same-task cross-study (RoBERTa) | 4 | 1.0% | safe |
| Domain shift (sentiment family) | 15 | 2.2% | safe |
| Source strength (QNLI, varying quality) | 15 | 2.4% | safe |
| Training style (QNLI, varying config) | 15 | 3.4% | safe |

**Total same-task-regime pairs tested:** 53. Maximum degradation: 3.4%.

### Cross-task boundary: SOLVED

**Evidence:** Task-relationship advisory fires on all cross-task pairs. 0 false positives in the evidence base. Advisory generalizes across DistilBERT and RoBERTa.

**Verdict:** The same-task / cross-task boundary is reliable. Core Gradience correctly identifies it.

### Core utility: CONFIRMED

**Evidence:** Utility round demonstrates that the current stable stack (QA + pair reports + task advisory + inventory summary) can reduce a mixed-task candidate space from 28 possible pairs to 4 retained candidates in a representative 8-adapter inventory.

**Verdict:** Gradience is useful today as a mixed-task inventory preflight system.

---

## 2. Open — Active Investigation

### Cross-task severity grading: REFRAMED (sidecar)

**Status:** The original question — "what determines cross-task severity?" — has been reframed. Severity is outcome-local (backbone × seed dependent) and is not a well-defined target variable across conditions. The sidecar's working question is now: *what determines whether a cross-task pair is stable or unstable?* (See n05.)

**Failed severity signals (all backbone-local):**

| Signal | Result | Backbone-portable? |
|--------|--------|--------------------|
| Task-pair identity | Severity varies >10× for the same pair across backbones | **No** |
| Core-space shared-basis | Did not predict catastrophic pairs | **No** |
| Pair-risk label | Same label (medium) on catastrophic and benign pairs | **No** |
| Format similarity | "Same format" pairs (NLI family) include both severe and mild cases | **No** |
| Source-strength gap | No correlation with cross-task severity | **No** |
| Reconstruction error | Indistinguishable between catastrophic and benign pairs | **No** |

**Why these failed:** All six signals attempted to predict severity magnitude — a backbone-local quantity. The failure is not in the signals but in the target variable. See n05 §3.

### Catastrophic interference: OPEN (sidecar priority)

**Known catastrophic anchors:**

| Anchor | Backbone | Worst Δ | Key feature |
|--------|----------|--------:|-------------|
| QNLI × MRPC | DistilBERT | 41.7% | Seed-fragile (range 28.9%). Mild on RoBERTa. |
| QNLI × SST-2 | RoBERTa | 27.2% | Seed-fragile (range 26.2%). Driven by qnli_s42. |

**Key finding:** No pair is catastrophic on both backbones. Catastrophic interference is a (task pair × backbone × seed) interaction, not a task-pair property. The unit of analysis is the triple, not the pair.

### Instability as working concept: PROMISING (sidecar — now central)

**Finding:** Instability — the variability of a pair's severity across seeds and backbones — cleanly separates backbone-reversal pairs (instability > 0.7) from stable-asymmetric pairs (< 0.3). The gap between the two clusters is 0.44 units; no pair currently occupies it.

**What instability captures that severity does not:** Instability rankings are consistent across both backbones, even though severity rankings are inverted. The two backbone-reversal pairs are the most unstable on *both* backbones.

**Working hypothesis:** Catastrophic interference has a threshold character — it requires shared-layer loading (both adapters concentrating norm mass in the same layers) combined with within-layer subspace incompatibility. Per-layer analysis (n15) confirmed the collision precondition: catastrophic pairs show higher layer-level alignment (ρ=0.76) than stable cross-task pairs (ρ=0.62). Per-module analysis (n21) identified the V-module (value projection) dimensionality ratio as the strongest correlate: catastrophic collision pairs show more asymmetric V-module effective ranks (mean 0.69) than safe collision pairs (mean 0.84), with zero range overlap and Cohen's d = 3.36. The mechanism is localized to specific attention components, not distributed across the aggregate subspace.

**Status:** Two-backbone result. Elevated to sidecar's working concept (n05), formalized as research program (n06). Per-layer structural analysis complete (n13–n15, MIXED). Within-layer analysis complete (n17–n18, MIXED/NEGATIVE). Per-module analysis complete (n19–n21, POSITIVE). Needs DeBERTa-v3 confirmation before any promotion claim. Compact case table at `sidecar/results/s01/instability_case_table.md`.

---

## 3. Ruled Out — Negative Results

### Task-pair identity as severity predictor

**Ruled out by:** QNLI×MRPC reversal (catastrophic → mild across backbones).

### Core-space shared-basis as general severity signal

**Ruled out by:** High shared-basis scores appear on both catastrophic and benign pairs.

### A smooth severity continuum

**Challenged by:** Catastrophic pairs are not the tail of a smooth distribution. They are seed-fragile with threshold character (range > 25%, vs < 8% for all other pairs).

### Per-layer divergence as catastrophe mechanism

**Ruled out by:** Per-layer analysis (n15). Catastrophic pairs show *lower* layer-level divergence and *higher* alignment than stable cross-task pairs. The mechanism is collision (shared-layer loading), not competition (divergent layer profiles).

### Aggregate within-layer subspace geometry as catastrophe threshold

**Ruled out by:** Within-layer geometry pilot (n18). When backbone is controlled, catastrophic cases (CA-02 on RoBERTa) are indistinguishable from safe collision controls in principal angle spectrum, top-direction overlap, dimensionality ratio, and directional conflict. CA-01's distinctive profile (high top-direction overlap) is plausibly a DistilBERT architecture effect (6-layer compression), not a catastrophe marker. Seed sensitivity within CA-01 is invisible in all four metrics.

### Collision as sufficient condition for catastrophe

**Ruled out by:** Collision subset analysis (n16). MRPC×SST-2 on RoBERTa has the highest cross-task alignment (ρ=0.89) but is stable (instability=0.21). Multiple high-alignment pairs are non-catastrophic. Collision is a risk amplifier, not a deterministic trigger.

### CA-01 (DistilBERT) seed sensitivity at per-module resolution

**Not explained by:** Per-module geometry analysis (n21). CA-01's 29-point seed range (s42×s7 at 41.7% vs. s7×s7 at 12.7%) remains invisible at per-module granularity — all Q/K/V/O deltas are < 0.07. The seed-dependent variable operates below per-module resolution (likely attention-head or weight-direction level).

### Readout geometry as sole catastrophe predictor

**Ruled out by:** Output-space readout audit (n32). SC-QMRB (QNLI × MRPC on RoBERTa, safe, Δ=1.7%) has virtually identical readout geometry to CA-01 (DistilBERT, catastrophic, Δ=41.7%): both have decision_axis_cos ≈ 0, margin_proxy ≈ 0.70, angle ≈ 89°. Multiple cross-task pairs with orthogonal readout are mild or safe. Readout incompatibility is a necessary background condition, not a discriminative marker.

### Readout geometry as seed-sensitivity explanation

**Ruled out by:** Output-space readout audit (n32). CA-01-catastrophic and CA-01-mild have virtually identical readout geometry (decision_axis_cos: 0.015 vs −0.059, margin_proxy: 0.712 vs 0.686) despite a 29pp performance gap. Readout explains zero variance in seed-sensitive modulation.

### Readout orthogonality as risk marker

**Ruled out by:** Seed-contingent readout analysis (n36). 5 of 14 same-task seed pairs show orthogonal decision axes (cos < 0.1), yet all merge safely (Δ ≤ 2.2%). Orthogonal readout is a routine feature of LoRA fine-tuning for multi-attractor tasks (QNLI consistently orthogonal, MRPC on DistilBERT), not a risk marker. It is harmless unless combined with upstream V-module pathology.

### Readout-upstream coupling as mechanism

**Ruled out by:** Seed-contingent readout analysis (n36). All same-task seed pairs have healthy V-module geometry (dim ratio > 0.78) regardless of readout classification (orthogonal or aligned). Readout axis selection is decoupled from upstream representation structure. The two conditions for catastrophe are independently determined.

---

## 4. Taxonomy of Cross-Task Pairs

| Category | Pairs | Characteristics |
|----------|-------|-----------------|
| **Backbone reversal** | QNLI×MRPC, QNLI×SST-2 | Catastrophic on one backbone, mild/severe on other. Highest instability. Highest seed range. |
| **Stable asymmetric** | MRPC×SST-2, RTE×SST-2, RTE×MRPC, QNLI×RTE | Degrades 5–15% consistently. Low instability. Similar profile across backbones. |
| **Stable mild** | (none in cross-task) | All cross-task pairs show > 5% degradation somewhere. |

---

## 5. Evidence Inventory

### Data assets

| Asset | Location | Pairs | Backbone |
|-------|----------|------:|----------|
| Cross-task adjudication | `results/cross_task_subtype_study_01/` | 28 | DistilBERT |
| Cross-task adjudication | `results/task_pair_severity_generalization_study_01/` | 28 | RoBERTa |
| Domain shift | `results/domain_shift_blind_spot/` | 15 | DistilBERT |
| Source strength | `results/source_strength_blind_spot/` | 15 | DistilBERT |
| Training style | `results/training_style_blind_spot/` | 15 | DistilBERT |
| Task advisory round | `results/task_advisory_round/` | 46 | DistilBERT |
| Broader benchmarks | `results/study14_broader_benchmarks/` | 107 adapters | Multiple |
| Merge ablation | `results/study17_smoke/` | varies | Llama-2-7b |

### Saved adapter weights

Adapter weights (.safetensors) exist for all blind-spot studies and cross-task studies on DistilBERT and RoBERTa. These can be used for CPU-only layerwise analysis without retraining.

### Sidecar outputs (this session)

| Output | Path |
|--------|------|
| Backbone comparison | `sidecar/results/s01/three_backbone_comparison.json` |
| Seed stability | `sidecar/results/s01/seed_stability.json` |
| Instability profiles | `sidecar/results/s01/instability_profiles.json` |
| Taxonomy | `sidecar/results/s01/taxonomy.json` |
| Regime summaries | `sidecar/results/s01/regime_summaries.json` |
| Backbone shift figure | `sidecar/figures/s01_backbone_shift.svg` |
| Seed stability figure | `sidecar/figures/s01_seed_stability.svg` |
| Taxonomy scatter | `sidecar/figures/s01_taxonomy_scatter.svg` |
| Regime contrast | `sidecar/figures/s01_regime_contrast.svg` |
| Instability case table | `sidecar/results/s01/instability_case_table.md` |
| Instability working concept | `sidecar/notes/n05_instability_as_working_concept.md` |
| Per-layer norms (JSON) | `sidecar/results/per_layer_analysis/per_layer_norms.json` |
| Per-layer metrics (JSON) | `sidecar/results/per_layer_analysis/per_layer_metrics.json` |
| Group comparison (JSON) | `sidecar/results/per_layer_analysis/group_comparison.json` |
| Norm mass profiles figure | `sidecar/figures/norm_mass_profiles.svg` |
| Group comparison figure | `sidecar/figures/group_comparison.svg` |
| Collision subset (JSON) | `sidecar/results/collision_subset/collision_subset_table.json` |
| Within-layer geometry (JSON) | `sidecar/results/within_layer_geometry/geometry_metrics.json` |
| Within-layer group comparison figure | `sidecar/figures/within_layer_group_comparison.svg` |
| Within-layer CA-01 vs SC-QMRB figure | `sidecar/figures/within_layer_ca01_vs_scqmrb.svg` |
| Within-layer CA-02 toxic vs benign figure | `sidecar/figures/within_layer_ca02_toxic_vs_benign.svg` |
| Per-module subset table (JSON) | `sidecar/results/per_module_geometry/per_module_subset_table.json` |
| Per-module metrics (JSON) | `sidecar/results/per_module_geometry/module_metrics.json` |
| Per-module group comparison (JSON) | `sidecar/results/per_module_geometry/group_module_comparison.json` |
| Per-module discrimination (JSON) | `sidecar/results/per_module_geometry/module_discrimination.json` |
| Per-module seed sensitivity (JSON) | `sidecar/results/per_module_geometry/seed_sensitivity_per_module.json` |
| Per-module group comparison figure | `sidecar/figures/per_module_group_comparison.svg` |
| Per-module discrimination heatmap | `sidecar/figures/per_module_discrimination.svg` |
| Per-module CA-02 seed sensitivity figure | `sidecar/figures/per_module_ca02_seed_sensitivity.svg` |
| V-module spotlight figure | `sidecar/figures/per_module_v_spotlight.svg` |
| Head-level V panel table (JSON) | `sidecar/results/head_level_v/head_panel_table.json` |
| Head-level V metrics (JSON) | `sidecar/results/head_level_v/head_metrics.json` |
| Head-level V group comparison (JSON) | `sidecar/results/head_level_v/group_head_comparison.json` |
| Head-level V discrimination (JSON) | `sidecar/results/head_level_v/head_discrimination.json` |
| Head-level V seed sensitivity (JSON) | `sidecar/results/head_level_v/seed_sensitivity_per_head.json` |
| Head-level V summary descriptors (JSON) | `sidecar/results/head_level_v/head_summary_descriptors.json` |
| Head-level discrimination figure | `sidecar/figures/head_level_discrimination.svg` |
| Head-level CA-01 seed sensitivity figure | `sidecar/figures/head_level_ca01_seed_sensitivity.svg` |
| Head-level CA-02 seed sensitivity figure | `sidecar/figures/head_level_ca02_seed_sensitivity.svg` |
| Head-level worst-head spotlight figure | `sidecar/figures/head_level_worst_head_spotlight.svg` |
| Output-space panel table (JSON) | `sidecar/results/output_space/artifact_panel_table.json` |
| Output-space panel table (md) | `sidecar/results/output_space/artifact_panel_table.md` |
| Readout metrics (JSON) | `sidecar/results/output_space/readout_metrics.json` |
| Margin audit (JSON) | `sidecar/results/output_space/margin_audit.json` |
| Example behavior summary (JSON) | `sidecar/results/output_space/example_behavior_summary.json` |
| Readout alignment figure | `sidecar/figures/output_space_readout_alignment.svg` |
| Margin compression figure | `sidecar/figures/output_space_margin_compression.svg` |
| CA-01 seed contrast figure | `sidecar/figures/output_space_ca01_seed_contrast.svg` |
| Neither-task score figure | `sidecar/figures/output_space_neither_task.svg` |
| Seed panel table (JSON) | `sidecar/results/seed_readout/seed_panel_table.json` |
| Seed panel table (md) | `sidecar/results/seed_readout/seed_panel_table.md` |
| Coupling metrics (JSON) | `sidecar/results/seed_readout/coupling_metrics.json` |
| Family summary table (JSON) | `sidecar/results/seed_readout/family_summary_table.json` |
| Seed readout decision axis figure | `sidecar/figures/seed_readout_decision_axis.svg` |
| Seed readout coupling scatter figure | `sidecar/figures/seed_readout_coupling_scatter.svg` |
| Seed readout cross-task linkage figure | `sidecar/figures/seed_readout_cross_task_linkage.svg` |
| Attractor panel table (JSON) | `sidecar/results/attractor_mapping/attractor_panel_table.json` |
| Attractor panel table (md) | `sidecar/results/attractor_mapping/attractor_panel_table.md` |
| Family readout metrics (JSON) | `sidecar/results/attractor_mapping/family_readout_metrics.json` |
| Attractor classifications (JSON) | `sidecar/results/attractor_mapping/attractor_classifications.json` |
| Attractor family map figure | `sidecar/figures/attractor_mapping_family_map.svg` |
| Attractor backbone contrast figure | `sidecar/figures/attractor_mapping_backbone_contrast.svg` |
| Attractor convergence contrast figure | `sidecar/figures/attractor_mapping_convergence_contrast.svg` |
| Attractor domain contrast figure | `sidecar/figures/attractor_mapping_domain_contrast.svg` |
| Decision axis projections (JSON) | `sidecar/results/attractor_origin/decision_axis_projections.json` |
| PC loading profiles (JSON) | `sidecar/results/attractor_origin/pc_loading_profiles.json` |
| Cross-family axis alignment (JSON) | `sidecar/results/attractor_origin/cross_family_axis_alignment.json` |
| Attractor origin PC loadings figure | `sidecar/figures/attractor_origin_pc_loadings.svg` |
| Attractor origin effective axes figure | `sidecar/figures/attractor_origin_effective_axes.svg` |
| Attractor origin SV spectra figure | `sidecar/figures/attractor_origin_sv_spectra.svg` |
| Attractor origin cross-family heatmap | `sidecar/figures/attractor_origin_cross_family_heatmap.svg` |
| Mechanism classification table (JSON) | `sidecar/results/attractor_mechanisms/mechanism_classification_table.json` |
| Mechanism classification table (MD) | `sidecar/results/attractor_mechanisms/mechanism_classification_table.md` |
| Determinant matrix (JSON) | `sidecar/results/attractor_mechanisms/determinant_matrix.json` |
| Family factor table (JSON) | `sidecar/results/attractor_mechanisms/family_factor_table.json` |
| Commensurability context table (JSON) | `sidecar/results/attractor_mechanisms/commensurability_context_table.json` |
| Mechanism map figure | `sidecar/figures/attractor_mechanism_map.svg` |
| Determinant matrix figure | `sidecar/figures/attractor_mechanism_determinant_matrix.svg` |
| Convergence contrast panel figure | `sidecar/figures/attractor_mechanism_convergence_panel.svg` |

---

## 6. What Core Gradience Can Now Claim

1. **Boundary detection works.** Same-task pairs are safe to merge (53 pairs, 5 studies, max Δ = 3.4%). The cross-task boundary is reliably detected (0 false positives, 2 backbones).
2. **Severity grading does not work across backbones.** No tested signal predicts severity portably. This validates core's current design, which stops at boundary detection.
3. **Cross-task pairs are caution zones, not rejections.** The current stack reduces candidate space substantially in mixed-task inventories without false-blocking safe pairs.
4. **Instability is the first candidate for a portable descriptor.** But it is not yet promotable — it requires DeBERTa-v3 confirmation (see n05 §5 for explicit success criteria).
5. **V-module dimensionality ratio is the first candidate for a predictive signal.** Within the collision regime, catastrophic pairs show more asymmetric V-module effective ranks than safe pairs (d=3.36, zero overlap on backbone-controlled comparison). If confirmed on DeBERTa, this could become a computable pair-level warning signal — a qualitative advance over the current binary boundary. (See n21.)
6. **The mechanism is conjunctive, not single-factor.** Output-space readout analysis (n32) establishes that catastrophic failure requires both V-module pathology and readout incompatibility. The readout layer is a gate: compatible readout absorbs upstream pathology, incompatible readout transmits it. This means a future warning system needs to check both representation-space and readout-space conditions.
7. **Readout orthogonality is not a risk marker.** Same-task seed analysis (n36) shows that 5/14 same-task seed pairs have orthogonal decision axes yet merge safely. A stand-alone readout-cosine metric would false-alarm on ~40% of same-task merges. Only the conjunction of readout incompatibility with upstream V-module pathology is predictive. The two conditions are independently determined and independently measurable (n37).

---

## 7. What the Sidecar Should Pursue Next

**Priority 1:** DeBERTa-v3 replication (S01 completion). The adjudication test: do the same pairs remain the most *unstable*, regardless of which is catastrophic? Success criteria in n05 §5 and S01 §DeBERTa Success Criterion.

**Priority 2:** DeBERTa V-module replication. The per-module analysis (n21) identified V-module dimensionality ratio as the strongest signal (d=3.36, zero overlap). DeBERTa-v3's disentangled attention architecture separates content and position — if the V-module signal survives, it becomes a candidate predictive metric.

**Priority 3:** CA-01 seed sensitivity — now resolved at head level (n24). The 29-point seed range was invisible at per-module resolution but localizes to 7 heads with |Δ_DR| ≥ 0.15 (max 0.229 at layer 3 head 6). The mechanism is head-level cancellation: opposite-sign deltas at different heads average to near-zero at the module level. The remaining question is causal: why do certain head configurations produce catastrophe while others with the same module average do not? This likely requires O-module weighting analysis.

**Priority 4:** O-module head-weight analysis (CPU-feasible). The head-level cancellation mechanism (n24) implies the O module determines which heads' incompatibilities manifest as classification errors. Extracting per-head output weights from the O-module LoRA product could explain CA-01's sensitivity.

**Priority 5:** Readout attractor geometry — deep structure (CPU-feasible). The attractor mapping lab (n39–n41) has classified 10 families: 6 single-attractor, 3 multi-attractor, 1 backbone-contingent. Task identity is the primary determinant, modulated by backbone, training depth, and training distribution. The remaining question is *why* — why does QNLI have multiple attractors while RTE has one? This requires analyzing the 768-dimensional decision axes themselves, not just pairwise cosines. Stage C territory per the attractor mapping spec.

**Priority 6:** If DeBERTa confirms both instability portability and V-module signal, write up as a standalone finding suitable for external communication.
