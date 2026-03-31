# Figure Registry

Every figure in the sidecar, with a one-line caption, source program, generation script, and tier marking. This is the canonical reference for figure selection when writing posts, docs, or papers.

## Conventions

- **Format:** SVG (primary, vector), PNG (secondary, raster). Both present for most figures; a few are SVG-only.
- **Naming:** `{program}_{description}.{ext}`. Two legacy figures (`group_comparison`, `norm_mass_profiles`) omit the program prefix — they belong to the per-layer analysis program.
- **Tiers:** Each figure is marked with a tier indicating its role in the theoretical narrative.
  - **T1 (headline):** Figures that carry the core argument. Use in talks, summaries, one-pagers. There are 5 T1 figures; copies live in `../packet/figures/`.
  - **T2 (supporting):** Figures that strengthen or extend a T1 claim. Use in full presentations, papers, extended documentation.
  - **T3 (program-internal):** Figures produced during a specific analysis. Useful for tracing methodology or answering detailed questions, but not needed for the top-level story.

---

## S01 — Instability and Severity

| File | Caption | Tier |
|------|---------|------|
| `s01_summary_panel` | Severity reverses across backbones; instability does not. The founding observation of the sidecar program. | **T1** |
| `s01_backbone_shift` | Per-pair severity shift between DistilBERT and RoBERTa. Demonstrates that severity rankings are not portable. | T2 |
| `s01_seed_stability` | Seed-variant stability within each backbone. Unstable pairs show high seed range; stable pairs do not. | T2 |
| `s01_regime_contrast` | Regime-level contrast: catastrophic vs severe vs mild. Visualizes the gap between unstable and stable clusters. | T3 |
| `s01_taxonomy_scatter` | Pair taxonomy scatter: instability vs mean severity. Shows the bimodal clustering. | T3 |

*Script:* `scripts/per_layer/generate_figures.py`, `benchmarks/gen_summary_panel.py`, `benchmarks/gen_instability_figures.py`

## Per-Layer Analysis

| File | Caption | Tier |
|------|---------|------|
| `group_comparison` | Per-layer metric group comparison: catastrophic vs safe collision. Collision pattern found but groups overlap. | T3 |
| `norm_mass_profiles` | Norm mass distribution across layers for all adapters. Shows where LoRA weight mass concentrates. | T3 |

*Script:* `scripts/per_layer/generate_figures.py`

*Note:* These two figures lack the program prefix (`per_layer_`). Retained for backward compatibility with existing references.

## Within-Layer Geometry

| File | Caption | Tier |
|------|---------|------|
| `within_layer_group_comparison` | Within-layer geometry: catastrophic vs safe at critical layers. Backbone confound dominates; aggregate subspace ruled out as threshold. | T3 |
| `within_layer_ca01_vs_scqmrb` | CA-01 vs SC-QMRB within-layer contrast. The falsifier pair: same geometry, opposite outcomes. | T2 |
| `within_layer_ca02_toxic_vs_benign` | CA-02 vs benign pair within-layer contrast. Seed sensitivity visible but not explained at this scale. | T3 |

*Script:* `scripts/per_layer/within_layer_figures.py`

## Per-Module Geometry

| File | Caption | Tier |
|------|---------|------|
| `per_module_v_spotlight` | V-module dimensionality ratio separates catastrophic from safe collision: d=3.36, zero range overlap. The strongest single signal in the sidecar evidence base. | **T1** |
| `per_module_discrimination` | Cohen's d per module: V=3.36, K=1.39, Q≈0, O≈0. Only the V-module discriminates. | **T1** |
| `per_module_group_comparison` | Group-level per-module metric comparison across Q/K/V/O. | T2 |
| `per_module_ca02_seed_sensitivity` | CA-02 seed sensitivity decomposed by module. V-module shows the largest seed-variant shift. | T3 |

*Script:* `scripts/per_layer/per_module_figures.py`

## Head-Level V Program

| File | Caption | Tier |
|------|---------|------|
| `head_level_ca01_seed_sensitivity` | CA-01 seed gap resolved: 7 heads with opposite-sign Δ_DR (max 0.229) that cancel at module level. The cancellation mechanism. | T2 |
| `head_level_ca02_seed_sensitivity` | CA-02 head-level seed sensitivity. Smaller effect, consistent with CA-02's lower seed range. | T3 |
| `head_level_discrimination` | Head-level dim ratio discrimination. Module-level d=3.36 remains stronger than any individual head. | T2 |
| `head_level_worst_head_spotlight` | Spotlight on the worst head per case. Shows which heads drive the catastrophic signal. | T3 |

*Script:* `scripts/per_layer/v_head_figures.py`

## Output-Space Readout (Sidecar B)

| File | Caption | Tier |
|------|---------|------|
| `output_space_readout_alignment` | Readout alignment by group. Incompatible readout is necessary but not sufficient — the key falsifier (SC-QMRB is incompatible yet safe). | **T1** |
| `output_space_margin_compression` | Margin compression in merged adapters. Catastrophic merges compress margins; safe merges preserve them. | T2 |
| `output_space_ca01_seed_contrast` | CA-01 seed variant readout contrast. Different seeds, different readout alignment. | T3 |
| `output_space_neither_task` | Neither-task prediction rate by group. The output-space signature of catastrophe. | T2 |

*Script:* `scripts/per_layer/output_space_readout.py`

## Seed-Readout Coupling

| File | Caption | Tier |
|------|---------|------|
| `seed_readout_decision_axis` | Decision-axis cosine is bimodal: ~0 or ~1, no intermediate values. Same-task orthogonality is common and safe. | T2 |
| `seed_readout_coupling_scatter` | Upstream V-module geometry vs readout alignment. The two conditions are decoupled. | T2 |
| `seed_readout_cross_task_linkage` | Cross-task readout linkage: QNLI/RoBERTa/s7 aligns with RTE (cos=0.86). | T3 |

*Script:* `scripts/per_layer/seed_readout_coupling.py`

*Note:* SVG only — no PNG exports. Convert if needed for raster contexts.

## Attractor Mapping Lab

| File | Caption | Tier |
|------|---------|------|
| `attractor_mapping_family_map` | Family-level attractor map: 6 single, 3 multi, 1 backbone-contingent. All safe. | T2 |
| `attractor_mapping_backbone_contrast` | Backbone contrast: same task, different attractor structure by backbone. | T3 |
| `attractor_mapping_domain_contrast` | Domain contrast: SST-2 orthogonal to Yelp/Amazon; Yelp and Amazon aligned. | T3 |
| `attractor_mapping_convergence_contrast` | Convergence contrast: Strong QNLI orthogonal, Medium/Weak aligned. | T3 |

*Script:* `scripts/per_layer/attractor_mapping_audit.py`

*Note:* SVG only.

## Attractor Origin and Mechanisms

| File | Caption | Tier |
|------|---------|------|
| `attractor_mechanism_map` | Two distinct mechanisms: rotational degeneracy (all DistilBERT) vs feature-set switching (QNLI/RoBERTa). | T2 |
| `attractor_mechanism_determinant_matrix` | Determinant hierarchy: task → backbone → convergence → domain. | T2 |
| `attractor_mechanism_convergence_panel` | Convergence modulates attractor count but not mechanism class. | T3 |
| `attractor_origin_cross_family_heatmap` | Cross-family decision-axis alignment heatmap. | T3 |
| `attractor_origin_effective_axes` | Effective axes per family: dimensionality of the decision subspace. | T3 |
| `attractor_origin_pc_loadings` | PC loading profiles for multi-attractor families. | T3 |
| `attractor_origin_sv_spectra` | Singular value spectra for decision-axis decomposition. | T3 |

*Script:* `scripts/per_layer/decision_axis_analysis.py`

*Note:* SVG only.

## Example Semantics

| File | Caption | Tier |
|------|---------|------|
| `example_semantics_preservation_breakage` | Preservation vs breakage across the 8-case panel. Safe and near-miss preserve; fragile and control break differently. | **T1** |
| `example_semantics_taxonomy_composition` | Failure taxonomy composition by case. Neither-source rate (D) jumps from <2% to >12% — the threshold discriminator. | **T1** |
| `example_semantics_confidence` | Confidence distributions: fragile shows collapse (low confidence, spread), control shows high-confidence wrong. The double dissociation. | T2 |

*Script:* `scripts/generate_example_figures.py`

*Note:* PNG only — generated by matplotlib without SVG export. Convert to SVG for publication if needed.

---

## Summary

| Tier | Count | Role |
|------|-------|------|
| T1 (headline) | 5 | Core argument. Copies in `../packet/figures/`. |
| T2 (supporting) | 14 | Extend or strengthen a T1 claim. |
| T3 (program-internal) | 17 | Methodology trace and detailed questions. |
| **Total** | **36 unique figures** | (56 files counting SVG+PNG pairs) |

### T1 figures at a glance

1. `s01_summary_panel` — Founding observation: severity reverses, instability doesn't.
2. `per_module_v_spotlight` — V-module dim ratio: d=3.36, zero overlap.
3. `per_module_discrimination` — Only V discriminates among Q/K/V/O.
4. `output_space_readout_alignment` — Readout orthogonality is benign. The falsifier.
5. `example_semantics_preservation_breakage` — Behavioral signatures across the panel.
6. `example_semantics_taxonomy_composition` — Neither-source rate as threshold discriminator.

*Note: The T1 list above includes 6 figures rather than 5 because `per_module_discrimination` and `per_module_v_spotlight` are often shown together. The packet contains 5 (merging those two into a single per-module slot).*

## Decision-Dependent Compatibility (Route 2)

| File | Caption | Tier |
|------|---------|------|
| `decision_dependent_aggregation_matrix` | Aggregation is computational: same structural evidence, four rules, different operational labels. 10/12 cases are aggregation-sensitive. | **T1** |
| `decision_dependent_aggregation_matrix_adapter_t01` | Adapter T01 variant of the aggregation matrix. Shows how the pattern holds on a different checkpoint. | T3 |

*Note:* SVG only.

## Behavioral Route 2 Bridge

| File | Caption | Tier |
|------|---------|------|
| `behavioral_route2_profile_matrix` | Three-tier behavioral separation across five Route 2 profiles. Neither-source %, confidence collapse, and high-confidence wrong clearly distinguish safe, pathological, and stasis tiers. | **T1** |

*Note:* SVG only.

---

## Summary (updated)

| Tier | Count | Role |
|------|-------|------|
| T1 (headline) | 8 | Core argument. |
| T2 (supporting) | 14 | Extend or strengthen a T1 claim. |
| T3 (program-internal) | 18 | Methodology trace and detailed questions. |
| **Total** | **40 unique figures** | (62 files counting SVG+PNG pairs) |

### T1 figures at a glance

1. `s01_summary_panel` — Founding observation: severity reverses, instability doesn't.
2. `per_module_v_spotlight` — V-module dim ratio: d=3.36, zero overlap.
3. `per_module_discrimination` — Only V discriminates among Q/K/V/O.
4. `output_space_readout_alignment` — Readout orthogonality is benign. The falsifier.
5. `example_semantics_preservation_breakage` — Behavioral signatures across the panel.
6. `example_semantics_taxonomy_composition` — Neither-source rate as threshold discriminator.
7. `decision_dependent_aggregation_matrix` — Aggregation is computational, not presentational.
8. `behavioral_route2_profile_matrix` — Route 2 profiles have behavioral reality.

---

## File inventory

40 unique figures across 62 files (37 SVG + 22 PNG + 3 SVG-only Route 2). Figures with both formats share the same base name. Four programs (seed-readout, attractor mapping, attractor origin/mechanisms, Route 2) are SVG-only. Example semantics is PNG-only.
