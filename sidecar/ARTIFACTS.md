# Artifact Index

A stable, navigable index of every major artifact in the sidecar and field trial directories. Organized by function, not by creation order. For the theoretical narrative, see `notes/n67_where_the_research_stands.md`. For project status, see `notes/n69_settled_open_next.md`.

*Last updated: 2026-04-01. 109 notes, 17 scripts, 114+ JSON results, 67 figures, 5 primary field trial inventories, 5 Ring 2 checkpoints.*

---

## Canonical memos

The documents that synthesize the research program. Read in this order for the complete picture.

| Document | Role |
|----------|------|
| `notes/n69_settled_open_next.md` | State-of-project dashboard. What is established, open, and next. |
| `notes/n67_where_the_research_stands.md` | Mechanism-ladder synthesis. Commensurability, V-module pathology, conjunctive failure, behavioral signatures. Supersedes n51, n25. |
| `notes/n93_route2_synthesis.md` | Route 2 synthesis. Decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge. Companion to n67. |
| `../docs/strategy/route2_claims_ladder_summary.md` | Route 2 claims ladder summary. Stable/moderate/thin/local calibration for communication and scope discipline. |
| `notes/n112_route2_claims_ladder_implications.md` | Route 2 claims ladder implications memo. Public/product/internal language guidance from calibrated claim statuses. |
| `../docs/strategy/aggregation_mixed_evidence_summary.md` | Mixed-evidence triage perturbation summary. Soft-middle coherence and guardrails for review/optional language. |
| `notes/n68_ruled_out_mechanisms.md` | Ruled-out mechanisms (definitive). Ten primary, five ancillary eliminations. Supersedes n38, n52. |
| `notes/n50_executive_research_summary.md` | Executive summary. Product identity, central claim, practitioner value. |
| `notes/n33_conjunctive_mechanism_synthesis.md` | Conjunctive model. V-module pathology × readout incompatibility → catastrophe. |
| `notes/n49_mechanism_and_commensurability_synthesis.md` | Commensurability v3. Readout decomposes by mechanism class. |
| `notes/n25_multiscale_mechanism_synthesis.md` | Multiscale mechanism ladder. Module → head → readout. Superseded by n67 but retains per-rung detail. |
| `notes/n66_behavior_mechanism_bridge.md` | Bridges example-level behavior to the mechanism ladder. Double dissociation confirmed. |
| `glossary.md` | Frozen definitions for canonical terms. |
| `../docs/strategy/broadened_substrate_scope.md` | Route 2 bounded scope checkpoint. Consolidated architecture/product posture. |
| `strategy_memo.md` | Founding document. Research programs, workstreams, promotion rules. |

## Strongest figures

The figures that tell the core theoretical story. Copies of the top 5 are also in `packet/figures/`.

| Figure | What it shows | Source program |
|--------|--------------|----------------|
| `figures/s01_summary_panel.png` | Founding observation: severity reverses across backbones, instability does not. | S01 analysis |
| `figures/per_module_v_spotlight.png` | V-module dim ratio separates catastrophic from safe (d=3.36, zero overlap). | Per-module geometry |
| `figures/per_module_discrimination.png` | All four modules compared — V discriminates, Q and O do not. | Per-module geometry |
| `figures/output_space_readout_alignment.png` | Readout orthogonality is benign in isolation. The key falsifier. | Output-space audit |
| `figures/head_level_ca01_seed_sensitivity.png` | CA-01 seed gap resolved at head level: 7 heads with opposite-sign deltas. | Head-level V program |
| `figures/head_level_discrimination.png` | Head-level dim ratio — module-level d=3.36 remains stronger. | Head-level V program |
| `figures/seed_readout_decision_axis.svg` | Bimodal readout: either ~0 or ~1, no intermediate. | Seed-readout coupling |
| `figures/attractor_mechanism_map.svg` | Two mechanisms: rotational degeneracy vs feature-set switching. | Attractor mechanisms |
| `figures/example_semantics_preservation_breakage.png` | Behavioral signatures: preservation vs breakage across the 8-case panel. | Example semantics |
| `figures/example_semantics_taxonomy_composition.png` | Failure taxonomy composition — neither-source rate as threshold discriminator. | Example semantics |
| `figures/example_semantics_confidence.png` | Confidence distributions: collapse in fragile vs high-confidence wrong in control. | Example semantics |

All 57 figures (34 SVG + 22 PNG + README) are in `figures/`. SVG for vector clarity, PNG for compatibility.

## Field trial artifacts

Operational validation of the preflight workflow. 5 inventories, 3 backbones, 16 evaluated merges.

### Inventories

| Directory | Adapters | Tasks | Backbone | Key finding |
|-----------|----------|-------|----------|-------------|
| `../field_trials/inventory_01_same_task_control/` | 4 | 1 (IMDB) | DistilBERT | Confirmatory workflow. All missing behavioral evidence → all excluded. |
| `../field_trials/inventory_02_mixed_task/` | 5 | 4 | RoBERTa | **Best result.** 10→1 retained, 90% reduction. Correct same-task detection. |
| `../field_trials/inventory_03_large_mixed_task/` | 9 | 6 | DistilBERT | Scale test. 36→3 retained. Neighborhoods useful at this size. |
| `../field_trials/inventory_04_distilbert_irony_cluster/` | 4 | 1 (irony) | DistilBERT | Single-task cluster. Near-miss detection validated. |
| `../field_trials/inventory_05_bert_hate_emotion/` | 6 | 2 | BERT | Multi-task cluster. Weak-source handling validated. |

### HTML preflight reports (open in browser)

| Report | Inventory |
|--------|-----------|
| `../field_trials/inventory_01_same_task_control/preflight_v2/report.html` | Same-task control |
| `../field_trials/inventory_02_mixed_task/preflight_ev_122511/report.html` | Mixed-task (best result) |
| `../field_trials/inventory_03_large_mixed_task/preflight_ev_123354/report.html` | Large mixed-task |
| `../field_trials/inventory_04_distilbert_irony_cluster/preflight_ev_145418/report.html` | Irony cluster |
| `../field_trials/inventory_05_bert_hate_emotion/preflight_ev_145533/report.html` | Hate + emotion |
| `../preflight_report_demo.html` | Demo report (standalone) |

### Preflight bundles (machine-readable)

Each inventory's `preflight_ev_*/bundle/` directory contains: `preflight_summary.json`, `preflight_summary.md`, `inventory_action_plan.md`, `review_packet.json`, `review_packet.md`, `run_manifest.json`.

### Evaluation results and analysis

| Document | What it covers |
|----------|---------------|
| `../field_trials/phase2_eval_130608/phase2_results.json` | Raw evaluation results for all 16 merges (retained, near-miss, control). |
| `../field_trials/product_validation_memo.md` | What the pipeline got right. Retained avg Δ=-0.024, near-miss -0.006, control -0.047. |
| `../field_trials/near_miss_validation.md` | Near-miss evaluation across 3 backbones and 3 task families. |
| `../field_trials/phase2_evaluation_report.md` | Full Phase 2 evaluation narrative. |
| `../field_trials/phase2b_confirmation_memo.md` | Phase 2b confirmation (near-miss + additional controls). |
| `../field_trials/validation_memo.md` | Phase 1 validation memo. |
| `../field_trials/pilot_phase1_comparison.md` | Phase 1 vs Phase 2 comparison. |
| `../field_trials/adapter_landscape_assessment.md` | Adapter landscape assessment across hubs. |
| `../field_trials/cpu_field_research_protocol.md` | **Next-phase protocol.** Four targeted micro-campaigns: task-family equivalence, marginal-adapter behavior, large-inventory stress, public-ecosystem robustness. |

### Next-phase micro-campaign directories

| Directory | Micro-campaign |
|-----------|---------------|
| `../field_trials/task_family_equivalence/` | A — Is exact task identity too strict? |
| `../field_trials/marginal_adapter_behavior/` | B — Do barely-weak adapters behave like near-miss or excluded? |
| `../field_trials/large_inventory_stress/` | C — Ergonomics at 10–14 adapters / 40–90 pairs. |
| `../field_trials/public_ecosystem_robustness/` | D — Messy real-world adapter handling. |
| `../field_trials/synthesis/` | Cross-campaign summary and product implications. |

### Field trial scripts

| Script | Purpose |
|--------|---------|
| `../field_trials/run_pilot.py` | Phase 1 pilot runner. |
| `../field_trials/run_phase2_eval.py` | Phase 2 evaluation runner. |
| `../field_trials/run_phase2b_eval.py` | Phase 2b confirmation runner. |
| `../field_trials/evidence_bootstrap.py` | Evidence bootstrap for adapter QA. |
| `../field_trials/rerun_preflight.py` | Preflight rerun utility. |

## Sidecar evidence tables

Structured JSON result files, organized by research program. All in `results/`.

### S01 — Catastrophic Anchor Replication (9 files)

| File | Contents |
|------|----------|
| `results/s01/instability_profiles.json` | Per-pair instability scores and components. |
| `results/s01/instability_case_table.json` | 12-row case table (all pairs × both backbones). |
| `results/s01/three_backbone_comparison.json` | Cross-backbone severity comparison. |
| `results/s01/taxonomy.json` | Pair classification taxonomy. |
| `results/s01/distilbert_analysis.json` | DistilBERT per-pair analysis. |
| `results/s01/roberta_analysis.json` | RoBERTa per-pair analysis. |
| `results/s01/seed_stability.json` | Seed-variant stability data. |
| `results/s01/regime_summaries.json` | Per-regime summary statistics. |
| `results/s01/same_task_contrast.json` | Same-task vs cross-task contrast. |

Markdown tables: `results/s01/instability_case_table.md`, `results/s01/instability_case_table_extended.md`, `results/s01/backbone_shift_table.md`.

### Per-module geometry (6 files)

| File | Contents |
|------|----------|
| `results/per_module_geometry/module_metrics.json` | Per-variant, per-module metrics (Q/K/V/O). |
| `results/per_module_geometry/module_discrimination.json` | Cohen's d and overlap per module. V=3.36, K=1.39, Q≈0, O≈0. |
| `results/per_module_geometry/group_module_comparison.json` | Group-level module comparison. |
| `results/per_module_geometry/seed_sensitivity_per_module.json` | Seed sensitivity decomposed by module. |
| `results/per_module_geometry/per_module_subset_table.json` | Module correspondence across backbones. |

### Head-level V (7 files)

| File | Contents |
|------|----------|
| `results/head_level_v/head_metrics.json` | Per-variant, per-head V metrics. |
| `results/head_level_v/head_discrimination.json` | Head-level discrimination statistics. |
| `results/head_level_v/seed_sensitivity_per_head.json` | Per-head seed deltas for CA-01 and CA-02. |
| `results/head_level_v/head_summary_descriptors.json` | Summary descriptors per case. |
| `results/head_level_v/group_head_comparison.json` | Group-level head comparison. |
| `results/head_level_v/head_panel_table.json` | Head panel definition. |

### Output-space readout (4 files)

`results/output_space/`: `readout_metrics.json`, `margin_audit.json`, `artifact_panel_table.json`, `example_behavior_summary.json`.

### Seed-readout coupling (3 files)

`results/seed_readout/`: `coupling_metrics.json`, `family_summary_table.json`, `seed_panel_table.json`.

### Attractor programs (10 files)

`results/attractor_mapping/`: `attractor_classifications.json`, `attractor_panel_table.json`, `family_readout_metrics.json`.

`results/attractor_mechanisms/`: `mechanism_classification_table.json`, `determinant_matrix.json`, `family_factor_table.json`, `commensurability_context_table.json`.

`results/attractor_origin/`: `cross_family_axis_alignment.json`, `decision_axis_projections.json`, `pc_loading_profiles.json`.

### Example semantics (22 files)

`results/example_semantics/`: 8 analyzed case files (`analyzed_SR-01.json` through `analyzed_AN-01.json`), `example_behavior_summary.json`, `example_flip_catalog.json`, `failure_taxonomy.json`, `mechanism_bridge_table.json`, `panel_table.json`, `preservation_breakage_table.json`.

`results/example_semantics/predictions/`: 8 raw prediction files (one per panel case).

### Earlier programs (8 files)

`results/per_layer_analysis/`: `per_layer_metrics.json`, `per_layer_norms.json`, `group_comparison.json`, `artifact_inventory.json`.

`results/within_layer_geometry/`: `geometry_metrics.json`, `group_comparison.json`, `contrast_panel.json`.

`results/collision_subset/`: `collision_subset_table.json`.

## Research packet

Self-contained bundle for collaborators. All in `packet/`.

| File | Contents |
|------|----------|
| `packet/00_packet_index.md` | Reading order and figure descriptions. |
| `packet/01_where_the_research_stands.md` | Anchor memo (copy of n67). |
| `packet/02_product_validation.md` | Condensed product validation (~600 words). |
| `packet/03_ruled_out.md` | Ruled-out mechanisms (copy of n68). |
| `packet/04_evidence_table.md` | 8 settled claims, 4 thin claims, 5 open questions, 6 ruled-out hypotheses. |
| `packet/05_gpu_reentry.md` | GPU re-entry note. DeBERTa protocol, predictions, decision tree. |
| `packet/figures/` | 5 strongest figures (PNG copies). |

### Route 2 Sub-Packet

Self-contained bundle for the broadened-compatibility story. All in `packet/route2/`.

| File | Contents |
|------|----------|
| `packet/route2/00_route2_packet_index.md` | Reading order, background, one-paragraph summary, pointers to all Route 2 documents. |
| `packet/route2/01_route2_orientation.md` | Substrate generalization: three axes (scenario, artifact-class, representation-path), what generalized, what didn't. |
| `packet/route2/figures/` | 2 key figures (symlinks): behavioral profile matrix, aggregation case matrix. |

References (not duplicated): n93 (synthesis), cross-artifact summary, aggregation-sensitive summary, behavioral summary, Ring 1 results, Ring 2 assessment, routing pilot results, checkpoint triage alpha workflow.

## GPU-blocked protocols

Everything needed to execute the DeBERTa adjudication when compute returns.

| Document | Role |
|----------|------|
| `notes/n07_deberta_adjudication_protocol.md` | **Complete pre-registered protocol.** Training, merging, evaluation, analysis, predictions A–F, decision tree, output spec, pre-run checklist. Executable as-is. |
| `notes/n06_instability_program_statement.md` | Program statement. Commitments, predictions, falsification conditions. |
| `packet/05_gpu_reentry.md` | Condensed re-entry note. 5 predictions, decision tree, ~3h budget. |
| `results/s01/instability_case_table_extended.md` | Extended case table with DeBERTa predictions and confidence ratings. |
| `studies/s01_catastrophic_anchor_replication.md` | Founding study. Two-backbone results, DeBERTa success criteria. |
| `panels/p01_catastrophic_anchors.md` | Canonical anchor panel with severity thresholds. |

## Analysis scripts

17 Python scripts for reproducing all computed results. All in `scripts/`.

### Top-level (example semantics program)

| Script | Purpose |
|--------|---------|
| `scripts/collect_example_predictions.py` | Collect per-example predictions for the 8-case panel. |
| `scripts/analyze_example_behavior.py` | Compute behavioral metrics from predictions. |
| `scripts/build_failure_taxonomy.py` | Derive the 5-category failure taxonomy from raw categories. |
| `scripts/generate_example_figures.py` | Generate example semantics figures (PNG). |

### Per-layer subdirectory (spectral geometry programs)

| Script | Purpose |
|--------|---------|
| `scripts/per_layer/compute_metrics.py` | Per-layer structural metrics for all pairs. |
| `scripts/per_layer/per_module_geometry.py` | Per-module (Q/K/V/O) geometry analysis. |
| `scripts/per_layer/v_head_geometry.py` | Head-level V-module analysis. |
| `scripts/per_layer/within_layer_geometry.py` | Within-layer collision geometry. |
| `scripts/per_layer/output_space_readout.py` | Readout alignment and margin audit. |
| `scripts/per_layer/seed_readout_coupling.py` | Seed-contingent readout-axis coupling. |
| `scripts/per_layer/attractor_mapping_audit.py` | Family readout attractor classification. |
| `scripts/per_layer/decision_axis_analysis.py` | Decision-axis PCA and cross-family alignment. |
| `scripts/per_layer/generate_figures.py` | Generate per-layer and S01 figures (SVG + PNG). |
| `scripts/per_layer/per_module_figures.py` | Generate per-module figures. |
| `scripts/per_layer/v_head_figures.py` | Generate head-level figures. |
| `scripts/per_layer/within_layer_figures.py` | Generate within-layer figures. |

### Benchmarking scripts

8 Python scripts in `benchmarks/` for instability analysis and figure generation from S01 data: `instability_analysis.py`, `compile_adjudication.py`, `gen_summary_panel.py`, `gen_instability_figures.py`, `gen_instability_case_table.py`, `gen_backbone_shift_figure.py`, `gen_seed_stability_figure.py`.

## Ring 1 PEFT Generalization

Artifact-class generality validation. LoHa adapters processed through the full Gradience pipeline via extraction shim. Zero core code modifications.

| Artifact | Role |
|----------|------|
| `../docs/strategy/ring1_peft_generalization_results.md` | **Assessment memo.** What generalized, what needed a shim, what stayed LoRA-specific. |
| `../docs/design/peft_generalization_audit.md` | **Design doc.** Substrate analysis, candidate assessment, shim architecture. |
| `../experiments/peft_ring1/artifact_support_matrix.json` | Support matrix: LORA, LOHA, LOKR, IA3 feasibility ratings. |
| `../experiments/peft_ring1/artifact_support_matrix.md` | Human-readable support matrix. |
| `../experiments/peft_ring1/loha_shim.py` | Extraction shim (~160 lines). Converts LoHa state dicts to LoRA format. |
| `../experiments/peft_ring1/measurement_compatibility_results.json` | Stage B results: 6 audit runs (3 adapters x 2 modes), all successful. |
| `../experiments/peft_ring1/inventory_pilot/` | Stage C: 3 pairwise reports, inventory summary, preflight bundle, field note. |
| `../experiments/peft_ring1/adapters/` | 3 trained LoHa adapters (r4/r8/r16, distilbert, SST-2). |

## Ring 2 Checkpoint-Delta Generalization

Representation-path generality validation. Full fine-tuned checkpoint deltas processed through audit and triage via summary-based representation (Representation C). Zero core code modifications.

| Artifact | Role |
|----------|------|
| `../docs/design/ring2_stage_a_checkpoint_delta_representation.md` | **Stage A design note.** Three representations compared; C selected. |
| `../docs/design/ring2_stage_b_representation_c_audit.md` | **Stage B design note.** Audit and pairwise comparison on summaries. |
| `../docs/design/ring2_stage_c_guardrail_triage.md` | **Stage C design note.** Guardrail triage and run-bundle packaging. |
| `../docs/design/ring2_stage_d_assessment_memo.md` | **Stage D assessment.** Plain assessment: workflow survives, representation path differs. |
| `../experiments/ring2_checkpoint_delta/checkpoint_delta.py` | Core extraction module: delta computation + 3 representations. |
| `../experiments/ring2_checkpoint_delta/train_checkpoints.py` | Training script for 5 distilbert checkpoints (sst2, mrpc, qnli, yelp). |
| `../experiments/ring2_checkpoint_delta/run_stage_a.py` | Stage A comparison harness. |
| `../experiments/ring2_checkpoint_delta/run_stage_b.py` | Stage B audit and pairwise harness. |
| `../experiments/ring2_checkpoint_delta/run_stage_c.py` | Stage C guardrail triage harness. |
| `../experiments/ring2_checkpoint_delta/stage_a_representation_results.json` | Stage A structured results. |
| `../experiments/ring2_checkpoint_delta/stage_a_representation_results.md` | Stage A human-readable results. |
| `../experiments/ring2_checkpoint_delta/stage_b_representation_c_results.json` | Stage B structured results. |
| `../experiments/ring2_checkpoint_delta/stage_b_representation_c_results.md` | Stage B human-readable results. |
| `../experiments/ring2_checkpoint_delta/stage_c_inventory_results.json` | Stage C structured results. |
| `../experiments/ring2_checkpoint_delta/stage_c_inventory_results.md` | Stage C human-readable results. |
| `../experiments/ring2_checkpoint_delta/checkpoints/` | 5 trained checkpoints (sst2_s42, sst2_s123, mrpc_s42, qnli_s42, yelp_s42). |
| `../field_trials/checkpoint_inventory_t02/build_alpha_bundle.py` | Route 2 alpha bundle builder (renders polished checkpoint-triage report). |
| `../field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html` | Canonical checkpoint-triage alpha HTML report. |
| `../field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/alpha_summary.json` | Compact alpha summary for automation and quick review. |
| `../docs/examples/checkpoint-triage-alpha-workflow.md` | Short \"how to use\" doc for checkpoint-triage alpha workflow. |
| `../docs/strategy/checkpoint_triage_alpha_scope.md` | Compact alpha scope contract (shared base, small encoders, classification, evidence required). |

## Cross-Artifact Compatibility Research Program (n76-n80)

Cross-artifact portability study: which compatibility signals transfer across LoRA, LoHa, and checkpoint delta artifact classes.

| Artifact | Role |
|----------|------|
| `notes/n76_cross_artifact_panel_definition.md` | Stage A: 9-case panel across 3 artifact classes. |
| `notes/n77_cross_artifact_invariant_signal_audit.md` | Stage B: invariant signal audit (5 families). |
| `notes/n78_representation_local_signal_audit.md` | Stage C: representation-local signal audit (7 signals). |
| `notes/n79_cross_artifact_compatibility_framework.md` | Stage D: three-layer compatibility framework. |
| `notes/n80_cross_artifact_product_relevance.md` | Stage E: product relevance filter. |
| `../docs/strategy/cross_artifact_product_relevance_summary.md` | Product-facing summary. |
| `results/cross_artifact_portability/panel_table.json` | Panel table (JSON). |
| `results/cross_artifact_portability/panel_table.md` | Panel table (markdown). |
| `results/cross_artifact_portability/invariant_signal_matrix.json` | Invariant signal matrix (JSON). |
| `results/cross_artifact_portability/invariant_signal_matrix.md` | Invariant signal matrix (markdown). |
| `results/cross_artifact_portability/local_signal_table.json` | Local signal table (JSON). |
| `results/cross_artifact_portability/local_signal_table.md` | Local signal table (markdown). |
| `results/cross_artifact_portability/framework_table.json` | Framework table (JSON). |
| `results/cross_artifact_portability/framework_table.md` | Framework table (markdown). |
| `results/cross_artifact_portability/product_relevance_filter.json` | Product relevance filter (JSON). |

## Behavioral Route 2 Bridge (n86-n92)

Behavioral bridge study: do broadened Route 2 compatibility profiles have distinct example-level behavioral signatures?

| Artifact | Role |
|----------|------|
| `notes/n86_behavioral_route2_panel_definition.md` | Stage A: 8-case panel across 5 Route 2 profiles. |
| `notes/n87_behavioral_route2_protocol.md` | Stage B: 6-metric, 6-category behavioral protocol. |
| `notes/n88_behavioral_route2_findings.md` | Stage C: profile-to-behavior findings. Three-tier model. |
| `notes/n89_behavioral_route2_dossier_optional_vs_fragile.md` | Dossier 1: optional vs fragile (threshold, not gradient). |
| `notes/n90_behavioral_route2_dossier_confusable_vs_separable.md` | Dossier 2: confusable vs separable (structural ≠ behavioral). |
| `notes/n91_behavioral_route2_dossier_qa_override.md` | Dossier 3: QA stasis vs structural collapse. |
| `notes/n92_behavioral_route2_bridge.md` | Stage E: bridge synthesis connecting to prior Route 2 programs. |
| `../docs/strategy/behavioral_route2_summary.md` | Product-facing summary. |
| `results/behavioral_route2_bridge/panel_table.json` | Panel table (JSON). |
| `results/behavioral_route2_bridge/panel_table.md` | Panel table (markdown). |
| `results/behavioral_route2_bridge/protocol_schema.json` | Protocol schema (JSON). |
| `results/behavioral_route2_bridge/behavior_summary.json` | Behavior summary (JSON). |
| `results/behavioral_route2_bridge/profile_behavior_table.json` | Profile behavior table (JSON). |
| `results/behavioral_route2_bridge/profile_behavior_table.md` | Profile behavior table (markdown). |
| `results/behavioral_route2_bridge/behavior_bridge_table.json` | Bridge table (JSON). |
| `figures/behavioral_route2_profile_matrix.svg` | Profile matrix figure. |

## Aggregation-Sensitive Compatibility Research Program (n81-n85)

Aggregation-sensitive compatibility study: how different aggregation rules transform the same structural evidence into different operational judgments.

| Artifact | Role |
|----------|------|
| `notes/n81_aggregation_panel_definition.md` | Stage A: 12-case panel with matched QA-regime pairs. |
| `notes/n82_aggregation_family_audit.md` | Stage B: 4 aggregation families formalized. |
| `notes/n83_aggregation_comparison_analysis.md` | Stage C: 12-case comparison matrix (2 full, 2 strong divergence, 8 partial). |
| `notes/n84_aggregation_sensitive_pattern_taxonomy.md` | Stage D: 5-pattern taxonomy. |
| `notes/n85_aggregation_sensitive_operational_implications.md` | Stage E: operational implications memo. |
| `../docs/strategy/aggregation_sensitive_route2_summary.md` | Product-facing Route 2 summary. |
| `results/aggregation_sensitive_compatibility/panel_table.json` | Panel table (JSON). |
| `results/aggregation_sensitive_compatibility/panel_table.md` | Panel table (markdown). |
| `results/aggregation_sensitive_compatibility/aggregation_family_specs.json` | Family specs (JSON). |
| `results/aggregation_sensitive_compatibility/aggregation_family_specs.md` | Family specs (markdown). |
| `results/aggregation_sensitive_compatibility/aggregation_comparison.json` | Comparison matrix (JSON). |
| `results/aggregation_sensitive_compatibility/aggregation_comparison.md` | Comparison matrix (markdown). |
| `results/aggregation_sensitive_compatibility/pattern_taxonomy.json` | Pattern taxonomy (JSON). |
| `results/aggregation_sensitive_compatibility/pattern_taxonomy.md` | Pattern taxonomy (markdown). |
| `results/aggregation_sensitive_compatibility/operational_implications.json` | Operational implications (JSON). |

## Decision-Dependent Compatibility (n70-n74)

Bounded sidecar program on scenario-dependent compatibility interpretation across merge, routing, and triage.

| Artifact | Role |
|----------|------|
| `notes/n70_decision_dependent_panel_definition.md` | Stage A panel definition (9-case shared panel, overlap across scenarios). |
| `notes/n71_shared_measurement_divergent_policy_audit.md` | Stage B audit of shared measurement vs scenario-specific layers. |
| `notes/n72_aggregation_sensitive_compatibility.md` | Stage C aggregation analysis (worst-case vs distributional vs QA gate-first). |
| `notes/n73_decision_profile_taxonomy.md` | Stage D six-profile decision taxonomy. |
| `notes/n74_decision_semantics_bridge.md` | Stage E behavioral bridge and caveats. |
| `scripts/build_decision_dependent_compatibility.py` | Builder script that emits all structured artifacts for n70-n74. |
| `results/decision_dependent_compatibility/panel_table.json` | Stage A panel table (JSON). |
| `results/decision_dependent_compatibility/scenario_stack_matrix.json` | Stage B scenario stack matrix. |
| `results/decision_dependent_compatibility/aggregation_comparison.json` | Stage C checkpoint-triage aggregation comparison (T02). |
| `results/decision_dependent_compatibility/aggregation_comparison_adapter_t01.json` | Stage C narrow stress test on adapter triage substrate (T01). |
| `results/decision_dependent_compatibility/decision_profile_table.json` | Stage D profile table (6 profiles). |
| `results/decision_dependent_compatibility/decision_semantics_table.json` | Stage E semantics bridge table. |
| `figures/decision_dependent_aggregation_matrix.svg` | Stage C matrix figure for checkpoint T02 pass. |
| `figures/decision_dependent_aggregation_matrix_adapter_t01.svg` | Stage C matrix figure for adapter T01 stress pass. |

## Cross-Artifact Compatibility Seed Note (n75)

Foundational cross-artifact comparison note for Route 2. Distinguishes substrate-shared versus representation-specific compatibility signals across LoRA, LoHa, and full checkpoint deltas; later expanded by n76-n80.

| Artifact | Role |
|----------|------|
| `notes/n75_cross_artifact_compatibility.md` | Bounded synthesis note: what transfers across artifact classes and what remains representation-local. |
| `results/cross_artifact_compatibility/shared_vs_specific_table.json` | Structured table of shared vs specific compatibility signals. |
| `results/cross_artifact_compatibility/shared_vs_specific_table.md` | Human-readable shared-vs-specific comparison table. |

## Cross-Artifact Portability Stability Check (n93-n97)

Substudy 1 of the Route 2 Stability and Replication Check. Tests whether cross-artifact conclusions survive panel perturbation.

| Artifact | Role |
|----------|------|
| `notes/n93_cross_artifact_stability_original_panel.md` | Stage A: original panel freeze (9 cases, 6 claims). |
| `notes/n94_cross_artifact_stability_perturbed_panel.md` | Stage B: perturbed panel definition (4 substitutions). |
| `notes/n95_cross_artifact_stability_rerun.md` | Stage C: rerun findings (B2 panel-sensitive, A1/A2/C1 stable). |
| `notes/n96_cross_artifact_stability_verdicts.md` | Stage D: claim-by-claim stability verdicts. |
| `notes/n97_cross_artifact_stability_memo.md` | Stage E: stability memo (final synthesis). |
| `../docs/strategy/cross_artifact_stability_summary.md` | Strategy-level summary of stability results. |
| `results/route2_stability/cross_artifact/original_panel_snapshot.json` | Original panel snapshot (JSON). |
| `results/route2_stability/cross_artifact/original_claims_snapshot.json` | Original claims snapshot (JSON). |
| `results/route2_stability/cross_artifact/perturbed_panel_table.json` | Perturbed panel table (JSON). |
| `results/route2_stability/cross_artifact/perturbed_panel_table.md` | Perturbed panel comparison (markdown). |
| `results/route2_stability/cross_artifact/panel_diff_table.md` | Panel diff table (markdown). |
| `results/route2_stability/cross_artifact/perturbed_invariant_signal_matrix.json` | Invariant signal audit on perturbed panel (JSON). |
| `results/route2_stability/cross_artifact/perturbed_local_signal_table.json` | Local signal audit on perturbed panel (JSON). |
| `results/route2_stability/cross_artifact/perturbed_signal_summary.md` | Signal summary (markdown). |
| `results/route2_stability/cross_artifact/stability_verdicts.json` | Stability verdicts (JSON). |
| `results/route2_stability/cross_artifact/stability_verdicts.md` | Stability verdicts (markdown). |
| `results/route2_stability/cross_artifact/stability_summary.json` | Optional compact stability summary (JSON). |

## Aggregation-Sensitive Compatibility Stability Check (n98-n102)

Substudy 2 of the Route 2 Stability and Replication Check. Tests whether aggregation-sensitive conclusions survive local panel perturbation.

| Artifact | Role |
|----------|------|
| `notes/n98_aggregation_stability_original_panel.md` | Stage A: original aggregation panel and claims freeze. |
| `notes/n99_aggregation_stability_perturbed_panel.md` | Stage B: perturbed panel definition (3 substitutions). |
| `notes/n100_aggregation_stability_rerun.md` | Stage C: rerun findings on perturbed panel. |
| `notes/n101_aggregation_stability_verdicts.md` | Stage D: claim-by-claim stability verdicts. |
| `notes/n102_aggregation_stability_memo.md` | Stage E: stability memo and Route 2 implications. |
| `../docs/strategy/aggregation_stability_summary.md` | Strategy-level summary of stability outcomes. |
| `results/route2_stability/aggregation/original_panel_snapshot.json` | Original panel snapshot (JSON). |
| `results/route2_stability/aggregation/original_claims_snapshot.json` | Original claims snapshot (JSON). |
| `results/route2_stability/aggregation/perturbed_panel_table.json` | Perturbed panel table (JSON). |
| `results/route2_stability/aggregation/perturbed_panel_table.md` | Perturbed panel table (markdown). |
| `results/route2_stability/aggregation/panel_diff_table.md` | Panel diff table (markdown). |
| `results/route2_stability/aggregation/perturbed_aggregation_comparison.json` | Perturbed aggregation comparison (JSON). |
| `results/route2_stability/aggregation/perturbed_aggregation_comparison.md` | Perturbed aggregation comparison (markdown). |
| `results/route2_stability/aggregation/stability_verdicts.json` | Stability verdicts (JSON). |
| `results/route2_stability/aggregation/stability_verdicts.md` | Stability verdicts (markdown). |
| `results/route2_stability/aggregation/stability_summary.json` | Optional compact stability summary (JSON). |
| `figures/aggregation_stability_comparison.svg` | Original-vs-perturbed comparison figure. |

## Aggregation Mixed-Evidence Triage Perturbation (n103-n107)

Targeted soft-middle stress test on triage-weighted mixed-evidence and same-family optional cases.

| Artifact | Role |
|----------|------|
| `notes/n103_aggregation_mixed_evidence_baseline.md` | Stage A: baseline claim freeze before mixed-evidence perturbation. |
| `notes/n104_aggregation_mixed_evidence_panel.md` | Stage B: mixed-evidence panel definition (8 cases, soft-middle weighted). |
| `notes/n105_aggregation_mixed_evidence_rerun.md` | Stage C: aggregation rerun on mixed-evidence panel. |
| `notes/n106_aggregation_mixed_evidence_interpretation.md` | Stage D: soft-middle interpretation verdicts. |
| `notes/n107_aggregation_mixed_evidence_triage_memo.md` | Stage E: triage memo and Route 2 language guidance. |
| `../docs/strategy/aggregation_mixed_evidence_summary.md` | Strategy-level summary for stable Route 2 packet language. |
| `results/route2_stability/aggregation_mixed_evidence/baseline_claims_snapshot.json` | Baseline claims snapshot (JSON). |
| `results/route2_stability/aggregation_mixed_evidence/panel_table.json` | Mixed-evidence panel table (JSON). |
| `results/route2_stability/aggregation_mixed_evidence/panel_table.md` | Mixed-evidence panel table (markdown). |
| `results/route2_stability/aggregation_mixed_evidence/panel_role_table.md` | Panel role table (markdown). |
| `results/route2_stability/aggregation_mixed_evidence/aggregation_comparison.json` | Aggregation comparison (JSON). |
| `results/route2_stability/aggregation_mixed_evidence/aggregation_comparison.md` | Aggregation comparison (markdown). |
| `results/route2_stability/aggregation_mixed_evidence/soft_middle_verdicts.json` | Soft-middle verdicts (JSON). |
| `results/route2_stability/aggregation_mixed_evidence/soft_middle_verdicts.md` | Soft-middle verdicts (markdown). |
| `results/route2_stability/aggregation_mixed_evidence/summary.json` | Optional compact summary (JSON). |
| `figures/aggregation_mixed_evidence_matrix.svg` | Mixed-evidence aggregation matrix figure. |

## Route 2 Claims Stability Ladder (n108-n112)

Confidence-calibration synthesis pass for Route 2 claims (stable / moderately stable / thin / local-only).

| Artifact | Role |
|----------|------|
| `notes/n108_route2_claims_inventory.md` | Stage A: fixed Route 2 claims inventory (20 claims). |
| `notes/n109_route2_claim_evidence_map.md` | Stage B: explicit evidence mapping for each claim. |
| `notes/n110_route2_claim_dimension_scoring.md` | Stage C: five-dimension scoring rationale and calibration notes. |
| `notes/n111_route2_claims_stability_ladder.md` | Stage D: final ladder assignment and status distribution. |
| `notes/n112_route2_claims_ladder_implications.md` | Stage E: communication and product-guardrail implications. |
| `../docs/strategy/route2_claims_ladder_summary.md` | Strategy-level summary for public/product/internal wording. |
| `results/route2_claims_ladder/claims_inventory.json` | Claims inventory (JSON). |
| `results/route2_claims_ladder/claims_inventory.md` | Claims inventory (markdown). |
| `results/route2_claims_ladder/claim_evidence_map.json` | Per-claim evidence source map (JSON). |
| `results/route2_claims_ladder/claim_evidence_map.md` | Condensed evidence map (markdown). |
| `results/route2_claims_ladder/claim_scoring.json` | Five-dimension scoring table (JSON). |
| `results/route2_claims_ladder/claim_scoring.md` | Five-dimension scoring table (markdown). |
| `results/route2_claims_ladder/stability_ladder.json` | Final ladder statuses and implications (JSON). |
| `results/route2_claims_ladder/stability_ladder.md` | Final ladder statuses (markdown). |
| `results/route2_claims_ladder/implications_summary.json` | Optional communication summary (JSON). |
| `figures/route2_claims_ladder.svg` | Ladder overview figure. |

## Panels, studies, and templates

| File | Role |
|------|------|
| `studies/s01_catastrophic_anchor_replication.md` | Founding study. |
| `studies/TEMPLATE.md` | Study template. |
| `panels/p01_catastrophic_anchors.md` | Canonical anchor panel. |
| `panels/TEMPLATE.md` | Panel template. |

## Product-side examples

Canonical examples for onboarding practitioners live in `../examples/` and `../docs/example-gallery.md`. Key directories:

| Directory | What it contains |
|-----------|-----------------|
| `../examples/inventory_preflight_same_task_control/` | Runnable same-task control preflight with real artifacts. |
| `../examples/inventory_preflight_mixed_task/` | Runnable mixed-task preflight (15 → 2 reduction). |
| `../examples/inventories/` | 5 fixture inventories for testing neighborhood behavior. |
| `../examples/qa/` | Canonical adapter QA artifacts (eligible, uncertain, flagged_weak). |
| `../examples/reports/` | Canonical merge report scenarios (safe, high-risk, strict-blocked). |
| `../docs/example-gallery.md` | 6 canonical scenarios with key outputs and what each demonstrates. |
