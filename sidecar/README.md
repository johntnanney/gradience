# Gradience Sidecar — Index

## What this is

A structured research directory for Gradience compatibility science. It started as an investigation of catastrophic cross-task interference in LoRA adapter merging, and now includes bounded Route 2 generalization work across artifact classes and decision scenarios (merge, routing, triage).

Core Gradience reliably detects the same-task / cross-task **boundary**. The sidecar still asks what happens *within* the cross-task regime, while also documenting where broader workflows are now validated and where they remain explicitly bounded.

---

## Entry points

If you are new to this repo, or returning after a break, start with the documents below. Everything else in this README is the detailed per-program archive — useful for tracing how findings were produced, but not required for understanding the current state.

| Entry point | Document | What you get |
|-------------|----------|-------------|
| **Start here** | [`notes/n69_settled_open_next.md`](notes/n69_settled_open_next.md) | What is established, what is unresolved, what comes next. The state-of-project dashboard in prose. |
| **Mechanism-ladder synthesis** | [`notes/n67_where_the_research_stands.md`](notes/n67_where_the_research_stands.md) | The mechanism-ladder account: commensurability, V-module pathology, conjunctive failure, behavioral signatures. |
| **Route 2 synthesis** | [`notes/n93_route2_synthesis.md`](notes/n93_route2_synthesis.md) | The broadened-compatibility account: decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge. Companion to n67. |
| **Core product** | [`../docs/product-validation.md`](../docs/product-validation.md) | What the field trials proved: evidence gate lesson, retained-vs-control outcomes, near-miss confirmation, product strengths and limitations. Gradience's own empirical identity. |
| **Field trial validation** | [`../field_trials/`](../field_trials/) and [`notes/n51_cpu_field_trial_plan.md`](notes/n51_cpu_field_trial_plan.md) | 5 inventories, 53+ pairs, 3 backbones. Raw data and per-inventory field notes. |
| **Ruled out** | [`notes/n68_ruled_out_mechanisms.md`](notes/n68_ruled_out_mechanisms.md) | Ten primary eliminations, five ancillary. What was tested, what failed, and why each elimination matters. The epistemic discipline of the program. |
| **GPU return plan** | [`notes/n07_deberta_adjudication_protocol.md`](notes/n07_deberta_adjudication_protocol.md) | Complete pre-registered protocol. 5 predictions, decision tree, ~3h compute budget. Executable as-is. |
| **Research packet** | [`packet/00_packet_index.md`](packet/00_packet_index.md) | Self-contained bundle: synthesis + product validation + ruled-out + evidence table + figures + GPU re-entry note. The thing you'd hand to a collaborator. |
| **Route 2 packet** | [`packet/route2/00_route2_packet_index.md`](packet/route2/00_route2_packet_index.md) | Self-contained Route 2 bundle: substrate generalization, four research programs, checkpoint triage alpha, key figures. For someone who wants the broadened-compatibility story without the mechanism ladder. |
| **Demo paths** | [`../docs/demo-paths.md`](../docs/demo-paths.md) | Three guided tours: stable product (Path A), broadened Route 2 (Path B), research program (Path C). Pick one based on what you want to understand. |
| **Ring 1 PEFT generalization** | [`../docs/strategy/ring1_peft_generalization_results.md`](../docs/strategy/ring1_peft_generalization_results.md) | Artifact-class generality: LoHa through the full pipeline via thin shim. What generalized, what stayed LoRA-specific. |
| **Ring 2 checkpoint-delta generalization** | [`../docs/design/ring2_stage_d_assessment_memo.md`](../docs/design/ring2_stage_d_assessment_memo.md) | Representation-path generality: full checkpoint deltas via summary-based reuse. What survived, what changed, what remains out of scope. |
| **Route 2 scope checkpoint** | [`../docs/strategy/broadened_substrate_scope.md`](../docs/strategy/broadened_substrate_scope.md) | Current bounded architecture/product scope: substrate broadening, checkpoint triage status, decision-dependent implications, and use-case pull criteria. |
| **Checkpoint triage alpha workflow** | [`../docs/examples/checkpoint-triage-alpha-workflow.md`](../docs/examples/checkpoint-triage-alpha-workflow.md) | First polished broadened workflow package (canonical T02 + clean HTML bundle + explicit scope contract). |
| **Cross-artifact stability** | [`../docs/strategy/cross_artifact_stability_summary.md`](../docs/strategy/cross_artifact_stability_summary.md) | Do the cross-artifact conclusions survive panel perturbation? 3 stable, 1 moderately stable, 1 panel-sensitive, 1 inconclusive. |
| **Aggregation stability** | [`../docs/strategy/aggregation_stability_summary.md`](../docs/strategy/aggregation_stability_summary.md) | Do aggregation-sensitive conclusions survive panel perturbation? 5 stable, 2 moderately stable in Substudy 2. |
| **Aggregation mixed-evidence triage** | [`../docs/strategy/aggregation_mixed_evidence_summary.md`](../docs/strategy/aggregation_mixed_evidence_summary.md) | Soft-middle stress pass: QA-dominant coherence held; same-family optional stayed review-like with guardrails. |
| **Route 2 claims ladder** | [`../docs/strategy/route2_claims_ladder_summary.md`](../docs/strategy/route2_claims_ladder_summary.md) | Confidence-calibrated synthesis of Route 2 claims: stable vs moderately stable vs thin vs local-only. |
| **Artifact index** | [`ARTIFACTS.md`](ARTIFACTS.md) | Every figure, script, JSON result, HTML report, and field trial bundle in one navigable page. |

### Reading order

**Product-first readers** (practitioners, potential users, "what does this do for me?"): Start with the executive summary (n50), then field trial validation, then the settled/open/next dashboard (n69). If you want to understand *why* the tool works and where it breaks, continue to n67 §§1–3 and §6.

**Research-first readers** (collaborators, reviewers, "what did you find and how solid is it?"): Start with the settled/open/next dashboard (n69), then the full synthesis (n67), then the ruled-out packet (n68). The research packet (`packet/`) gives you the complete picture with figures. For any specific finding, trace it back through the per-program sections below — each program has its own panel definition, protocol, and findings note.

---

## Current state

Route 2 initial implementation checkpoint is now complete in bounded form: broadened substrate documentation, checkpoint triage stabilization artifacts, a polished checkpoint-triage alpha workflow package, decision-dependent compatibility consolidation, completed cross-artifact portability clarification (n76-n80, anchored by n75), a local robustness/stability pass on that portability line (n93-n97), a completed aggregation-sensitive stability check (n98-n102), a targeted mixed-evidence triage perturbation pass (n103-n107), and a calibrated claims-stability ladder synthesis (n108-n112).

**Instability** — the variability of a pair's **severity** across seeds and backbones — is the sidecar's central organizing concept and has been formalized as a **research program** (n06). The key finding: severity rankings reverse across backbones, but instability rankings do not. Two pairs cluster as highly unstable (instability > 0.7) and four as stable (< 0.3), with a clean gap between the clusters. The working hypothesis is **thresholded subspace interference**: catastrophic outcomes require specific seed-dependent geometric conditions to trigger.

## What is currently being tested

**Study S01** (Catastrophic Anchor Replication) is the founding study. The two-backbone phase (DistilBERT + RoBERTa) is complete. The DeBERTa-v3 leg is the next empirical step and requires GPU compute not currently available. A complete pre-registered adjudication protocol is in **n07**.

The DeBERTa leg is an adjudication test: *do the same pairs remain the most unstable, regardless of which pair is catastrophic on the new backbone?* Five pre-registered predictions with operational success criteria are defined in n07 §3–5.

## Sidecar B status (Output-Space Compatibility)

Sidecar B Stage B (readout geometry and margin audit) is complete. **Verdict: mixed signal with a critical falsifier.** The same pair (QNLI × MRPC) on RoBERTa has nearly identical readout geometry to the catastrophic CA-01 on DistilBERT, yet produces only 1.7% degradation. Readout incompatibility is a **necessary background condition**, not a discriminative cause. Compatible readout reliably predicts safety (3/3), but incompatible readout does not predict catastrophe (4/7 incompatible cases are catastrophic/moderate, 2 mild, 1 safe). The mechanism is conjunctive: **V-module pathology × readout incompatibility → catastrophic failure**. Either factor alone is insufficient.

Deliverables: n30 (panel definition), n31 (protocol), n32 (findings), n33 (conjunctive synthesis), 3 JSON result files, 4 figures.

## Seed-Contingent Readout-Axis Selection status

Stages A–B complete. **Key finding: readout orthogonality is bimodal, common even in same-task seed pairs, and decoupled from upstream V-module geometry.** 5 of 14 same-task seed pairs show orthogonal decision axes (cos ≈ 0), yet all merge safely (Δ ≤ 2.2%). The distribution is sharply bimodal — either ~0 or ~1, no intermediate values. QNLI always produces orthogonal readout, RTE/SST-2 always aligned, MRPC varies by backbone. Training convergence also matters: Strong QNLI is orthogonal while Medium/Weak are aligned. The two conditions for catastrophe (V-module pathology + readout incompatibility) are independently determined by different mechanisms.

Deliverables: n34 (panel), n35 (protocol), n36 (findings), 4 JSON result files, 3 figures.

## Attractor Mapping Lab status

Stages A–B complete. **Key finding: the attractor landscape is structured, discrete, and mappable.** 10 task families classified: 6 single-attractor, 3 multi-attractor, 1 backbone-contingent (MRPC). All families merge safely regardless of attractor type. Task identity is the primary determinant of attractor structure, modulated by backbone architecture, training depth, and training distribution. Multi-attractor ≠ fragile. All six hypotheses from the spec confirmed. Cross-domain readout alignment tracks within-family attractor structure: SST-2 (domain, multi-attractor) is orthogonal to Yelp/Amazon (single-attractor); Yelp and Amazon are aligned with each other.

Deliverables: n39 (panel), n40 (protocol), n41 (findings), 1 analysis script, 4 JSON result files, 4 figures.

## Attractor Mechanism Determinants status

Stages A–C complete. **Key finding: a structured determinant hierarchy governs mechanism choice — task identity (primary) → backbone architecture (secondary) → training convergence (tertiary) → domain structure (weak).** All 5 multi-attractor conditions classified: 4 rotational degeneracy (all DistilBERT), 1 feature-set switching (QNLI/RoBERTa). Mechanism and backbone are perfectly confounded in the current panel — all degeneracy on DistilBERT, all switching on RoBERTa. Training depth modulates attractor count but not mechanism class. The QNLI cross-backbone contrast is the strongest evidence: same task, same seeds, different mechanism by backbone alone. Commensurability refined to version 3: readout incompatibility now decomposes by mechanism class, with different failure semantics (incoherent confidence for degeneracy, systematic misclassification for switching).

Deliverables: n46 (classification), n47 (protocol), n48 (findings), n49 (synthesis), 3 JSON result files, 1 JSON context table, 3 figures.

## Phase 3 status (CPU-only sidecar deepening)

Phase 3 work is complete, including the within-layer collision program (Stages A–B), the per-module geometry program (Stages A–B), and the V-module head-level program (Stages A–B). The sidecar now contains 47 notes, 1 study, 1 panel definition, 12 analysis scripts, 58 structured data outputs, and 36 figures. Deliverables:

- **Project F (Instability Program Consolidation):** n06 (program statement), n07 (DeBERTa adjudication protocol), extended case table with per-pair predictions
- **Project H (Catastrophic Anchor Dossiers):** n08 (CA-01 dossier), n09 (CA-02 dossier), n10 (dossier synthesis), dossier template
- **Project I (Backbone-Local Interpretation):** n11 (DistilBERT vs. RoBERTa local regularities)
- **Project G (Local Artifact Mining):** n12 (mining inventory), n13–n15 (per-layer analysis — **MIXED**: collision pattern found)
- **Within-Layer Collision Program:** n16 (collision subset definition), n17–n18 (within-layer geometry — **MIXED/NEGATIVE**: backbone confound, seed sensitivity not explained, aggregate subspace ruled out as threshold)
- **Per-Module Geometry Program:** n19 (per-module subset), n20–n21 (per-module geometry — **POSITIVE**: V-module dimensionality ratio achieves d=3.36 with zero range overlap, first clean separation in the backbone-controlled comparison)
- **V-Module Head-Level Program:** n22 (head panel), n23–n24 (head-level geometry — **MIXED-POSITIVE**: CA-01 seed sensitivity resolved at head level, 7 heads show |Δ_DR| ≥ 0.15, but module-level d=3.36 remains the strongest group discriminator)
- **Sidecar B — Output-Space Compatibility:** n30 (panel definition), n31 (protocol), n32 (findings — **MIXED**: readout alignment predicts safety, but readout incompatibility does not predict catastrophe; readout is a gate, not an amplifier)

The only blocked work is training 8 new DeBERTa-v3 adapters and evaluating 28 merge pairs.

## Key documents

### Program and concept

| Document | What it is |
|----------|------------|
| `notes/n06_instability_program_statement.md` | **Program statement.** Formalizes instability as a research program with commitments, predictions, falsification conditions. |
| `notes/n05_instability_as_working_concept.md` | **Concept argument.** Why instability, not severity, is the right variable. |
| `notes/n07_deberta_adjudication_protocol.md` | **Adjudication protocol.** Complete pre-registered protocol for the DeBERTa leg. |
| `studies/s01_catastrophic_anchor_replication.md` | **Founding study.** Two-backbone results, DeBERTa predictions, success criteria. |

### Evidence and interpretation

| Document | What it is |
|----------|------------|
| `notes/n04_evidence_atlas.md` | Full map of what is known, open, and ruled out. Three-level framework. |
| `notes/n11_backbone_local_interpretation.md` | Per-backbone local regularities: what transfers and what does not. |
| `notes/n12_local_artifact_mining_inventory.md` | Mining inventory: what data exists, preliminary findings, priority analysis. |
| `notes/n15_per_layer_findings.md` | **Per-layer structural findings.** MIXED outcome: collision pattern found (catastrophic pairs show higher alignment, lower divergence), but groups overlap. |
| `notes/n16_collision_subset_definition.md` | **Collision subset.** Classifies all 20 pair×backbone cases. Collision is a risk amplifier, neither necessary nor sufficient. |
| `notes/n18_within_layer_geometry_findings.md` | **Within-layer geometry findings.** MIXED/NEGATIVE: backbone confound dominates, seed sensitivity not explained, aggregate subspace ruled out as threshold. |
| `notes/n19_per_module_subset_definition.md` | **Per-module subset.** Defines the per-module contrast panel and module correspondence across backbones. |
| `notes/n20_per_module_geometry_protocol.md` | **Per-module protocol.** Same 4 metrics applied separately per Q/K/V/O module. |
| `notes/n21_per_module_geometry_findings.md` | **Per-module findings.** POSITIVE: V-module dimensionality ratio cleanly separates catastrophic from safe collision (d=3.36, zero overlap). CA-02 seed sensitivity now partly explained. |
| `notes/n22_v_head_panel_definition.md` | **Head-level V panel.** Defines the per-head contrast panel, head architecture, critical V layers, and preliminary reconnaissance. |
| `notes/n23_v_head_geometry_protocol.md` | **Head-level V protocol.** Per-head metrics, summary descriptors, decision criteria for the head-level pilot. |
| `notes/n24_v_head_geometry_findings.md` | **Head-level V findings.** MIXED-POSITIVE: CA-01 seed sensitivity resolved (7 heads with |Δ_DR| ≥ 0.15, max 0.229), but module-level d=3.36 remains strongest discriminator. Cancellation mechanism identified. |
| `notes/n25_multiscale_mechanism_synthesis.md` | **Multiscale synthesis.** Integrates module-level (discrimination), head-level (seed sensitivity), and downstream (amplification) into a four-link causal chain. States what each scale explains and what remains open. |

### Sidecar B — Output-Space Compatibility

| Document | What it is |
|----------|------------|
| `notes/n30_output_space_panel_definition.md` | **Panel definition.** 11-case panel across 5 groups, classifier head weights confirmed in all adapters. |
| `notes/n31_output_space_protocol.md` | **Protocol.** 5 metrics (readout alignment, pre-classifier alignment, merged readout geometry, decision boundary angle, LoRA-readout coupling), 5 contrasts, interpretation rules. |
| `notes/n32_output_space_findings.md` | **Findings.** MIXED signal. Readout incompatibility is necessary but not sufficient for catastrophe. The readout layer is a gate, not an amplifier. Conjunctive model: V-module pathology × readout incompatibility → catastrophic failure. |
| `notes/n33_conjunctive_mechanism_synthesis.md` | **Conjunctive synthesis.** Integrates Sidecar A (V-module pathology, head-level modulation) and Sidecar B (readout gating) into the complete mechanism model. Supersedes n25 Rung 3. Identifies seed-contingent readout as the best remaining CPU-feasible question. |

### Seed-Contingent Readout-Axis Selection

| Document | What it is |
|----------|------------|
| `notes/n34_seed_readout_panel_definition.md` | **Panel definition.** 14 same-task seed pairs + 3 adjacent-task pairs across core, domain-shift, and source-strength studies. |
| `notes/n35_upstream_readout_coupling_protocol.md` | **Protocol.** Readout, upstream V-module, and coupling metrics. |
| `notes/n36_upstream_readout_coupling_findings.md` | **Findings.** MIXED (decoupled). 5/14 same-task seed pairs show orthogonal readout yet merge safely. Bimodal distribution, task-specific attractor structure, decoupled from upstream geometry. |
| `notes/n37_conjunctive_model_update.md` | **Conjunctive model update.** Integrates same-task evidence into mechanism model. Confirms attractor hypothesis, establishes independence of conditions. Current best statement of the complete model. Major negative-positive result. |
| `notes/n38_ruled_out.md` | **Ruled-out summary.** Eight hypotheses tested and rejected, with evidence and replacements. The sidecar's epistemic discipline in one page. |

### Attractor Mapping Lab

| Document | What it is |
|----------|------------|
| `notes/n39_attractor_panel_definition.md` | **Panel definition.** 14 family×backbone entries across 4 groups (core, domain, backbone contrast, convergence). All success criteria met. |
| `notes/n40_family_readout_audit_protocol.md` | **Protocol.** 4 metrics, 5 contrasts, deterministic classification rule. |
| `notes/n41_family_readout_audit_findings.md` | **Findings.** POSITIVE: attractor landscape is structured and discrete. 6 single-attractor, 3 multi-attractor, 1 backbone-contingent. All safe. |
| `notes/n42_readout_solution_topology.md` | **Synthesis.** Topology of readout solution spaces: three classes, hierarchy of influence, why descriptive not predictive, how it sharpens commensurability. |
| `notes/n43_attractor_origin_research_program.md` | **Research program.** Why do some tasks admit multiple readout attractors? Feature plurality hypothesis, 3-stage CPU-feasible design, falsification conditions. |
| `notes/n44_decision_axis_analysis_findings.md` | **Findings.** MIXED-POSITIVE: simple feature plurality partially falsified; two mechanisms identified — rotational degeneracy (DistilBERT) and feature-set switching (RoBERTa). QNLI/rb/s7 aligns with RTE (cos=0.86). |
| `notes/n45_conjunctive_model_with_mechanisms.md` | **Synthesis.** Current best statement of the complete mechanism model. Rung 3 bifurcates into 3a (rotational degeneracy) and 3b (feature-set switching). Supersedes n37 §4 and n33 Rung 3. |

### Attractor Mechanism Determinants

| Document | What it is |
|----------|------------|
| `notes/n46_attractor_mechanism_classification.md` | **Classification audit.** All 5 multi-attractor conditions classified: 4 rotational degeneracy (DistilBERT), 1 feature-set switching (QNLI/RoBERTa). Operational classification rules. No unresolved cases. |
| `notes/n47_attractor_mechanism_determinants_protocol.md` | **Protocol.** 5 candidate determinants, 4 contrasts, interpretation framework. |
| `notes/n48_attractor_mechanism_determinants_findings.md` | **Findings.** MIXED-POSITIVE: structured hierarchy (task → backbone → convergence → domain) with critical backbone confound. Causal model proposed. |
| `notes/n49_mechanism_and_commensurability_synthesis.md` | **Synthesis.** Three kinds of benign diversity. Commensurability v3: readout decomposes by mechanism. Different failure semantics per mechanism. Supersedes n45 §commensurability discussion. |

### Catastrophic anchor dossiers

| Document | What it is |
|----------|------------|
| `notes/n08_anchor_dossier_CA01_qnli_mrpc_distilbert.md` | Full dossier: QNLI×MRPC on DistilBERT (CA-01). |
| `notes/n09_anchor_dossier_CA02_qnli_sst2_roberta.md` | Full dossier: QNLI×SST-2 on RoBERTa (CA-02). |
| `notes/n10_anchor_dossier_synthesis.md` | Cross-dossier patterns and what they reveal. |

### Earlier notes and reference

| Document | What it is |
|----------|------------|
| `notes/n13_artifact_inventory.md` | Per-layer artifact inventory: all 16 adapters confirmed, contrast panel defined. |
| `notes/n14_per_layer_protocol.md` | Per-layer comparison protocol: 4 metrics, computation procedure, interpretation framework. |
| `notes/n03_instability_vs_severity.md` | Technical argument for instability over severity. Superseded by n05/n06. |
| `notes/n02_catastrophic_anchor_dossiers.md` | Original compact dossiers. Superseded by n08/n09. |
| `notes/n01_anchor_replication_preliminary.md` | First interpretation note from the two-backbone phase. |
| `panels/p01_catastrophic_anchors.md` | Canonical anchor panel definition with severity thresholds and rerun protocol. |
| `results/s01/instability_case_table.md` | Compact 12-row table: all pairs × both backbones, ranked by instability. |
| `results/s01/instability_case_table_extended.md` | Extended table with DeBERTa predictions and confidence ratings. |
| `glossary.md` | **Frozen definitions** for canonical terms. |
| `cpu_only_roadmap.md` | Medium-term roadmap: 12 projects across core and sidecar, sequenced in 4 phases. |
| `strategy_memo.md` | Founding document. Research programs, workstreams, promotion rules. |

## Canonical terms

Eight terms are frozen (see `glossary.md` for full definitions):

- **boundary** — the same-task / cross-task classification (core, settled)
- **severity** — magnitude of degradation in one condition (backbone-local, not portable)
- **instability** — variability of severity across conditions (the sidecar's working concept)
- **catastrophic anchor** — a (task pair × backbone) combination that produces > 15% worst-case degradation
- **portable descriptor** — a merge-risk signal that generalizes across backbones
- **thresholded subspace interference** — the hypothesis that catastrophic outcomes require specific geometric conditions to trigger
- **V-module dimensionality mismatch** — the strongest current correlate of catastrophic threshold within the collision regime; pending DeBERTa confirmation
- **multiscale mechanism ladder** — the three-rung explanatory hierarchy: module-level risk (Rung 1), head-level modulation (Rung 2), readout gating (Rung 3, resolved — gate not amplifier, bimodal, decoupled from upstream)
- **readout attractor** — a stable decision-axis direction in representation space; some tasks have one (RTE, SST-2), others have multiple orthogonal attractors (QNLI, MRPC on DistilBERT)
- **commensurability** — the conjunction of upstream V-module compatibility and readout compatibility; the integrated concept for whether two adapters can be safely merged (version 3: readout condition decomposes by mechanism class — angular for degeneracy, structural for switching)
- **mechanism determinant hierarchy** — the ordering of factors governing mechanism choice: task identity (primary, determines attractor count) → backbone architecture (secondary, selects mechanism) → training convergence (tertiary, gates attractor count) → domain structure (weak)

### Output Example Semantics Program

| Document | What it is |
|----------|------------|
| `notes/n59_output_example_panel_definition.md` | **Panel definition.** 8-case panel across 5 classes (safe, fragile, control, near-miss, anchor), 2 backbones, 3 task families. |
| `notes/n60_example_behavior_protocol.md` | **Stage A protocol.** Data collection, per-example classification (8 raw categories), 5 metrics. |
| `notes/n61_example_behavior_findings.md` | **Stage A findings.** 6 findings. Joint-source breakage separates classes (threshold ~5%). Neither-source rate is binary discriminator (<2% vs >14%). Fragile fails with confidence collapse; safe fails with high confidence. |
| `notes/n62_failure_taxonomy_protocol.md` | **Stage B protocol.** Bottom-up taxonomy derivation from n61 categories. |
| `notes/n63_failure_taxonomy_findings.md` | **Stage B findings.** Final 5-category taxonomy (A, C, D, E, X). B absorbed into D. D (neither-source) is 12–14% in fragile/control, <2% in safe/near-miss. |
| `notes/n64_example_dossier_safe_vs_fragile.md` | **Stage C dossier.** Safe vs fragile at example level. Walk-throughs of preservation, neither-source, confidence collapse, better-source loss. Maps to mechanism ladder. |
| `notes/n65_example_dossier_near_miss.md` | **Stage C dossier.** Near-miss cases. Threshold model confirmed — near-miss is behaviorally safe, not a fragile precursor. |
| `notes/n66_behavior_mechanism_bridge.md` | **Stage D synthesis.** Bridges example-level behavior to mechanism ladder. Double dissociation between fragile (confidence collapse) and control (high-confidence wrong). Conjunctive model confirmed at example level. |

### Synthesis

| Document | What it is |
|----------|------------|
| `notes/n67_where_the_research_stands.md` | **Comprehensive synthesis.** The single current-best account of the sidecar's theoretical picture. Supersedes n51 and n25. Commensurability, instability, V-module pathology, head-level modulation, readout attractors, conjunctive failure, behavioral signatures. |
| `notes/n69_settled_open_next.md` | **State-of-project dashboard.** Standalone settled/open/next index. Updated independently of n67. The quick-reference version of what is established, unresolved, and prioritized. |
| `notes/n50_executive_research_summary.md` | **Executive summary.** Project identity, central claim, current product role, research findings, practitioner value, north star. |

### Decision-Dependent Compatibility Program

| Document | What it is |
|----------|------------|
| `notes/n70_decision_dependent_panel_definition.md` | **Stage A panel.** Shared case panel across merge, routing, and triage with explicit overlap cases. |
| `notes/n71_shared_measurement_divergent_policy_audit.md` | **Stage B audit.** Layer-by-layer shared vs scenario-specific stack; first divergence point analysis. |
| `notes/n72_aggregation_sensitive_compatibility.md` | **Stage C analysis.** Worst-case vs distributional vs QA-gate aggregation on shared structural inputs. |
| `notes/n73_decision_profile_taxonomy.md` | **Stage D taxonomy.** Six bounded decision-dependent compatibility profiles. |
| `notes/n74_decision_semantics_bridge.md` | **Stage E bridge.** Maps profiles to behavioral manifestations and evidence-strength caveats. |
| `notes/n75_cross_artifact_compatibility.md` | **Cross-artifact extension.** Initial shared-vs-specific comparison across LoRA, LoHa, and full checkpoint-delta paths. |
| `results/decision_dependent_compatibility/` | **Structured outputs.** Panel table, scenario stack matrix, aggregation comparison, profile table, semantics table, plus figure. |
| `results/cross_artifact_compatibility/` | **Structured outputs.** Shared-vs-specific signal table for the cross-artifact Route 2 pass. |

### Field Trial Plan and Results

| Document | What it is |
|----------|------------|
| `notes/n51_cpu_field_trial_plan.md` | **CPU-only field trial plan.** 8–10 inventory workflow validation program. |

### Ring 1 — PEFT Generalization

| Document | What it is |
|----------|------------|
| [`../docs/strategy/ring1_peft_generalization_results.md`](../docs/strategy/ring1_peft_generalization_results.md) | **Assessment memo.** What generalized (measurement, comparison, inventory), what needed a shim (extraction), what stayed LoRA-specific (merge execution). |
| [`../docs/design/peft_generalization_audit.md`](../docs/design/peft_generalization_audit.md) | **Design doc.** Substrate analysis, candidate selection (LoHa), shim architecture, success criteria. |
| [`../experiments/peft_ring1/`](../experiments/peft_ring1/) | **Experiments.** Trained adapters, shim, Stage B/C scripts, measurement results, inventory pilot, field note. |

### Ring 2 — Checkpoint-Delta Generalization

| Document | What it is |
|----------|------------|
| [`../docs/design/ring2_stage_a_checkpoint_delta_representation.md`](../docs/design/ring2_stage_a_checkpoint_delta_representation.md) | **Stage A design note.** Three candidate representations compared; Representation C (layerwise summary) selected for CPU feasibility and stability. |
| [`../docs/design/ring2_stage_b_representation_c_audit.md`](../docs/design/ring2_stage_b_representation_c_audit.md) | **Stage B design note.** Single-artifact audit and pairwise comparison on summary representation. Same-task vs cross-task separation confirmed. |
| [`../docs/design/ring2_stage_c_guardrail_triage.md`](../docs/design/ring2_stage_c_guardrail_triage.md) | **Stage C design note.** Inventory guardrail triage and run-bundle packaging from summary-based inputs. |
| [`../docs/design/ring2_stage_d_assessment_memo.md`](../docs/design/ring2_stage_d_assessment_memo.md) | **Stage D assessment.** Plain assessment: workflow survives, representation path differs, evidence and QA remain central, merge execution out of scope. |
| [`../experiments/ring2_checkpoint_delta/`](../experiments/ring2_checkpoint_delta/) | **Experiments.** 5 trained checkpoints, extraction scripts, Stage A/B/C harnesses, structured results. |

### Cross-Artifact Compatibility Research Program (n76-n80)

| Document | What it is |
|----------|------------|
| `notes/n76_cross_artifact_panel_definition.md` | **Stage A panel.** 9-case panel across LoRA, LoHa, and checkpoint delta. Coverage matrix and known gaps. |
| `notes/n77_cross_artifact_invariant_signal_audit.md` | **Stage B audit.** Five signal families tested for cross-artifact recurrence. 2 strong invariants, 2 moderate, 1 inconclusive. |
| `notes/n78_representation_local_signal_audit.md` | **Stage C audit.** Seven representation-local signals identified. V-module ratio is representation-locked. No structural metric is fully portable. |
| `notes/n79_cross_artifact_compatibility_framework.md` | **Stage D framework.** Three-layer model: artifact-invariant signals, representation-family features, decision-dependent interpretation. |
| `notes/n80_cross_artifact_product_relevance.md` | **Stage E filter.** Product relevance classification: 2 safe, 3 guarded, 3 research-only, 3 not stable enough. |
| [`../docs/strategy/cross_artifact_product_relevance_summary.md`](../docs/strategy/cross_artifact_product_relevance_summary.md) | **Product summary.** What transfers, what doesn't, product language guidance. |
| `results/cross_artifact_portability/` | **Structured outputs.** Panel table, invariant signal matrix, local signal table, framework table, product relevance filter. |

### Cross-Artifact Portability Stability Check (n93-n97)

| Document | What it is |
|----------|------------|
| `notes/n93_cross_artifact_stability_original_panel.md` | **Stage A.** Original panel freeze: 9 cases, 6 claims. |
| `notes/n94_cross_artifact_stability_perturbed_panel.md` | **Stage B.** Perturbed panel: 4 substitutions across LoRA and checkpoint delta. |
| `notes/n95_cross_artifact_stability_rerun.md` | **Stage C.** Rerun findings: A1/A2 stable, C1 locality stable, B2 panel-sensitive, D1 still inconclusive. |
| `notes/n96_cross_artifact_stability_verdicts.md` | **Stage D.** Claim-by-claim verdicts: 3 stable, 1 moderately stable, 1 panel-sensitive, 1 still inconclusive. |
| `notes/n97_cross_artifact_stability_memo.md` | **Stage E.** Final stability memo and Route 2 implications. |
| [`../docs/strategy/cross_artifact_stability_summary.md`](../docs/strategy/cross_artifact_stability_summary.md) | **Strategy summary.** Compact verdicts and implications for Route 2. |
| `results/route2_stability/cross_artifact/` | **Structured outputs.** Panel snapshots, perturbed signals, verdicts (8 JSON + 4 markdown). |

### Aggregation-Sensitive Compatibility Research Program (n81-n85)

| Document | What it is |
|----------|------------|
| `notes/n81_aggregation_panel_definition.md` | **Stage A panel.** 12-case panel with matched QA-regime pairs across merge, routing, and triage substrates. |
| `notes/n82_aggregation_family_audit.md` | **Stage B audit.** Four aggregation families formalized: worst-case, distributional, QA-dominant, QA-gated distributional. |
| `notes/n83_aggregation_comparison_analysis.md` | **Stage C comparison.** 12-case comparison matrix. 2 full agreement, 2 strong divergence, 8 partial. Key finding: aggregation is not presentation. |
| `notes/n84_aggregation_sensitive_pattern_taxonomy.md` | **Stage D taxonomy.** Five stable patterns predictable from QA regime and task relation. |
| `notes/n85_aggregation_sensitive_operational_implications.md` | **Stage E implications.** Operational guidance: 3 safe, 2 guarded, 2 research-only. |
| [`../docs/strategy/aggregation_sensitive_route2_summary.md`](../docs/strategy/aggregation_sensitive_route2_summary.md) | **Product summary.** Route 2 summary with safe/not-safe product language guidance. |
| `results/aggregation_sensitive_compatibility/` | **Structured outputs.** Panel table, family specs, comparison matrix, pattern taxonomy, operational implications. |

### Aggregation-Sensitive Stability Check (n98-n102)

| Document | What it is |
|----------|------------|
| `notes/n98_aggregation_stability_original_panel.md` | **Stage A.** Original aggregation panel and claim freeze for stability testing. |
| `notes/n99_aggregation_stability_perturbed_panel.md` | **Stage B.** Local panel perturbation (3 substitutions, one per scenario family). |
| `notes/n100_aggregation_stability_rerun.md` | **Stage C.** Perturbed rerun findings: aggregation seam and divergence patterns persist. |
| `notes/n101_aggregation_stability_verdicts.md` | **Stage D.** Claim-by-claim stability verdicts (5 stable, 2 moderately stable). |
| `notes/n102_aggregation_stability_memo.md` | **Stage E.** Stability memo and Route 2 implication framing. |
| [`../docs/strategy/aggregation_stability_summary.md`](../docs/strategy/aggregation_stability_summary.md) | **Strategy summary.** Compact stability outcomes and language guidance. |
| `results/route2_stability/aggregation/` | **Structured outputs.** Original snapshots, perturbed comparison, verdicts, stability summary. |
| `figures/aggregation_stability_comparison.svg` | **Figure.** Original vs perturbed agreement distribution with verdict roll-up. |

### Aggregation Mixed-Evidence Triage Perturbation (n103-n107)

| Document | What it is |
|----------|------------|
| `notes/n103_aggregation_mixed_evidence_baseline.md` | **Stage A.** Baseline freeze of stable vs guarded aggregation claims before the soft-middle stress test. |
| `notes/n104_aggregation_mixed_evidence_panel.md` | **Stage B.** 8-case mixed-evidence panel (anchors + review + same-family optional emphasis). |
| `notes/n105_aggregation_mixed_evidence_rerun.md` | **Stage C.** Aggregation rerun on soft-middle panel; outputs remained interpretable across families. |
| `notes/n106_aggregation_mixed_evidence_interpretation.md` | **Stage D.** Soft-middle interpretation verdicts (`coherent` / `coherent_with_guardrails`). |
| `notes/n107_aggregation_mixed_evidence_triage_memo.md` | **Stage E.** Memo for Route 2 triage language: structured middle, guarded thresholds. |
| [`../docs/strategy/aggregation_mixed_evidence_summary.md`](../docs/strategy/aggregation_mixed_evidence_summary.md) | **Strategy summary.** Compact guidance for review/optional language in mixed-evidence triage. |
| `results/route2_stability/aggregation_mixed_evidence/` | **Structured outputs.** Baseline snapshot, panel tables, aggregation comparison, soft-middle verdicts, summary JSON. |
| `figures/aggregation_mixed_evidence_matrix.svg` | **Figure.** Aggregation-family outputs across the mixed-evidence triage panel. |

### Route 2 Claims Stability Ladder (n108-n112)

| Document | What it is |
|----------|------------|
| `notes/n108_route2_claims_inventory.md` | **Stage A inventory.** Frozen 20-claim Route 2 set for confidence calibration. |
| `notes/n109_route2_claim_evidence_map.md` | **Stage B mapping.** Per-claim source map across notes, results, and strategy docs. |
| `notes/n110_route2_claim_dimension_scoring.md` | **Stage C scoring.** Five-dimension scoring (evidence, perturbation survival, coverage, grounding, product relevance). |
| `notes/n111_route2_claims_stability_ladder.md` | **Stage D ladder.** Final statuses (`stable`, `moderately_stable`, `thin`, `local_only`). |
| `notes/n112_route2_claims_ladder_implications.md` | **Stage E implications.** Public/product/internal language guidance from ladder status. |
| [`../docs/strategy/route2_claims_ladder_summary.md`](../docs/strategy/route2_claims_ladder_summary.md) | **Strategy summary.** Calibrated Route 2 wording guidance and guardrail framing. |
| `results/route2_claims_ladder/` | **Structured outputs.** Claims inventory, evidence map, scoring table, ladder statuses, implications summary. |
| `figures/route2_claims_ladder.svg` | **Figure.** One-page ladder visualization of claim-status distribution. |

### Behavioral Route 2 Bridge (n86-n92)

| Document | What it is |
|----------|------------|
| `notes/n86_behavioral_route2_panel_definition.md` | **Stage A panel.** 8-case panel reinterpreting n59-n66 cases through 5 Route 2 profiles. |
| `notes/n87_behavioral_route2_protocol.md` | **Stage B protocol.** 6-metric, 6-category behavioral comparison protocol adapted for Route 2. |
| `notes/n88_behavioral_route2_findings.md` | **Stage C findings.** Profile-to-behavior analysis. Three-tier behavioral model. H1-H6 results. |
| `notes/n89_behavioral_route2_dossier_optional_vs_fragile.md` | **Dossier 1.** Optional (NM-01) vs fragile (FR-01): threshold, not gradient. |
| `notes/n90_behavioral_route2_dossier_confusable_vs_separable.md` | **Dossier 2.** Confusable (NM-01) vs separable (CT-01): structural confusability ≠ behavioral confusion. |
| `notes/n91_behavioral_route2_dossier_qa_override.md` | **Dossier 3.** QA stasis (AN-01) vs structural collapse (FR-01): evidence gating is behaviorally justified. |
| `notes/n92_behavioral_route2_bridge.md` | **Stage E synthesis.** Bridge connecting behavioral findings to cross-artifact, decision-dependent, and aggregation-sensitive programs. |
| [`../docs/strategy/behavioral_route2_summary.md`](../docs/strategy/behavioral_route2_summary.md) | **Product summary.** What Route 2 profiles mean behaviorally. Safe/not-safe guidance. |
| `results/behavioral_route2_bridge/` | **Structured outputs.** Panel table, protocol schema, behavior summary, profile behavior table, bridge table. |
| `figures/behavioral_route2_profile_matrix.svg` | **Profile matrix figure.** Three-tier behavioral separation across 3 discriminating metrics. |

### Research Synthesis and Negative Results

| Document | What it is |
|----------|------------|
| `notes/n51_research_synthesis_memo.md` | **Research synthesis (prior).** Superseded by n67 for the overall picture, but retains the product-implications framing. |
| `notes/n68_ruled_out_mechanisms.md` | **Ruled-out mechanisms (definitive).** Supersedes n38 and n52. Ten primary eliminations, five ancillary. Portable severity, task-pair lookup, aggregate threshold, readout-as-risk, readout-as-amplifier, feature plurality as universal origin. What survived and the epistemic structure of the eliminations. |
| `notes/n52_ruled_out_mechanisms_packet.md` | **Ruled-out mechanisms (prior).** Superseded by n68. |
| `notes/n38_ruled_out.md` | **Ruled-out hypotheses (original).** Superseded by n68. |

## What should happen next

See [`notes/n69_settled_open_next.md`](notes/n69_settled_open_next.md) for the prioritized next-steps list. The short version: execute the DeBERTa adjudication protocol (n07, ~3h GPU), then follow the decision tree based on which predictions pass. The GPU re-entry note in [`packet/05_gpu_reentry.md`](packet/05_gpu_reentry.md) has the condensed checklist.
