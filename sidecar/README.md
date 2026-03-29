# Gradience Sidecar — Index

## What this is

A structured research directory for investigating catastrophic cross-task interference in LoRA adapter merging. This work is too exploratory for core Gradience but is empirically grounded and designed for eventual promotion if results warrant it.

Core Gradience reliably detects the same-task / cross-task **boundary**. The sidecar asks what happens *within* the cross-task regime — specifically, why some pairs fail catastrophically while others degrade predictably.

## Current state

**Instability** — the variability of a pair's **severity** across seeds and backbones — is the sidecar's central organizing concept and has been formalized as a **research program** (n06). The key finding: severity rankings reverse across backbones, but instability rankings do not. Two pairs cluster as highly unstable (instability > 0.7) and four as stable (< 0.3), with a clean gap between the clusters. The working hypothesis is **thresholded subspace interference**: catastrophic outcomes require specific seed-dependent geometric conditions to trigger.

## What is currently being tested

**Study S01** (Catastrophic Anchor Replication) is the founding study. The two-backbone phase (DistilBERT + RoBERTa) is complete. The DeBERTa-v3 leg is the next empirical step and requires GPU compute not currently available. A complete pre-registered adjudication protocol is in **n07**.

The DeBERTa leg is an adjudication test: *do the same pairs remain the most unstable, regardless of which pair is catastrophic on the new backbone?* Three pre-registered predictions with operational success criteria are defined in S01 §DeBERTa-v3 Success Criterion, n05 §5, and n07 §3–5.

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
| `notes/n50_executive_research_summary.md` | **Executive summary.** Project identity, central claim, current product role, research findings, practitioner value, north star. |

### Field Trial Plan and Results

| Document | What it is |
|----------|------------|
| `notes/n51_cpu_field_trial_plan.md` | **CPU-only field trial plan.** 8–10 inventory workflow validation program. |

### Research Synthesis and Negative Results

| Document | What it is |
|----------|------------|
| `notes/n51_research_synthesis_memo.md` | **Research synthesis (prior).** Superseded by n67 for the overall picture, but retains the product-implications framing. |
| `notes/n68_ruled_out_mechanisms.md` | **Ruled-out mechanisms (definitive).** Supersedes n38 and n52. Ten primary eliminations, five ancillary. Portable severity, task-pair lookup, aggregate threshold, readout-as-risk, readout-as-amplifier, feature plurality as universal origin. What survived and the epistemic structure of the eliminations. |
| `notes/n52_ruled_out_mechanisms_packet.md` | **Ruled-out mechanisms (prior).** Superseded by n68. |
| `notes/n38_ruled_out.md` | **Ruled-out hypotheses (original).** Superseded by n68. |

## What should happen next (when GPU returns)

DeBERTa-v3 adjudication (n07) now tests three things, not one:

1. **Predictions A–C:** Does the instability ranking survive on a third backbone?
2. **Prediction D (Rung 1):** Does the module-level V-module dim ratio signal survive on disentangled attention?
3. **Prediction E (Rung 2):** Does head-level cancellation / modulation recur on any seed-sensitive DeBERTa case?

The joint outcome of D and E determines the next mechanistic step (Prediction F): if both survive, O-module head-weight analysis (Rung 3) is the confirmed escalation. If D survives but E does not, head-level modulation may be backbone-specific and the escalation should target DeBERTa's disentangled architecture instead.

LoRA configuration: rank 16, alpha 16, all four attention modules (Q/K/V/O). See n07 for the complete protocol.
