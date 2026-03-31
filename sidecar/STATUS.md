# Sidecar Status — March 2026

## Current state

Phase 3 CPU-only sidecar deepening is complete, **Sidecar B (Output-Space Compatibility) Stage B** is complete, and **Seed-Contingent Readout-Axis Selection Stages A–B** are complete. The sidecar now has a formalized **instability research program** (n06), a complete **DeBERTa adjudication protocol** (n07), expanded **catastrophic anchor dossiers** (n08, n09, n10), **backbone-local interpretation notes** (n11), a **local artifact mining inventory** (n12), a **per-layer structural analysis** (n13–n15) with a MIXED outcome (collision pattern found), a **collision subset formalization** (n16), a **within-layer geometry pilot** (n17–n18) with a MIXED-trending-NEGATIVE outcome, a **per-module geometry pilot** (n19–n21) with a **POSITIVE** outcome — the V-module dimensionality ratio cleanly separates catastrophic from safe collision (d=3.36, zero range overlap) — a **head-level V program** (n22–n24) resolving CA-01 seed sensitivity, and an **output-space readout audit** (n30–n32) with a **MIXED** outcome — readout incompatibility is a necessary background condition but not a discriminative cause.

The mechanism ladder Rung 3 is now resolved: the readout layer functions as a **gate** (transmitting or absorbing upstream V-module pathology), not an amplifier. The conjunctive model is: **V-module pathology × readout incompatibility → catastrophic failure**. The seed-contingent readout program (n34–n36) has shown that readout orthogonality is **bimodal, common even in same-task seed pairs (5/14), and decoupled from upstream V-module geometry**. The two conditions for catastrophe are independently determined.

The **Attractor Mapping Lab** (n39–n41) has mapped the readout attractor landscape across 10 task families. The landscape is **structured, discrete, and safe**: 6 families are single-attractor, 3 multi-attractor, 1 backbone-contingent (MRPC). All families merge safely regardless of attractor type. Task identity is the primary determinant, modulated by backbone architecture, training depth, and training distribution. Multi-attractor ≠ fragile.

The **Attractor Origin Program** Stage A (n43–n44) has identified two distinct mechanisms for multi-attractor structure: **rotational degeneracy** (DistilBERT families — seeds find orthogonal orientations within a shared low-rank subspace) and **feature-set switching** (QNLI on RoBERTa — seeds lock onto genuinely different principal components). One QNLI/RoBERTa seed uses a decision direction aligned with RTE (cos=0.86), confirming cross-task feature exploitation. The simple feature plurality hypothesis is partially falsified; the refined version is backbone-dependent.

The **Attractor Mechanism Determinants** program (n46–n49) has classified all 5 multi-attractor conditions and identified a structured determinant hierarchy: **task identity** (primary, determines attractor count) → **backbone architecture** (secondary, selects mechanism) → **training convergence** (tertiary, gates attractor count) → **domain structure** (weak). Key findings: mechanism and backbone are perfectly confounded (all degeneracy on DistilBERT, all switching on RoBERTa); training depth modulates attractor count but not mechanism; the QNLI cross-backbone contrast is the cleanest evidence for backbone as mechanism selector. Commensurability refined to version 3: readout incompatibility has different semantic content depending on mechanism class (incoherent confidence for rotational degeneracy, systematic misclassification for feature-set switching).

**Output Example Semantics Program** (n59–n66) is complete. The program examined per-example behavioral consequences of safe, fragile, and catastrophic merges across an 8-case panel. **Key findings:** (1) A stable 5-category failure taxonomy captures all observed patterns — preserved consensus (A), better-source loss (C), neither-source behavior (D), benign absorption (E), shared failure (X, excluded). (2) Neither-source behavior (D) is the cleanest discriminator: 12–14% in fragile/control, <2% in safe/near-miss, with nothing in between — a sharp threshold, not a gradient. (3) Two qualitatively distinct failure modes confirmed at example level: fragile merges show confidence collapse (V-module pathology faithfully transmitted), control merges show high-confidence wrong predictions (readout contamination). (4) Near-miss merges are behaviorally indistinguishable from safe retained on all discriminating metrics. (5) The conjunctive model holds at example level — the behavioral signatures map onto the mechanism ladder with a double dissociation between failure modes.

Deliverables: n59 (panel), n60 (protocol), n61 (behavior findings), n62 (taxonomy protocol), n63 (taxonomy findings), n64 (safe vs fragile dossier), n65 (near-miss dossier), n66 (behavior–mechanism bridge), 4 analysis scripts, 8 prediction files, 6 JSON result artifacts, 3 figures.

**Ring 1 PEFT Generalization** is complete. The audit, pairwise comparison, and inventory triage pipeline ran on LoHa (Low-Rank Hadamard Product) adapters via a ~160-line extraction shim, with zero core code modifications. All 6 audit runs (3 adapters x 2 modes) succeeded with meaningful spectral metrics. All 3 pairwise comparisons produced valid reports. The full inventory preflight bundle was emitted. This confirms artifact-class generality for low-rank PEFT methods. Non-low-rank methods (IA3) remain out of scope. Results: `docs/strategy/ring1_peft_generalization_results.md`. Experiments: `experiments/peft_ring1/`.

**Ring 2 Checkpoint-Delta Generalization** is complete (Stages A–D). Full fine-tuned checkpoint deltas from a shared base model (`distilbert-base-uncased`) were audited and triaged on CPU using summary-based representation (Representation C). Stage A compared three candidate representations (raw deltas, truncated low-rank, layerwise summary) and selected Representation C for stability and CPU feasibility. Stage B confirmed single-artifact audit and pairwise comparison work on the summary representation (same-task compatibility 0.89 vs cross-task 0.70). Stage C produced inventory guardrail triage with a full run bundle (6 pairs → 1 retained, QA as binding constraint). Stage D assessed: the workflow survives with a different representation path; evidence bootstrap and QA remain central; merge execution is out of scope. Design notes: `docs/design/ring2_stage_a_checkpoint_delta_representation.md` through `ring2_stage_d_assessment_memo.md`. Experiments: `experiments/ring2_checkpoint_delta/`.

**Field trial validation (Phases 1–2b)** is complete. The preflight workflow is operationally validated across 5 inventories, 3 backbones, and 16 evaluated merges. Near-miss confirmed as a distinct product category (7 pairs, avg Δ vs best = -0.006, comparable to retained). Product validation memo and near-miss validation in `field_trials/`. Research synthesis (n51) and ruled-out mechanisms packet (n52) added to the notes.

**Research synthesis** is now split into two companion documents. **n67** (Where the Research Stands) is the mechanism-ladder synthesis: commensurability, instability, V-module pathology (d=3.36), head-level modulation, readout attractors, conjunctive failure model, and behavioral confirmation through the example-level double dissociation. **n93** (Route 2 Synthesis) consolidates the four Route 2 programs — decision-dependent compatibility, cross-artifact portability, aggregation-sensitive compatibility, and behavioral bridge — into a coherent account of what compatibility means beyond merge, beyond LoRA, and beyond structural measurement. Together they constitute the project's theoretical account.

**Ruled-out mechanisms** (n68) is definitive, superseding n38 and n52. Ten primary eliminations (portable severity, task-pair lookup, aggregate threshold, readout-as-risk, readout-as-amplifier, universal feature plurality) plus five ancillary. The surviving framework and the epistemic structure of the eliminations are documented.

**Settled/open/next index** (n69) is the standalone state-of-project dashboard, maintained independently of n67/n93. Fourteen settled claims (including Route 2 behavioral findings), six open questions, six prioritized next steps. Read this file first when returning to the project after a break.

**Decision-Dependent Compatibility sidecar** (n70-n74) is now complete in bounded form. The program defines a shared merge/routing/triage panel, formalizes shared-vs-specific stack layers, and confirms aggregation as the first practical decision seam in this scope. Stage C was run twice: a checkpoint-triage pass (T02) and a narrow adapter-triage stress pass (targeted confirmation T01). The adapter pass preserved the same-task / same-family / cross-task distributional separation pattern while showing the expected QA-regime contrast (`qa_clear` in adapter T01 vs QA-dominant blocking in checkpoint T02). This supports profile stability without expanding scope. This line is now frozen unless a new external use case appears, GPU unlock enables materially stronger validation, or contradiction appears in real use.

**Route 2 implementation checkpoint** is now complete in bounded form. The repo now includes: broadened substrate scope docs (`docs/strategy/broadened_substrate_scope.md` and companion matrices), checkpoint triage stabilization docs (`docs/strategy/checkpoint_triage_summary.md`, `docs/design/checkpoint_triage_language_and_boundaries.md`, `field_trials/checkpoint_inventory_summary.md`), a polished checkpoint-triage alpha workflow package (`field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/`, `docs/examples/checkpoint-triage-alpha-workflow.md`, `docs/strategy/checkpoint_triage_alpha_scope.md`), decision-dependent consolidation implications (`docs/strategy/decision_dependent_compatibility_implications.md`), a completed cross-artifact portability pass (`notes/n76_cross_artifact_panel_definition.md` through `notes/n80_cross_artifact_product_relevance.md`, anchored by `notes/n75_cross_artifact_compatibility.md`), and an external use-case scan (`docs/strategy/external_use_case_scan.md`).

**Cross-Artifact Compatibility Research Program** (n76-n80) is complete. A 9-case panel spanning LoRA, LoHa, and checkpoint delta artifact classes was audited for signal portability across representation families. **Key findings:** (1) Two strong invariants -- evidence regime gating and conservative candidate narrowing -- recur across all three artifact classes at the workflow level. (2) Two moderate invariants -- task-relation ordering and same-family intermediate status -- recur where testable. (3) No structural metric is fully portable; the V-module signal (d=3.36) is representation-locked to factorized artifacts. (4) Triage is the only cross-artifact decision scenario; merge and routing remain LoRA-specific. Framework: three layers (artifact-invariant, representation-family, decision-dependent). Results: `docs/strategy/cross_artifact_product_relevance_summary.md`. Notes: n76-n80.

**Route2 Substudy 1 — Cross-Artifact Portability Stability Check** (n93-n97) is complete. The original n76-n80 conclusions were re-run under a small panel perturbation (4/9 case substitutions, no artifact expansion). **Stability verdicts:** A1 (evidence gating) stable, A2 (conservative narrowing) stable, C1 (structural locality) stable, B1 (same-task vs cross-task separation) moderately stable, B2 (same-family strict intermediate ordering) panel-sensitive, D1 (near-miss portability) still inconclusive. Results: `docs/strategy/cross_artifact_stability_summary.md`. Notes: n93-n97.

**Aggregation-Sensitive Compatibility Research Program** (n81-n85) is complete. A 12-case panel with matched QA-regime pairs tested whether different aggregation rules produce genuinely different operational judgments from the same structural evidence. **Key findings:** (1) Only 2/12 cases are aggregation-invariant (both cross-task with clear QA). (2) The routing confusable/moderate/separable gradient is visible only under distributional aggregation; worst-case collapses all three to the same label. (3) QA-dominant can override the strongest structural signal (0.892 compatibility blocked). (4) The hybrid (QA-gated distributional) is the richest family, preserving both evidence constraints and structural gradation. (5) Five stable aggregation-sensitive patterns are predictable from QA regime and task relation. Aggregation is a computational step, not a presentation layer. Results: `docs/strategy/aggregation_sensitive_route2_summary.md`. Notes: n81-n85.

**Route2 Substudy 2 — Aggregation-Sensitive Compatibility Stability Check** (n98-n102) is complete. The original aggregation conclusions were re-run under a disciplined local perturbation (3/12 substitutions, one each for merge, routing, triage; no aggregation-family changes). **Stability verdicts:** A1 stable, B1 stable, B3 stable, C2 stable, D1 stable; B2 moderately stable; C1 moderately stable. No previously strong aggregation claim degraded to panel-sensitive in this pass. Results: `docs/strategy/aggregation_stability_summary.md`. Notes: n98-n102.

**Route2 Aggregation Mixed-Evidence Triage Perturbation** (n103-n107) is complete. A triage-weighted soft-middle stress pass (8 cases, intentionally mixed evidence / same-family optional heavy) tested whether QA-dominant logic and taxonomy usability hold beyond clean retained-vs-blocked panels. **Soft-middle verdicts:** QA-dominant coherence `coherent`; same-family optional safe-likeness, soft-middle taxonomy usability, structural nuance in mixed review, and review-vs-low-value differentiation `coherent_with_guardrails`; 0 ambiguous; 0 weakened. Results: `docs/strategy/aggregation_mixed_evidence_summary.md`. Notes: n103-n107.

**Route 2 Claims Stability Ladder** (n108-n112) is complete. This synthesis program graded 20 Route 2 claims across evidence base, perturbation survival, artifact coverage, behavioral grounding, and product relevance. **Ladder distribution:** 11 `stable`, 6 `moderately_stable`, 2 `thin`, 1 `local_only`, 0 `blocked_or_open`. Stable cluster is dominated by workflow invariants and aggregation seam claims; most guarded claims are same-family ordering/optionality portability and scenario-local behavioral transfer. Results: `docs/strategy/route2_claims_ladder_summary.md`. Structured outputs: `sidecar/results/route2_claims_ladder/`. Notes: n108-n112.

**Behavioral Route 2 Bridge** (n86-n92) is complete. An 8-case panel (reusing n59-n66 example-level data) was reinterpreted through 5 Route 2 compatibility profiles to test whether broadened profiles have distinct behavioral signatures. **Key findings:** (1) Four of five profiles have strong, distinct behavioral footprints. (2) Behavioral signatures cluster into three tiers: no pathology (safe/optional, neither-source <2%), localized pathology (collapse/cross-task, neither-source ~14%), and stasis (QA review, shared failure 65%). (3) The collapse/contamination mode split is the most important behavioral finding: same failure rate but different channels (uncertainty-driven vs confident-wrong). (4) Same-family optional is behaviorally indistinguishable from aggregation-invariant safe. (5) Routing-confusability does not produce confusion-like behavior in the merge setting. Results: `docs/strategy/behavioral_route2_summary.md`. Notes: n86-n92.

**Route 2 Packet** (`packet/route2/`) is complete. A self-contained sub-packet summarizing all broadened-compatibility work: substrate generalization (3 axes), four research programs (decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge), checkpoint triage alpha workflow, and key figures. Designed to be handed to a collaborator or external reader without requiring the full sidecar.

Total assets: 109 notes, 1 study, 17 analysis scripts, 114 structured data outputs, 67 figures, 1 dossier template, 1 extended case table, 2 research packets.

## What is blocked on GPU

Training 8 DeBERTa-v3-base adapters (4 GLUE tasks × 2 seeds) and evaluating 28 merge pairs. This is the DeBERTa leg of S01 — the adjudication test for whether **instability** is a **portable descriptor**. Estimated compute: ~3 hours on a single consumer GPU. The complete protocol is pre-registered in n07.

## The next decisive question

Do the same pairs remain the most unstable on a third backbone, regardless of which pair is catastrophic?

Three pre-registered predictions (S01 §DeBERTa-v3 Success Criterion, n07 §3):

- **A:** QNLI×MRPC and QNLI×SST-2 have the highest seed ranges on DeBERTa.
- **B:** The four stable-asymmetric pairs stay below 10% seed range.
- **C:** The instability gap (0.30–0.74) remains empty.

If A holds, instability is portable. If A fails, the working concept needs revision. Full decision tree in n07 §5.

## What was accomplished in Phase 3

### Project F — Instability Program Consolidation
- n06: formal program statement (commitments, predictions, falsification conditions)
- n07: complete DeBERTa adjudication protocol (training, merge, analysis, decision tree)
- Extended instability case table with DeBERTa predictions and confidence ratings

### Project H — Catastrophic Anchor Dossiers
- Dossier template for systematic anchor documentation
- n08: full CA-01 dossier (QNLI×MRPC on DistilBERT)
- n09: full CA-02 dossier (QNLI×SST-2 on RoBERTa)
- n10: cross-dossier synthesis (patterns, core implications, contrast cases)

### Project I — Backbone-Local Interpretation Notes
- n11: per-backbone regularities (SST-2 escalation, toxic adapter, same-task tightness, cross-backbone contrasts)

### Project G — Local Artifact Mining
- n12: mining inventory (available assets, preliminary findings, priority analysis)
- Key findings: task-asymmetry does not predict catastrophe, qnli_s42 appears in both anchors' worst variants, core signals are non-discriminating within cross-task regime
- n13: per-layer artifact inventory (all 16 adapters located and verified, contrast panel defined)
- n14: per-layer comparison protocol (4 metrics: norm mass, pair divergence, concentration index, alignment proxy)
- n15: per-layer findings (**MIXED** outcome)
  - **Collision pattern found:** Catastrophic pairs show *lower* pair divergence (JS=0.007) and *higher* alignment (ρ=0.76) than stable cross-task pairs (JS=0.014, ρ=0.62). The mechanism looks like same-layer interference, not cross-layer mismatch.
  - **Groups overlap:** No single metric cleanly separates catastrophic from stable pairs. The alignment proxy is the best discriminator but ranges overlap.
  - **Seed sensitivity is sub-layer:** Per-layer metrics do not explain the 29-point seed range within CA-01. The seed-dependent variable operates within layers, not across them.
  - **Recommended follow-up:** Within-layer subspace angle analysis (principal angles at high-norm layers)

### Within-Layer Collision Program (Stages A–B)
- n16: collision subset definition — classified all 20 pair×backbone cases into collision categories
  - Key finding: collision is a **risk amplifier**, neither necessary nor sufficient for catastrophe
  - MRPC×SST-2 on RoBERTa: highest cross-task alignment (ρ=0.89) but stable (instability=0.21)
  - CA-02 (QNLI×SST-2 on RoBERTa): only moderate alignment (ρ=0.66) despite being catastrophic
- n17: within-layer geometry protocol (4 metrics: principal angles, top-direction overlap, dimensionality ratio, directional conflict)
- n18: within-layer geometry findings (**MIXED, trending NEGATIVE**)
  - **Backbone confound dominates:** CA-01 (DistilBERT) shows high top-direction overlap due to 6-layer compression, not catastrophe-specific geometry. When backbone is controlled, CA-02 is indistinguishable from safe RoBERTa controls.
  - **Seed sensitivity NOT explained:** CA-01's 29-point seed range is invisible in all four within-layer metrics.
  - **Directional conflict is reversed:** Catastrophic pairs show *lower* conflict — consistent with a "similarity without identity" mechanism rather than "opposition."
  - **Ruling out:** Aggregate per-layer subspace geometry is NOT the threshold variable for catastrophic interference.

### Per-Module Geometry Program (Stages A–B)
- n19: per-module subset definition — same 6-case contrast panel decomposed into Q/K/V/O
- n20: per-module geometry protocol (same 4 metrics applied per module)
- n21: per-module geometry findings (**POSITIVE**)
  - **V-module dimensionality ratio is the strongest signal in the sidecar program:** Cohen's d = 3.36, zero range overlap between catastrophic and safe collision cases (backbone-controlled).
  - **K module is secondary discriminator:** d = 1.39 on dimensionality ratio.
  - **CA-02 seed sensitivity now partly explained:** The toxic adapter (qnli_s42) shows dramatically different O-module (Δcos = -0.31) and V-module (Δcos = -0.15) geometry vs. the benign adapter — invisible in the aggregate analysis.
  - **CA-01 seed sensitivity remains unexplained** at per-module resolution (all deltas < 0.07).
  - **The concatenation was diluting the signal:** Per-module decomposition succeeds precisely because it avoids averaging across modules with structurally different patterns.

### Sidecar B — Output-Space Compatibility (Stages A–B) + Conjunctive Synthesis
- n30: output-space panel definition — 11 cases across 5 groups, classifier head weights confirmed in all 16 source adapters
- n31: output-space readout protocol — 5 metrics, 5 contrasts, interpretation rules
- n32: output-space readout findings (**MIXED — critical falsifier**)
- n33: conjunctive mechanism synthesis — integrates Sidecar A (Rungs 1–2) and Sidecar B (Rung 3) into unified model; supersedes n25 Rung 3
  - **SC-QMRB falsifies readout-alone hypothesis:** Same readout geometry as CA-01 (both ~89° orthogonal, both ~0.70 margin proxy) but safe (Δ=1.7% vs 41.7%). Readout incompatibility is task-pair-determined, not backbone-determined.
  - **Compatible readout reliably predicts safety:** 3/3 cases with decision_axis_cos > 0.95 are safe/mild.
  - **Incompatible readout does NOT predict catastrophe:** Only 4/7 orthogonal-readout cases are catastrophic or moderate; 2 are mild, 1 is safe.
  - **Seed contrast shows zero readout signal:** CA-01-cat and CA-01-mild have virtually identical readout geometry despite 29pp gap.
  - **CA-02 toxic/benign shows seed-contingent readout:** Same task pair produces orthogonal or aligned readout depending on seed.
  - **Readout layer is a gate, not an amplifier:** It transmits or absorbs upstream V-module pathology rather than generating new risk.
  - **Conjunctive model confirmed:** V-module pathology × readout incompatibility → catastrophic failure.

### Seed-Contingent Readout-Axis Selection (Stages A–B)
- n34: seed readout panel definition — 14 same-task seed pairs + 3 adjacent-task pairs across 3 study groups
- n35: upstream-readout coupling protocol — readout + V-module + coupling metrics
- n36: upstream-readout coupling findings (**MIXED — decoupled**)
  - **Same-task seeds show orthogonal readout:** 5 of 14 same-task seed pairs have decision_axis_cos ≈ 0, yet all merge safely (Δ ≤ 2.2%)
  - **Bimodal distribution:** Decision-axis cosine clusters at ~0 or ~1 with no intermediate values — same pattern as cross-task pairs
  - **Decoupled from upstream:** All same-task pairs have healthy V-module geometry (dim ratio > 0.78) regardless of readout classification
  - **Task-specific attractor structure:** QNLI always orthogonal, RTE/SST-2 always aligned, MRPC varies by backbone
  - **Training convergence matters:** Strong QNLI is orthogonal, Medium/Weak are aligned — same task, same seeds, different training duration
  - **Readout orthogonality is not pathological:** It is a routine feature of LoRA fine-tuning, harmless unless paired with upstream V-module pathology

### Attractor Mapping Lab (Stages A–B)
- n39: attractor panel definition — 14 family×backbone entries across 4 groups, all success criteria met
- n40: family readout audit protocol — 4 metrics, 5 contrasts, deterministic classification rule
- n41: family readout audit findings (**POSITIVE**)
  - **Attractor landscape is discrete:** All families fall cleanly into single-attractor or multi-attractor, no intermediate cases
  - **6 single-attractor families:** RTE, SST-2 (core), Yelp, Amazon, Medium QNLI, Weak QNLI
  - **3 multi-attractor families:** QNLI, SST-2 (domain shift), Strong QNLI
  - **1 backbone-contingent family:** MRPC (orthogonal on DistilBERT, aligned on RoBERTa)
  - **Task identity is primary determinant:** QNLI is multi-attractor everywhere; RTE is single-attractor everywhere
  - **Convergence-contingent:** Strong QNLI is multi-attractor, Medium/Weak are single-attractor (same task, same seeds)
  - **Multi-attractor ≠ fragile:** All families merge safely regardless of attractor type (max Δ = 2.2%)
  - **Cross-domain alignment tracks attractor structure:** SST-2 (domain) orthogonal to Yelp/Amazon; Yelp×Amazon aligned

### Attractor Origin Program (Stage A — Decision-Axis Analysis)
- n43: attractor origin research program — feature plurality hypothesis, 3-stage design
- n44: decision-axis analysis findings (**MIXED-POSITIVE**)
  - **Simple feature plurality partially falsified:** Most multi-attractor families' seeds use the same PCs but combine them in orthogonal directions
  - **Two mechanisms identified:** Mechanism 1 (rotational degeneracy, DistilBERT) and Mechanism 2 (feature-set switching, QNLI/RoBERTa)
  - **QNLI/RoBERTa is the only clear feature-plurality case:** Energy overlap = 0.255, shared top-3 PCs = 0
  - **Cross-task feature exploitation confirmed:** QNLI/rb/s7 aligns with RTE/rb/s7 (cos = 0.86) — one QNLI seed uses an RTE-like decision direction
  - **Effective axis cosine cleanly separates groups:** Single-attractor > 0.94, multi-attractor < 0.76, zero overlap
  - **PC effective rank is a backbone signature, not an attractor-class signature:** DistilBERT ≈ 1–3, RoBERTa ≈ 2–13

### Attractor Mechanism Determinants (Stages A–C)
- n46: mechanism classification audit (**POSITIVE**)
  - All 5 multi-attractor conditions classified with high confidence
  - 4 rotational degeneracy (QNLI/db, SST-2(dom)/db, Strong QNLI/db, MRPC/db)
  - 1 feature-set switching (QNLI/rb)
  - 0 mixed/unresolved
  - Mechanism and backbone perfectly confounded in current panel
- n47: determinant protocol — 5 factors, 4 contrasts
- n48: determinant findings (**MIXED-POSITIVE**)
  - Structured hierarchy: task → backbone → convergence → domain
  - Backbone contrast: QNLI changes mechanism across backbones (degeneracy on db, switching on rb)
  - Convergence contrast: training depth changes attractor count but not mechanism
  - Causal model: task sets attractor capacity; backbone selects mechanism; convergence gates count
  - Critical limitation: backbone confound not resolvable without DeBERTa data
- n49: mechanism-to-commensurability synthesis
  - Three kinds of benign diversity: single-attractor stability, rotational degeneracy, feature-set switching
  - Commensurability v3: readout condition decomposes by mechanism class
  - Different failure semantics: incoherent confidence (degeneracy) vs systematic misclassification (switching)
  - Commensurability context table produced

## What should happen next

### When GPU returns
Execute the DeBERTa adjudication protocol (n07). This is the single most important next step. The adjudication now has two targets:

1. **Instability ranking portability** (Predictions A–C): do the same pairs remain the most unstable on DeBERTa?
2. **V-module dimensionality mismatch portability** (Prediction D): does the V-module dimensionality-ratio signal separate catastrophic from safe collision on a backbone with disentangled attention?

Prediction D is the sharper and more falsifiable target. LoRA adapters must be trained with rank 16 targeting all four attention modules (matching the existing evidence base, not the original S01 spec of r=8). See n07 for the complete updated protocol, including DeBERTa module mapping, per-module analysis steps, and the expanded decision tree.

### CPU-only maintenance
- Promotion Dossier Template (Project K) — useful for formalizing the instability promotion assessment
- Publication-ready figures (Project L) — the existing figure set is adequate but could be improved
- Per-module seed sensitivity for CA-01 at sub-module resolution (attention-head-level decomposition within V module)

## What should NOT be done before DeBERTa

- Do not attempt to promote instability or V-module dimensionality ratio to core. Two backbones are suggestive, not decisive.
- Do not refine the composite instability score. The current formula is adequate; recalibration should wait for three-backbone data.
- Do not write external-facing summaries. The V-module finding is strong within the evidence base but the evidence base is too thin for public claims.
- Do not prototype V-module dimensionality ratio as a core Gradience metric. Wait for DeBERTa confirmation.
