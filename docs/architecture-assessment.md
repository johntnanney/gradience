# Architecture Assessment: Gradience Substrate Generality

**Status:** FROZEN — canonical position document
**Date:** 2026-03-31
**Purpose:** Map which components of the Gradience vnext implementation are inherently merge-specific, which are already general-purpose, and where a clean extraction boundary exists for non-merge triage scenarios.

> This document is the canonical answer to "is Gradience broader than merge?" All future generalization work — including the routing pilot and Ring evaluations — should be evaluated against the zone boundaries and extraction costs described here. If any result invalidates a claim in this document, update this document first.

---

## 1. Motivation

The sidecar research program settled two product-facing questions (same-family routing and near-miss severity ordering) and confirmed that Gradience's core workflow — audit individual adapters, compare pairs, aggregate an inventory, produce an action plan — is empirically validated for merge preflight. The strategic question is whether this workflow generalizes beyond merge, and if so, how much of the current codebase is already substrate for that generalization versus how much is merge-specific machinery that would need to be duplicated or abstracted.

This memo answers that question by reading the code, not by speculating about future products.

---

## 2. Component Map

The vnext implementation has five functional layers. Each is assessed for merge specificity.

### Layer 1: Single-Adapter Measurement

**Modules:** `vnext/audit/lora_audit.py`, `vnext/audit/gain_metrics.py`, `vnext/audit/rank_policies.py`

These modules compute spectral properties of individual LoRA adapters: stable rank, utilization, energy rank, structural flags. They take a PEFT directory or state dict as input and return a `LoRAAuditResult` — a frozen dataclass of per-layer measurements.

**Merge specificity: None.** These modules never mention merge. They are pure measurement of individual adapter geometry. Any triage scenario that starts with "tell me about this adapter" reuses this layer without modification.

### Layer 2: Adapter QA Judgment

**Modules:** `vnext/audit/qa_artifact.py`, `vnext/merge/eligibility.py`

The QA artifact (`AdapterQAArtifact`) combines structural measurements from Layer 1 with optional behavioral evidence (adapter score, base score, eval dataset) to produce an eligibility judgment: eligible, uncertain, flagged_weak, or unknown. The `build_qa_artifact()` function is the builder; the output is a frozen v1-schema JSON artifact.

**Merge specificity: Low, with one coupling point.** The QA artifact itself is general — it answers "is this adapter worth using?" regardless of what "using" means. The coupling point is `to_qa_result()`, which bridges to `AdapterQAResult` (defined in `merge/eligibility.py`). This bridge exists because the merge pipeline was the first consumer. The eligibility types (`EligibilityStatus`, `ConfidenceLevel`) are defined in the merge package but contain no merge-specific logic. If extraction were needed, moving these types to a shared location would be a mechanical rename, not a design change.

### Layer 3: Pair Comparison

**Modules:** `vnext/merge/spectral_compat.py`, `vnext/merge/verdicts.py`, `vnext/merge/recommend.py` (Stage A only)

This is where the architecture gets interesting.

`spectral_compat.py` computes `SubspaceMetrics` from two sets of LoRA factors: principal angle cosines, mean overlap, directional agreement, magnitude ratio, Frobenius norms, scale ratios. The math is pure linear algebra on paired subspaces. The docstring cites Bjorck & Golub (1973) on principal angles.

`verdicts.py` translates `SubspaceMetrics` into `LayerVerdict` objects via a six-branch decision tree (safe, redundant, conflicting, imbalanced). Thresholds are tunable via `VerdictThresholds`.

`recommend.py` Stage A (`diagnose_layer`, `diagnose_pair`) extracts a `PairDiagnosis` from the spectral data — structured facts about compression need, risk level, and conflict classification. Stage A is documented as "pure spectral analysis, no policy decisions."

**Merge specificity: Naming only.** The actual computation in these modules is pair-wise subspace comparison. Nothing in the math or decision logic depends on the operation being a merge. The question "how do these two adapters' learned subspaces relate geometrically?" is meaningful for any scenario where you have two adapters and want to know whether combining, stacking, routing, or sequencing them will cause interference. The modules live in `vnext/merge/` and import from `gradience.exceptions.MergeError`, but that is a packaging decision, not an architectural one.

**Important caveat:** `recommend.py` Stage B (the policy layer) *is* merge-specific. It translates diagnosis into merge strategy choices (`linear`, `ties`, `dare_ties`, `norm_equalized`, `audit_aware`) with specific coefficient tuning. This is the first layer where "what should you do about it?" becomes merge-flavored.

### Layer 4: Inventory Aggregation and Action Planning

**Modules:** `vnext/inventory/summary.py`, `vnext/inventory/portfolio.py`

`summary.py` defines `InventorySummary` (count distributions over adapter statuses, pair risks, strategies, dominant issues) and `InventoryActionPlan` (sorted buckets: exclude, same-task priority, cross-task caution, evaluate-first; plus near-miss candidates with severity classification). The `build_action_plan()` function takes lists of QA artifacts and merge reports and produces an action plan by reading existing judgments — it does not compute new spectral data.

**Merge specificity: Moderate.** The aggregation logic (count things, sort things, bucket things) is completely general. The *vocabulary* of what gets counted is currently merge-vocabulary: pair risk, merge strategy, merge dominant issue. The action plan's bucket names (`same_task_priority`, `cross_task_caution`) assume a merge triage context. But the *structure* — "here are your items, sorted by how much attention they need, with the most actionable ones first" — is a general triage pattern.

`portfolio.py` scans directories for adapter artifacts. General-purpose.

### Layer 5: Merge Execution

**Modules:** `vnext/merge/__init__.py` (orchestrator), `vnext/merge/executor.py`, `vnext/merge/plan.py`, `vnext/merge/strategies.py`, `vnext/merge/io.py`, `vnext/merge/report.py`, `vnext/merge/qa_report.py`

These modules implement the actual merge: load adapter weights, apply a merge strategy (linear interpolation, TIES, DARE, norm-equalized), write the merged adapter, and generate reports.

**Merge specificity: Complete.** This is the merge product. It consumes the outputs of Layers 1–4 and produces a merged adapter. No generalization needed or desired — this is the thing Gradience does.

### Supporting Modules

`vnext/merge/task_families.py` — Static taxonomy mapping dataset names to task-family labels. General-purpose; useful for any scenario that needs to know "are these two tasks similar?"

`vnext/inventory/batch.py`, `neighborhoods.py`, `run_bundle.py`, `html_report.py`, `corpus_manifest.py` — Merge-inventory-specific operational tooling.

---

## 3. The Extraction Boundary

The codebase reveals a clean three-zone architecture:

**Zone A — General adapter intelligence (no merge dependency):**
- `audit/lora_audit.py` — structural measurement
- `audit/qa_artifact.py` — eligibility judgment
- `audit/gain_metrics.py` — spectral utilities
- `audit/rank_policies.py` — structural assessment
- `merge/eligibility.py` — eligibility types (mislocated; belongs in a shared package)
- `merge/task_families.py` — task taxonomy (mislocated; belongs in a shared package)
- `inventory/summary.py` — aggregation engine (general structure, merge-specific vocabulary)
- `inventory/portfolio.py` — directory scanning

**Zone B — Pair-wise geometric comparison (merge-named, math-general):**
- `merge/spectral_compat.py` — subspace metrics
- `merge/verdicts.py` — compatibility classification
- `merge/recommend.py` Stage A — pair diagnosis

**Zone C — Merge product (inherently merge-specific):**
- `merge/__init__.py` — merge orchestration
- `merge/recommend.py` Stage B — merge policy
- `merge/executor.py`, `plan.py`, `strategies.py`, `io.py` — merge execution
- `merge/report.py`, `qa_report.py` — merge reporting
- `inventory/batch.py`, `neighborhoods.py`, etc. — merge inventory operations

The boundary between Zone B and Zone C falls exactly at the diagnosis/policy split that `recommend.py` already documents. This is not an accident — the two-stage architecture was designed with this separation in mind. The code already knows where the general-purpose analysis ends and the merge-specific decisions begin.

---

## 4. What Extraction Would Actually Require

### Mechanical work (low risk):

1. **Move eligibility types to a shared location.** `EligibilityStatus`, `ConfidenceLevel`, and `AdapterQAResult` currently live in `merge/eligibility.py`. Moving them to `vnext/types.py` or a new `vnext/eligibility.py` would break no interfaces — only import paths change. The `to_qa_result()` bridge on `AdapterQAArtifact` would import from the new location.

2. **Move `task_families.py` out of `merge/`.** The task-family registry is already merge-independent. Moving it to `vnext/` or `vnext/taxonomy/` is a rename.

3. **Rename `MergeError` usages in Zone B modules.** `spectral_compat.py` raises `MergeError` for invalid inputs. A general `CompatibilityError` or reuse of the existing `AuditError` would be more accurate.

### Vocabulary work (moderate effort):

4. **Parameterize action plan vocabulary.** `InventoryActionPlan` uses merge-specific bucket names. A non-merge triage scenario would need different bucket labels (e.g., "routing candidates" instead of "same-task priority," "interference risk" instead of "cross-task caution"). This could be handled by making the action plan a more generic structure with scenario-specific labeling, or by building scenario-specific plan builders that reuse the underlying sort/bucket logic.

5. **Abstract the report layer.** `MergeQAReport` is inherently merge-specific. A non-merge scenario would need its own report type, but would consume the same `SubspaceMetrics` and `PairDiagnosis` data.

### Design work (requires thought):

6. **Define what "Stage B" means for a non-merge scenario.** The diagnosis/policy split is clean, but the diagnosis only becomes useful when there is a policy layer that translates it into actionable recommendations. For merge, this is strategy selection and coefficient tuning. For a different scenario — say, adapter routing or interference detection — the policy layer would need to map the same diagnosis facts to different decisions. The question is whether this is a plugin architecture (a `TriagePolicy` protocol with scenario-specific implementations) or whether each scenario just writes its own consumer of the diagnosis output.

---

## 5. One Non-Merge Pilot Scenario: Adapter Routing Triage

The most natural non-merge test case for the existing substrate is **adapter routing** — given a set of task-specific LoRA adapters deployed behind a router, assess which pairs have overlapping subspaces that might cause confusion or interference if the router makes a wrong assignment.

### Why this scenario:

- It uses the same adapter QA artifacts (Layer 1–2): each adapter still needs structural measurement and eligibility judgment.
- It uses the same pair-wise comparison (Layer 3, Zone B): "how much do these two adapters' subspaces overlap?" is exactly the question you ask when worrying about routing confusion.
- It needs a *different* policy layer: high overlap between two adapters isn't a merge risk, it's a routing disambiguation problem. The verdicts flip: "redundant" (merge-bad) becomes "confusable" (routing-bad), while "conflicting" (merge-dangerous) becomes "well-separated" (routing-good).
- It needs a different action plan vocabulary: instead of "merge this pair safely / with caution," the triage says "these adapters are easy to route between / hard to distinguish / essentially interchangeable."

### What it would test:

This pilot would validate whether Zone B's pair-wise geometric analysis is genuinely reusable by writing a thin policy layer (~100–200 lines) that consumes `PairDiagnosis` objects and produces a routing-specific assessment. If the pilot works without modifying any Zone A or Zone B code, the substrate generality claim is confirmed. If it requires changes to the diagnosis layer, those changes reveal what was implicitly merge-specific.

### Minimal implementation:

1. A `routing_verdicts.py` module that maps `SubspaceMetrics` → routing-specific classifications (confusable, distinguishable, redundant).
2. A `routing_report.py` that formats the assessment for a routing audience.
3. A `routing_action_plan` builder that aggregates routing assessments into a fleet-level triage.
4. A script that takes the same inventory of adapters used in a field trial and produces a routing triage report instead of a merge triage report.

No changes to `spectral_compat.py`, `verdicts.py` Stage A, `lora_audit.py`, `qa_artifact.py`, or `summary.py`'s aggregation logic. If this holds, the extraction thesis is validated.

---

## 6. What This Assessment Does Not Recommend

This memo deliberately avoids:

- **Proposing a new product.** The question is architectural, not commercial.
- **Refactoring now.** The extraction boundary is clean enough that no refactoring is needed before testing the thesis. The routing pilot can import from `vnext/merge/spectral_compat` and `vnext/merge/recommend` directly. If it works, *then* the mechanical renames in §4 become worth doing. If it doesn't, the merge-specific packaging was correct all along.
- **Generalizing the inventory layer prematurely.** The action plan vocabulary should only be generalized if the routing pilot actually produces a natural inventory-level triage. Abstracting vocabulary before having a second concrete scenario is the definition of premature generalization.

---

## 7. Summary

| Layer | Module(s) | Merge-specific? | Extraction cost |
|-------|-----------|-----------------|-----------------|
| Single-adapter measurement | `audit/lora_audit`, `gain_metrics`, `rank_policies` | No | Zero |
| Adapter QA judgment | `audit/qa_artifact`, `merge/eligibility` | Naming only | One import-path move |
| Pair-wise comparison | `spectral_compat`, `verdicts`, `recommend` Stage A | Naming only | One exception rename |
| Inventory aggregation | `inventory/summary`, `portfolio` | Vocabulary | Moderate (parameterize buckets) |
| Merge policy + execution | `recommend` Stage B, `executor`, `strategies`, etc. | Yes | N/A — this is the product |

### Revised architectural model (post-pilot)

The original assessment identified a two-layer split: diagnosis vs. policy. The routing pilot revealed a finer-grained four-layer model:

1. **Measurement** — per-adapter and per-pair spectral computation (`lora_audit`, `spectral_compat`). Fully general. Produces geometric facts.
2. **Diagnosis** — classification of measurements into named conditions (`verdicts`, `recommend` Stage A). Mostly general, though merge currently owns the vocabulary. Optional for non-merge consumers — the routing pilot consumed `SubspaceMetrics` directly and bypassed the verdict layer entirely.
3. **Aggregation** — how per-layer assessments combine into a pair-level or fleet-level picture. This is a scenario-specific design choice that the original assessment did not identify. Merge uses worst-case aggregation (correct: one bad layer can ruin a merge). Routing uses distributional aggregation (correct: typical separability matters more than worst-case overlap). Future scenarios may use thresholded, confidence-weighted, or hybrid aggregation strategies.
4. **Policy** — scenario-specific decisions and vocabulary. Merge: strategy selection, coefficient tuning. Routing: confusability classification, dedup/disambiguation recommendations.

The two parameterization points for extraction are **aggregation strategy** and **policy vocabulary**. Both emerged from the pilot empirically. This model replaces the original two-layer framing as the canonical description of the extraction seam.

---

## 8. Pilot Validation (2026-03-29)

The routing pilot (`experiments/routing_pilot/`) ran successfully against the thesis in this document. Full results are in `routing_pilot_field_note.md`. Key findings:

- **Zero modifications to any existing module.** The pilot consumed `load_adapter`, `match_layers`, `extract_factors`, `compute_subspace_metrics`, and `assess_layer` exactly as documented.
- **5/6 pairs showed policy-layer divergence.** Merge called all pairs "redundant" (worst-case aggregation); routing discriminated them into high / moderate / low confusability (distributional aggregation). Same spectral data, different operational guidance.
- **The generalization boundary is at the metrics level, not the verdict level.** The routing layer consumed `SubspaceMetrics` directly and did not need `assess_layer` for its own path. The real abstraction seam is below where this assessment predicted — the measurement output is fully general, and even the verdict layer is optional for non-merge consumers.
- **Aggregation strategy is a second parameterization point.** Beyond policy vocabulary, worst-case vs. distributional aggregation over per-layer data is a meaningful design choice that different scenarios resolve differently.

The pilot validated substrate generality and revised the extraction seam model. The original assessment predicted a two-layer split (diagnosis vs. policy). The pilot revealed that aggregation strategy — how per-layer assessments combine into pair-level or fleet-level pictures — is a distinct parameterization point between diagnosis and policy. Section 7 now reflects the revised four-layer model. Full results: `docs/routing-pilot-results.md`.

Assessment status: **CONFIRMED.** The substrate generality thesis holds. The mechanical renames in §4 are now justified if a second concrete generalization result materializes.

## 9. Ring 1 PEFT Generalization (2026-03-30)

Ring 1 tested the substrate along a second axis: **artifact-class generality**. The routing pilot (§8) proved scenario generality (merge vs. routing on the same LoRA artifacts). Ring 1 proved that the same pipeline operates on a different PEFT artifact class (LoHa) without modifying any core module.

Key findings:

- **Zero core code modifications.** A ~160-line extraction shim (`experiments/peft_ring1/loha_shim.py`) converts LoHa state dicts to LoRA-format keys. The existing `_iter_lora_pairs()`, `load_adapter()`, `compute_subspace_metrics()`, and the entire inventory pipeline consumed the shimmed artifacts unmodified.
- **Both analysis modes succeeded.** Factor-level analysis (treating each Hadamard factor pair as an independent LoRA pair) and materialized analysis (SVD of the full Hadamard product) produced meaningful, non-degenerate spectral metrics across 3 adapters and 6 audit runs.
- **Pairwise comparison transferred cleanly.** All 3 LoHa adapter pairs produced valid merge-audit reports with sensible pair-risk and strategy recommendations.
- **Inventory triage transferred cleanly.** `build_inventory_summary()`, `build_action_plan()`, and `emit_run_bundle()` produced correct output. The report vocabulary (pair risk, strategy, dominant issue, action plan zones) remained meaningful without LoHa-specific wording.
- **The LoRA-specific surface is extraction-level only.** The blockers are: `_iter_lora_pairs()` (key patterns), config parsing (`lora_alpha`), and the merge executor (weight reconstruction). Everything above the extraction layer is artifact-agnostic.

This confirms the four-layer model from §7:

| Layer | LoRA-specific? | Ring 1 evidence |
|-------|---------------|-----------------|
| Measurement | No | LoHa factor pairs and materialized deltas produce valid spectral metrics |
| Diagnosis | No | Pairwise subspace comparison works on LoHa-derived factors |
| Aggregation | No | Inventory summary, action plan, and preflight bundle transfer unchanged |
| Policy | Partially | Merge execution remains LoRA-specific; audit and triage vocabulary is general |

**Revised assessment status (post-Ring 1): CONFIRMED on two axes.** The substrate generalizes along both the scenario axis (merge vs. routing, §8) and the artifact-class axis (LoRA vs. LoHa, this section). The LoRA-specific surface is limited to weight-key extraction and merge execution. The mechanical renames in §4 are now clearly justified.

Full results: `docs/strategy/ring1_peft_generalization_results.md`. Design doc: `docs/design/peft_generalization_audit.md`. Experiments: `experiments/peft_ring1/`.

## 10. Ring 2 Checkpoint-Delta Generalization (2026-03-30)

Ring 2 tested a third axis: **representation-path generality** for full fine-tuned checkpoint deltas from a shared base model. Unlike LoRA/LoHa, this setting does not expose low-rank factors directly, so the question was whether the workflow survives when the representation changes.

Ring 2 outcomes (Stages A-D):

- **Representation selection (Stage A):** tested three candidates for full-checkpoint deltas (raw deltas, truncated low-rank approximation, layer-summary representation). Representation C (layerwise summaries) was selected as the CPU-feasible operational path.
- **Audit + pairwise validation (Stage B):** the selected summary representation supported single-artifact audit and pairwise comparison on the bounded panel.
- **Guardrail triage packaging (Stage C):** inventory triage and run-bundle outputs were produced without core refactor; policy posture remained intentionally narrow because source QA was the binding constraint.
- **Assessment memo (Stage D):** documented the architectural conclusion and limits.

This adds a key architectural distinction:

- **low-rank PEFT generalizes via factor-based reuse** (Ring 1),
- **full checkpoint deltas generalize via summary-based reuse** (Ring 2),
- the **workflow survives, but the representation path differs**.

Evidence and QA remain central in both paths, and merge execution remains out of scope for Ring 2.

Orthogonal generalization map:

| Axis | Validation | Result |
|------|------------|--------|
| Scenario generality | Routing pilot (§8) | Confirmed (`merge` vs `routing` from same spectral substrate) |
| Artifact-class generality (low-rank PEFT) | Ring 1 (§9) | Confirmed (`LoRA` vs `LoHa` via thin extraction shim) |
| Representation-path generality (full checkpoints) | Ring 2 (this section) | Confirmed in bounded scope (`factor-based` vs `summary-based` path) |

**Revised assessment status: CONFIRMED on three orthogonal axes (bounded).** Broader checkpoint-delta triage is now plausible, but still narrow: one backbone family, small encoder checkpoints, CPU-only, and no merge execution.

Practical bounded claim: the substrate generalizes across artifact classes and across downstream decisions, with aggregation and policy as the main scenario-specific seams. In bounded scope, this now supports evidence-aware triage for both adapter inventories and full checkpoint inventories on shared-base small encoder models, with explicit same-task, same-family, and cross-task distinctions.

Ring 2 docs: `docs/design/ring2_stage_a_checkpoint_delta_representation.md`, `docs/design/ring2_stage_b_representation_c_audit.md`, `docs/design/ring2_stage_c_guardrail_triage.md`, `docs/design/ring2_stage_d_assessment_memo.md`.

## 12. Cross-Artifact Compatibility Portability (2026-03-31)

The cross-artifact compatibility research program (sidecar n76-n80) formally tested which compatibility signals transfer across artifact classes (LoRA, LoHa, checkpoint delta) and which are representation-local.

Key findings:

- **Two strong invariants** recur across all three classes: evidence regime gating and conservative candidate narrowing. Both are workflow-level signals, not structural-metric-level.
- **Two moderate invariants** recur where testable: task-relation ordering (same_task > same_family > cross_task) and same-family intermediate status.
- **No structural metric is fully portable.** The sidecar's strongest signal (V-module dimensionality ratio, d=3.36) is representation-locked to factorized artifacts. Compatibility scores, risk labels, and stable rank have the same names across classes but different scales and semantics.
- **Triage is the only cross-artifact decision scenario.** Merge and routing remain operationally restricted to LoRA.
- **Three-layer framework:** Layer 1 (artifact-invariant: evidence gating, task ordering), Layer 2 (representation-local: factor geometry, summary profiles), Layer 3 (decision-dependent: merge/routing/triage).

This confirms H3: artifact broadening preserves workflow shape more than feature parity. The cross-artifact substrate is real but narrow -- it consists of workflow-level and task-relational signals, not structural measurements.

Product relevance summary: `docs/strategy/cross_artifact_product_relevance_summary.md`. Sidecar notes: n76-n80 and structured outputs in `sidecar/results/cross_artifact_portability/`.

## 13. Aggregation-Sensitive Compatibility (2026-03-31)

The aggregation-sensitive compatibility research program (sidecar n81-n85) tested whether different aggregation rules produce genuinely different operational judgments from the same structural evidence.

Key findings:

- **Aggregation is not presentation.** Only 2/12 cases are aggregation-invariant (both cross-task with clear QA). The remaining 10 produce different operational labels under different aggregation rules.
- **The routing gradient is distributional-only.** Worst-case collapses confusable, moderate, and separable cases to the same label. Only distributional aggregation reveals the three-tier ordering.
- **QA can override structure.** The pair with the highest compatibility score (0.892) is blocked by QA-dominant aggregation. Evidence status is a genuinely independent dimension.
- **The hybrid is the richest family.** QA-gated distributional preserves both evidence constraints and structural gradation. It is the correct general-purpose default.
- **Five stable patterns** are predictable from two features: QA regime and task relation.

This reinforces the cross-artifact finding (§12): the stable layer is workflow design, not numeric outputs. Aggregation family selection should be decision-context-dependent: merge → worst-case, routing → distributional, triage → QA-dominant, general-purpose → hybrid.

Route 2 summary: `docs/strategy/aggregation_sensitive_route2_summary.md`. Sidecar notes: n81-n85 and structured outputs in `sidecar/results/aggregation_sensitive_compatibility/`.

---

## 14. Behavioral Route 2 Bridge (2026-04-01)

The behavioral Route 2 bridge program (sidecar n86-n92) tested whether broadened Route 2 compatibility profiles have distinct example-level behavioral signatures.

Key findings:

- **Four of five profiles have distinct behavioral footprints.** Aggregation-invariant safe, worst-case collapse, cross-task separable, and QA-dominant review all produce identifiable behavioral patterns. Same-family optional is behaviorally indistinguishable from aggregation-invariant safe.
- **Three-tier behavioral model.** Tier 1 (no pathology): safe/optional, neither-source <2%. Tier 2 (localized pathology): collapse/cross-task, neither-source ~14%. Tier 3 (stasis): QA review, shared failure 65%.
- **Collapse vs contamination is the key behavioral distinction.** Same failure rate (~14% neither-source) but different channels: worst-case collapse produces uncertainty (28-30 confidence collapses), cross-task contamination produces confident wrong predictions (23 high-confidence wrong). This behaviorally justifies decision-context-dependent aggregation.
- **Evidence gating is behaviorally grounded.** QA-dominant aggregation identifies behavioral stasis (nothing to preserve or destroy), not structural weakness.
- **Routing-confusability is structural, not behavioral.** In the merge setting, routing-confusable cases look safe-like. The behavioral signature may only appear in actual routing scenarios.

This closes the behavioral loop on Route 2: the framework is grounded in model behavior, not only structural measurement.

Behavioral summary: `docs/strategy/behavioral_route2_summary.md`. Sidecar notes: n86-n92 and structured outputs in `sidecar/results/behavioral_route2_bridge/`.

---

*This assessment is grounded in the vnext codebase as of 2026-04-01. Module paths, class names, and function signatures are verified against the current source. Routing pilot validation 2026-03-29. Ring 1 PEFT generalization 2026-03-30. Ring 2 checkpoint-delta generalization 2026-03-30. Decision-dependent compatibility sidecar update 2026-03-31. Cross-artifact portability program 2026-03-31. Aggregation-sensitive compatibility program 2026-03-31. Behavioral Route 2 bridge 2026-04-01.*
