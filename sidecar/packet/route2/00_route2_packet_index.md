# Route 2 Packet — Broadened Compatibility Science

**Date:** March 2026
**Status:** Complete for current scope. DeBERTa adjudication is the next empirical step; it does not affect Route 2 claims directly.

---

## What this is

A self-contained packet summarizing Gradience's Route 2 research program — the extension of compatibility science beyond LoRA merge into routing, triage, additional artifact classes, and behavioral grounding. Designed for a collaborator, external workflow owner, or future re-entry after time away.

**You do not need to read the full sidecar.** This packet contains the key documents and figures. If you want the mechanism-ladder story (why merges fail), see the main research packet (`../00_packet_index.md`). This packet answers the next question: what happens when we extend compatibility science beyond merge?

---

## Background (30 seconds)

Gradience is a Python library for spectral analysis of LoRA adapters. Its stable product detects task boundaries, gates merge candidates by behavioral evidence, and produces action-plan-based inventory triage. It works: zero false positives across 5 inventories and 53+ pairs.

Route 2 asked four questions:
1. Does the substrate extend beyond merge? (Yes — routing, triage.)
2. Does it extend beyond LoRA? (Yes — LoHa, checkpoint deltas, at the workflow level.)
3. Is aggregation a real computational step? (Yes — different rules, different operational judgments.)
4. Do these structural distinctions show up in model behavior? (Yes — four of five profiles have distinct behavioral signatures.)

---

## Packet contents

Read in this order.

### Core documents

| # | Document | What it is | Length |
|---|----------|------------|--------|
| 1 | [`01_route2_orientation.md`](01_route2_orientation.md) | **Orientation.** The broadened substrate scope, what generalized, what didn't. Start here. | ~800 words |
| 2 | [`../../notes/n93_route2_synthesis.md`](../../notes/n93_route2_synthesis.md) | **Synthesis memo.** The four-layer Route 2 architecture, how each layer depends on the ones before it, settled claims. The theoretical narrative. | ~2000 words |
| 3 | [`../../../docs/strategy/cross_artifact_product_relevance_summary.md`](../../../docs/strategy/cross_artifact_product_relevance_summary.md) | **Cross-artifact portability.** What transfers across LoRA, LoHa, and checkpoint deltas. Three-layer framework: artifact-invariant / representation-family / decision-dependent. | ~500 words |
| 4 | [`../../../docs/strategy/cross_artifact_stability_summary.md`](../../../docs/strategy/cross_artifact_stability_summary.md) | **Cross-artifact stability check.** Which portability claims stayed stable vs panel-sensitive under local perturbation. | ~400 words |
| 5 | [`../../../docs/strategy/aggregation_sensitive_route2_summary.md`](../../../docs/strategy/aggregation_sensitive_route2_summary.md) | **Aggregation sensitivity.** Five patterns. Decision-context-dependent family selection: merge = worst-case, routing = distributional, triage = QA-dominant, general = hybrid. | ~500 words |
| 6 | [`../../../docs/strategy/aggregation_stability_summary.md`](../../../docs/strategy/aggregation_stability_summary.md) | **Aggregation stability check.** Confirms aggregation as seam, QA-dominant distinctness, and worst-case collapse under local perturbation. | ~400 words |
| 7 | [`../../../docs/strategy/mixed_evidence_triage_summary.md`](../../../docs/strategy/mixed_evidence_triage_summary.md) | **Mixed-evidence triage stress test.** Follow-on soft-middle pass: QA-dominant remains coherent; same-family optional stays review-like with guardrails. | ~350 words |
| 8 | [`../../../docs/strategy/behavioral_route2_summary.md`](../../../docs/strategy/behavioral_route2_summary.md) | **Behavioral bridge.** Three-tier behavioral model. The collapse/contamination mode split. Same-family optional is safe-like. | ~500 words |
| 9 | [`../../../docs/strategy/collapse_vs_contamination_summary.md`](../../../docs/strategy/collapse_vs_contamination_summary.md) | **Collapse/contamination replication.** Follow-on bounded replication pass reinforcing channel separation with guardrails. | ~350 words |
| 10 | [`../../../docs/strategy/route2_claims_ladder_summary.md`](../../../docs/strategy/route2_claims_ladder_summary.md) | **Claims ladder synthesis.** Stable/moderate/thin/local calibration for Route 2 communication and scope discipline. | ~500 words |

### Substrate evidence

| # | Document | What it is |
|---|----------|------------|
| 11 | [`../../../docs/strategy/broadened_substrate_scope.md`](../../../docs/strategy/broadened_substrate_scope.md) | **Route 2 scope checkpoint.** Stable bounded scope statement for broadened substrate claims. |
| 12 | [`../../../docs/strategy/ring1_peft_generalization_results.md`](../../../docs/strategy/ring1_peft_generalization_results.md) | **Ring 1.** LoHa through the full pipeline via ~160-line shim. Zero core code changes. |
| 13 | [`../../../docs/design/ring2_stage_d_assessment_memo.md`](../../../docs/design/ring2_stage_d_assessment_memo.md) | **Ring 2.** Checkpoint deltas via summary-based representation. Triage works; merge out of scope. |
| 14 | [`../../../docs/routing-pilot-results.md`](../../../docs/routing-pilot-results.md) | **Routing pilot.** ~370 lines, zero core changes. Same substrate, different policy layer. |

### Alpha workflow

| # | Document | What it is |
|---|----------|------------|
| 15 | [`../../../docs/examples/checkpoint-triage-alpha-workflow.md`](../../../docs/examples/checkpoint-triage-alpha-workflow.md) | **Checkpoint triage alpha walkthrough.** Canonical use path, artifacts, and interpretation guide. |
| 16 | [`../../../docs/strategy/checkpoint_triage_alpha_scope.md`](../../../docs/strategy/checkpoint_triage_alpha_scope.md) | **Alpha scope contract.** Explicit boundaries for the Route 2 checkpoint triage workflow. |
| 17 | [`../../../field_trials/checkpoint_inventory_t02/README.md`](../../../field_trials/checkpoint_inventory_t02/README.md) | **Checkpoint triage alpha — mini-product README.** What it's for, how to run it, what outputs mean, example inventory, adaptation guide. |
| 18 | [`../../../field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html`](../../../field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html) | **Example HTML report.** The canonical T02 output — open in a browser. |

### Figures

| Figure | What it shows |
|--------|--------------|
| [`figures/behavioral_route2_profile_matrix.svg`](figures/behavioral_route2_profile_matrix.svg) | **Three-tier behavioral separation.** Neither-source %, confidence collapse, and high-confidence wrong by Route 2 profile. The clearest single picture of why profiles are not just labels. |
| [`figures/decision_dependent_aggregation_matrix.svg`](figures/decision_dependent_aggregation_matrix.svg) | **Aggregation is computational.** Same structural evidence, four aggregation rules, different operational labels. Shows why aggregation family selection must be decision-dependent. |
| [`../../figures/collapse_vs_contamination_replication_matrix.svg`](../../figures/collapse_vs_contamination_replication_matrix.svg) | **Behavioral channel replication matrix.** Nearby case/slice perturbations preserve collapse-vs-contamination confidence-channel separation. |

---

## The one-paragraph version

Gradience's spectral substrate — task-boundary detection, evidence gating, inventory triage — generalizes beyond LoRA merge. Three experiments validated three generalization axes: scenario (routing pilot, zero core changes), artifact class (LoHa via shim, checkpoint deltas via summary representation), and representation path (factor-based for PEFT, summary-based for full checkpoints). Within this broadened scope, four research programs established that: (1) the same structural evidence means different things under different decisions, (2) portable compatibility signals live at the workflow level not the metric level, (3) aggregation is a computational step that selects which structural truths become operative, and (4) four of five Route 2 compatibility profiles have distinct behavioral signatures. Stability checks now strengthen this picture: aggregation as seam, QA-dominant distinctness, and worst-case collapse all survived local perturbation; mixed-evidence triage passes showed the soft middle remains coherent with guardrails; and a follow-on collapse-vs-contamination replication pass strengthened the channel distinction in bounded merge-facing settings. The checkpoint triage alpha workflow is the first working product of this research.

---

## Where to go deeper

- **Mechanism ladder** (why merges fail): `../00_packet_index.md`
- **Settled/open/next dashboard**: `../../notes/n69_settled_open_next.md`
- **Full sidecar index**: `../../README.md`
- **Project map** (stable vs alpha vs experimental vs research): `../../../docs/00_start_here/project-map.md`
- **Route 2 claims ladder summary**: `../../../docs/strategy/route2_claims_ladder_summary.md`
- **Route 2 communication policy summary**: `../../../docs/strategy/route2_communication_policy_summary.md`
- **Collapse vs contamination summary**: `../../../docs/strategy/collapse_vs_contamination_summary.md`
- **Boundaries and non-generalizations**: `../../../docs/boundaries-and-non-generalizations.md`
- **All structured data**: `../../results/` (decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge, and route2_stress_tests subdirectories)
