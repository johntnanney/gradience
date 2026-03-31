# n93 -- Route 2 Synthesis: Decision-Dependent Compatibility Science

**Type:** synthesis
**Date:** 2026-04-01
**Depends on:** n70-n74 (decision-dependent), n76-n80 (cross-artifact), n81-n85 (aggregation-sensitive), n86-n92 (behavioral bridge), n93-n97 (cross-artifact stability), n98-n102 (aggregation stability), n103-n107 (mixed-evidence triage perturbation), and the mechanism ladder (n67 §§1-10)
**Status:** The single current-best account of Route 2 findings. Companion to n67 (mechanism-ladder synthesis).

---

## What this document is

This note synthesizes the four Route 2 research programs completed since the mechanism ladder was established. n67 is the mechanism-ladder synthesis — it answers "what causes catastrophic merge failure and how does the mechanism work?" This note answers a different question:

> **What happens when we extend compatibility science beyond merge, beyond LoRA, and beyond structural measurement?**

The answer has four layers, each built by a distinct research program:

1. **Decision-dependent compatibility** (n70-n74): the same structural relation means different things under merge, routing, and triage.
2. **Cross-artifact compatibility** (n76-n80): portable compatibility signals live at the workflow level, not the metric level.
3. **Aggregation-sensitive compatibility** (n81-n85): aggregation is a computational step that selects which structural truths become operative.
4. **Behavioral bridge** (n86-n92): four of five Route 2 profiles have distinct behavioral signatures, grounding the framework in model behavior.

Each layer depends on the ones before it. Together they form a coherent picture of what compatibility means in the broadened setting.

---

## 1. The decision-dependent layer (n70-n74)

The mechanism ladder was developed in a merge-only context. Route 2 asks: does the same structural evidence mean the same thing under different decisions?

A shared 9-case panel was defined across merge, routing, and triage. A stack audit showed the first practical divergence appears at aggregation, not at measurement — the structural substrate is shared, but the policy layer that acts on it is scenario-specific.

**The central finding:** the substrate generalizes across artifact classes and downstream decisions, with aggregation and policy as the main scenario-specific seams. A compact six-profile decision taxonomy was stress-tested on a second triage substrate, supporting profile stability.

This does not replace the mechanism ladder. It adds the cross-decision layer that explains when the same geometry should produce different actions.

---

## 2. The cross-artifact layer (n76-n80)

The decision-dependent program raised a question it could not answer: are the compatibility patterns observed across artifact classes genuinely shared, or are they different local phenomena that merely resemble each other?

A 9-case panel spanning LoRA, LoHa, and checkpoint delta was audited for signal portability.

**The central finding:** cross-artifact compatibility exists primarily at the level of workflow invariants, while the strongest structural discriminators remain representation-local.

Specifically:

- **Two strong invariants** recur across all three artifact classes: evidence regime gating and conservative candidate narrowing. Both are workflow-level — they operate on evidence metadata and triage policy, not factor geometry.
- **Two moderate invariants** recur where testable: task-relation ordering (same_task > same_family > cross_task) and same-family intermediate status.
- **No structural metric is fully portable.** The V-module dimensionality ratio (d=3.36) requires factor-level subspace geometry. Compatibility scores, risk labels, and stable rank share names across classes but have different scales and semantics.
- **Triage is the only cross-artifact decision scenario.** Merge and routing remain operationally restricted to factorized artifacts.

The framework that emerges has three layers:

| Layer | Content | Portability |
|-------|---------|-------------|
| 1: Artifact-invariant | Evidence gating, narrowing, task-relation ordering | References across artifact classes without qualification |
| 2: Representation-family | Factor geometry (LoRA/LoHa), summary profiles (checkpoint delta) | Must be qualified by artifact class |
| 3: Decision-dependent | Merge (worst-case), routing (distributional), triage (QA-gate-first) | Same Layer 2 signal has different implications per decision |

**Practical implication:** product language can safely reference the workflow shape across artifact classes. It should not reference structural measurements, numeric scores, or merge strategies as cross-artifact primitives.

---

## 3. The aggregation layer (n81-n85)

The cross-artifact program established that portable compatibility lives at the workflow level. The aggregation program asks: within a single artifact class, do different aggregation rules produce genuinely different operational judgments from the same structural evidence?

A 12-case panel with matched QA-regime pairs (blocked vs clear, same task relations) isolated the aggregation effect from the structural effect. Four aggregation families were applied: worst-case, distributional, QA-dominant, and QA-gated distributional (hybrid).

**The central finding:** aggregation is not a presentation layer. It is a computational step that selects which structural truths become operative.

The evidence:

- **Only 2/12 cases are aggregation-invariant** (both cross-task with clear QA).
- **The routing gradient is distributional-only.** Worst-case collapses confusable, moderate, and separable to one label.
- **QA can override the strongest structural signal.** The highest-compatibility pair (0.892) is blocked by QA because both sources lack evidence.
- **The hybrid is the richest family.** QA-gated distributional never produces less information than any single family.
- **Five stable patterns** are predictable from two features: QA regime and task relation.

**Practical implication:** aggregation family selection should be decision-context-dependent. Merge → worst-case. Routing → distributional. Triage → QA-dominant. General-purpose → hybrid. This is not a preference — it follows from what each decision context optimizes for.

**Stability update:** the local robustness pass (n98-n102) retained the core aggregation conclusions under disciplined panel perturbation. In particular, three claims held: aggregation as a seam, QA-dominant as distinct family, and worst-case collapse of routing gradation. A targeted mixed-evidence triage pass (n103-n107) then stress-tested the soft middle and found it coherent with guardrails: same-family optional states remain review-like, while fine-grained review thresholds remain explicitly bounded.

---

## 4. The behavioral layer (n86-n92)

The three preceding programs established the structural, cross-artifact, and aggregation architecture of Route 2 compatibility. The behavioral bridge asks: do these architectural distinctions correspond to observable differences in model behavior on examples?

An 8-case panel with 4,000 examples — reusing the validated n59-n66 data — was reinterpreted through 5 Route 2 profiles.

**The central finding:** broadened Route 2 compatibility profiles have behavioral reality. Four of five profiles correspond to distinct, identifiable behavioral signatures.

### The three-tier behavioral model

| Tier | Profiles | Signature | Key metrics |
|------|----------|-----------|-------------|
| 1: No pathology | Aggregation-invariant safe, same-family optional | Stable preservation, no novel failure | Neither-source <2%, confidence collapse = 0 |
| 2: Localized pathology | Worst-case collapse, cross-task separable | Concentrated failure, two channels | Neither-source ~14%; collapse (conf <0.4) vs contamination (conf >0.7) |
| 3: Stasis | QA-dominant review | Shared failure, no merge-induced change | Shared failure 65%, neither-source 7%, all pathology = 0 |

### The collapse/contamination mode split

The most operationally important finding. Worst-case collapse and cross-task contamination produce the same rate of novel failure (~14% neither-source) but through different channels:

- **Collapse** → the model becomes uncertain (28-30 confidence collapses, 0 high-confidence wrong). The model knows it doesn't know.
- **Contamination** → the model becomes confidently wrong (23 high-confidence wrong, 3 confidence collapses). The model doesn't know it doesn't know.

This is the behavioral confirmation that different aggregation strategies are needed for different decision contexts — they track different failure channels.

### Same-family optional is behaviorally safe-like

Same-family optional (near-miss) cases are indistinguishable from aggregation-invariant safe on all discriminating metrics. The structural distinction (retained vs near-miss) matters for evidence gating, not for behavioral outcome. QA constraints on these cases are about evidence gaps, not behavioral risk.

### Routing-confusability has no merge-visible behavioral signature

Routing-confusable cases look safe-like in the merge setting. The behavioral signature of routing-confusability likely only appears in actual routing scenarios. This is consistent with the finding that routing and merge are different decision contexts.

---

## 5. How the four layers fit together

The Route 2 programs form a coherent stack:

```
Layer 4 — Behavioral bridge (n86-n92)
  "The profiles correspond to observable behavior"
    ↓ grounds
Layer 3 — Aggregation-sensitive (n81-n85)
  "Different aggregation rules select different operational truths"
    ↓ explains
Layer 2 — Cross-artifact (n76-n80)
  "Portable signals are workflow-level, not metric-level"
    ↓ constrains
Layer 1 — Decision-dependent (n70-n74)
  "Same structure means different things under different decisions"
    ↓ extends
Mechanism ladder (n67 §§1-10)
  "V-module pathology × readout gating → catastrophic failure"
```

Each layer answers a question the previous layer raised:
- The mechanism ladder says *how* merges fail → the decision-dependent layer asks *does this extend beyond merge?*
- The decision-dependent layer says *yes, but through different policies* → the cross-artifact layer asks *does this extend beyond LoRA?*
- The cross-artifact layer says *at the workflow level, yes* → the aggregation layer asks *is the workflow-level distinction real or just presentation?*
- The aggregation layer says *real — aggregation is computational* → the behavioral bridge asks *and does it show up in model behavior?*
- The behavioral bridge says *yes — four of five profiles have distinct behavioral signatures.*

The stack is now complete in its current scope. The remaining open questions (DeBERTa generalization, routing-confusability behavioral signature, broader panel validation) require either GPU compute or a fundamentally different experimental setup.

---

## 6. Settled Route 2 claims

These claims are supported by the converging evidence from all four programs:

1. **Decision-context-dependent aggregation is structurally and behaviorally justified.** Merge, routing, and triage require different aggregation strategies because they optimize for different failure channels (concentrated pathology, confusability gradient, evidence presence).

2. **Cross-artifact compatibility is workflow-level, not metric-level.** Evidence gating and conservative narrowing transfer across artifact classes. No structural metric is fully portable.

3. **Aggregation is computational, not presentational, and now locally stability-checked.** Different rules produce genuinely different operational judgments from the same evidence, and the key distinctions survived local perturbation (n98-n102). The appropriate rule is a function of what the practitioner is deciding.

4. **Same-family optional is behaviorally safe-like.** The structural near-miss classification does not predict behavioral pathology. QA constraints on these cases are about evidence, not risk.

5. **Collapse and contamination are distinct operational failure channels.** Same failure rate (~14% neither-source), different confidence patterns, different operational consequences. Collapse is recoverable (the model signals its uncertainty); contamination is dangerous (the model conceals its error).

6. **Evidence gating identifies behavioral stasis.** QA-dominant aggregation does not over-conservatively block structurally sound cases — it correctly identifies cases where structural measurement has no behavioral substrate.

7. **Route 2 claims ladder now has two robustness anchors.** Cross-artifact portability claims were stress-tested in n93-n97; aggregation-sensitive claims were stress-tested in n98-n102. This strengthens route-level confidence while preserving scope guardrails.

8. **Soft-middle triage structure survives mixed-evidence weighting.** The additional triage-weighted perturbation (n103-n107) supports guarded language for review and optional states: QA-dominant remains coherent; same-family optional remains review/clear-leaning rather than collapse-like; exact thresholding remains non-canonical.

9. **Aggregation claims are now seam-stable, threshold-guarded, and behaviorally grounded.** Seam-level claims (aggregation as decision seam, family distinctness, worst-case collapse) are reinforced by stability checks; threshold-level claims remain explicitly bounded; behavioral bridge results explain why these aggregation distinctions are operationally meaningful rather than architectural artifacts.

---

## 7. Route 2 boundaries

What Route 2 has *not* established:

- **Routing-confusability behavioral signature.** The structural confusability gradient does not produce confusion-like behavior in the merge setting. Whether it does in a routing setting is untested.
- **Generalization beyond the current panel.** The three-tier behavioral model and the five aggregation-sensitive patterns describe the current evidence base. Whether they generalize to other backbones, tasks, or artifact classes is open.
- **Numeric thresholds.** The <2% vs ~14% neither-source boundary, the 5 aggregation patterns, and the 3 behavioral tiers are descriptive of this panel. They should not be hardcoded as calibrated boundaries.
- **Non-LoRA behavioral data.** Per-example behavioral data exists only for LoRA. Whether behavioral tiers transfer across artifact classes is unknown.

---

## 8. Relationship to n67

n67 (Where the Research Stands) is the mechanism-ladder synthesis. It answers: what causes catastrophic merge failure?

This note (n93) is the Route 2 synthesis. It answers: what happens when we extend compatibility science beyond merge?

Together they constitute the project's theoretical account:

- **n67** = the mechanism (why merges fail)
- **n93** = the broader framework (what compatibility means across decisions, artifacts, and aggregation strategies)
- **n69** = the project dashboard (what is settled, open, and next)

---

## 9. Deliverables

| Program | Notes | Data | Product summary |
|---------|-------|------|----------------|
| Decision-dependent | n70-n74 | `results/decision_dependent_compatibility/` | `docs/strategy/decision_dependent_compatibility_implications.md` |
| Cross-artifact | n76-n80 | `results/cross_artifact_portability/` | `docs/strategy/cross_artifact_product_relevance_summary.md` |
| Cross-artifact stability | n93-n97 | `results/route2_stability/cross_artifact/` | `docs/strategy/cross_artifact_stability_summary.md` |
| Aggregation-sensitive | n81-n85 | `results/aggregation_sensitive_compatibility/` | `docs/strategy/aggregation_sensitive_route2_summary.md` |
| Aggregation stability | n98-n102 | `results/route2_stability/aggregation/` | `docs/strategy/aggregation_stability_summary.md` |
| Aggregation mixed-evidence triage perturbation | n103-n107 | `results/route2_stability/aggregation_mixed_evidence/` | `docs/strategy/aggregation_mixed_evidence_summary.md` |
| Behavioral bridge | n86-n92 | `results/behavioral_route2_bridge/` | `docs/strategy/behavioral_route2_summary.md` |
