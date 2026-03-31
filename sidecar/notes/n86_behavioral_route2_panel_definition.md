# n86 -- Behavioral Route 2 Panel Definition

**Type:** panel definition
**Date:** 2026-03-31
**Program:** Behavioral Route 2 Bridge
**Stage:** A
**Depends on:** n59-n66 (output example semantics), n70-n74 (decision-dependent compatibility), n81-n85 (aggregation-sensitive compatibility), n76-n80 (cross-artifact compatibility)
**Status:** complete

---

## Question

What cases should form the behavioral panel for testing whether broadened Route 2 compatibility profiles have distinct example-level signatures?

---

## Design rationale

The panel reuses the 8 cases from the Output Example Semantics program (n59-n66), which have full per-example prediction data (500 examples each, with labels, predictions from both sources and merged model, and softmax confidence scores). No new merges or evaluations are required.

The reuse is not arbitrary. The n59-n66 cases span the same behavioral contrasts that Route 2 compatibility profiles are designed to distinguish — safe retained, fragile, cross-task, near-miss — but they were originally analyzed only through the merge-failure lens. This program reinterprets them through the broadened Route 2 profile lens, asking whether the structural/aggregation/decision-relative categories map onto distinct behavioral footprints.

The panel adds one case type not directly available from n59-n66: the QA-dominant review case. AN-01 (both sources deeply weak, near chance) serves this role — it is the closest available case to a "structurally present but evidence-absent" scenario.

---

## Route 2 profile reinterpretation

Each n59-n66 case is mapped to the Route 2 profile it most closely represents:

| Original class | Route 2 profile | Why |
|----------------|----------------|-----|
| safe_retained | Aggregation-invariant safe | All aggregation families agree: safe to merge, structurally sound, QA clear. This is the floor of the aggregation problem. |
| near_miss | Same-family optional / near-miss-like | Structurally marginal but behaviorally safe-like. Would be QA-constrained in strict mode. The distributional gradient matters here but worst-case collapses it. |
| fragile | Worst-case-collapse | A single weak source or bad layer drives the verdict. Worst-case aggregation correctly identifies the risk; distributional might underweight it. |
| control | Cross-task separable | All families agree on exclusion. Aggregation-invariant at the opposite end from safe. |
| anchor | QA-dominant review | Both sources near chance — the evidence status is the binding constraint, not the structural compatibility. QA-dominant aggregation would flag this as under-evidenced regardless of structural measurement. |

---

## Panel

| Case ID | Original class | Route 2 profile | Artifact class | Backbone | Task | Δ vs best | Source A | Source B | Merged | Examples |
|---------|---------------|-----------------|---------------|----------|------|-----------|----------|----------|--------|----------|
| SR-01 | safe_retained | aggregation_invariant_safe | LoRA | DistilBERT | irony (binary) | -0.006 | 0.632 | 0.620 | 0.626 | 500 |
| SR-02 | safe_retained | aggregation_invariant_safe | LoRA | BERT | hate (binary) | +0.028 | 0.514 | 0.588 | 0.616 | 500 |
| NM-01 | near_miss | same_family_optional | LoRA | DistilBERT | irony (binary) | -0.012 | 0.632 | 0.618 | 0.620 | 500 |
| NM-02 | near_miss | same_family_optional | LoRA | BERT | hate (binary) | -0.002 | 0.514 | 0.498 | 0.572 | 500 |
| FR-01 | fragile | worst_case_collapse | LoRA | BERT | emotion (4-class) | -0.088 | 0.752 | 0.204 | 0.664 | 500 |
| FR-02 | fragile | worst_case_collapse | LoRA | BERT | emotion (4-class) | -0.088 | 0.752 | 0.136 | 0.664 | 500 |
| CT-01 | control | cross_task_separable | LoRA | BERT | ag_news/hate (cross) | -0.096 | 0.922 | n/a | 0.826 | 500 |
| AN-01 | anchor | qa_dominant_review | LoRA | BERT | emotion (4-class) | -0.002 | 0.204 | 0.136 | 0.202 | 500 |

### Coverage check

| Case type (spec §6) | Required | Met by |
|---------------------|----------|--------|
| Aggregation-invariant safe | ≥1 | SR-01, SR-02 |
| Same-family optional / near-miss-like | ≥1 | NM-01, NM-02 |
| Routing-confusable | ≥1 | NM-01 (high overlap, same-task, confusable under distributional) |
| Worst-case-collapse | ≥1 | FR-01, FR-02 |
| QA-dominant review / blocked | ≥1 | AN-01 |
| Cross-task separable (optional) | ≥1 | CT-01 |

Note on routing-confusable: NM-01 serves double duty. Under routing-oriented distributional aggregation, same-task pairs with high overlap and similar performance are the canonical confusable case — the routing system cannot easily distinguish the sources. NM-01 (both sources ~0.62 accuracy, same task, same backbone) is exactly this. The behavioral question is whether routing-confusability produces confusion-like behavior rather than merge-fragility-like behavior.

---

## Data availability per case

| Case ID | Labels | Source A preds | Source B preds | Merged preds | Confidence/logits | Per-example JSON |
|---------|--------|---------------|---------------|-------------|-------------------|-----------------|
| SR-01 | yes | yes | yes | yes | yes | `predictions/predictions_SR-01.json` |
| SR-02 | yes | yes | yes | yes | yes | `predictions/predictions_SR-02.json` |
| NM-01 | yes | yes | yes | yes | yes | `predictions/predictions_NM-01.json` |
| NM-02 | yes | yes | yes | yes | yes | `predictions/predictions_NM-02.json` |
| FR-01 | yes | yes | yes | yes | yes | `predictions/predictions_FR-01.json` |
| FR-02 | yes | yes | yes | yes | yes | `predictions/predictions_FR-02.json` |
| CT-01 | yes | yes | no (cross-task) | yes | yes | `predictions/predictions_CT-01.json` |
| AN-01 | yes | yes | yes | yes | yes | `predictions/predictions_AN-01.json` |

All prediction files are in `sidecar/results/example_semantics/predictions/`. Each contains 500 examples with ground truth labels, per-model predictions, and softmax probability vectors.

CT-01 has no source B predictions because it is a cross-task merge (ag_news × hate); source B was trained on hate, evaluated on ag_news — its predictions are not meaningful as a behavioral reference on the ag_news test set.

---

## Success criteria assessment

- Panel covers at least 5 distinct Route 2 compatibility situations: **met** (6 types covered)
- Each case has enough outputs to support example-level comparison: **met** (500 examples × 8 cases, all with labels + predictions + confidence)
- Panel is small enough to analyze carefully: **met** (8 cases, same as n59-n66)

---

## Output artifacts

- `sidecar/notes/n86_behavioral_route2_panel_definition.md` (this note)
- `sidecar/results/behavioral_route2_bridge/panel_table.json`
- `sidecar/results/behavioral_route2_bridge/panel_table.md`
