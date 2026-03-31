# n75 — Cross-Artifact Compatibility (Route 2 Workstream 3)

**Type:** synthesis note  
**Date:** 2026-03-31  
**Depends on:** Ring 1 results, Ring 2 Stages A-D, adapter and checkpoint triage artifacts  
**Status:** initial bounded completion

---

## Question

Which compatibility signals transfer across artifact classes, and which remain representation-local?

Artifact classes in this pass:

- LoRA,
- LoHa (shimmed),
- full checkpoint delta (summary representation).

---

## Panel and constraints

This pass uses a bounded CPU panel built from existing artifacts:

- adapter triage/routing evidence (`field_trials/targeted_confirmation_same_family/inventory_t01/preflight` and prior adapter field trials),
- Ring 1 LoHa pilot (`experiments/peft_ring1/`),
- checkpoint triage T02 (`field_trials/checkpoint_inventory_t02/`).

Constraint: LoHa coverage is same-task only in this pass.

---

## Output tables

- `sidecar/results/cross_artifact_compatibility/shared_vs_specific_table.json`
- `sidecar/results/cross_artifact_compatibility/shared_vs_specific_table.md`

---

## Findings

### Shared signals (transfer observed)

1. Evidence bootstrap and QA gating remain operationally central.
2. Pairwise structural comparison remains useful for narrowing.
3. Relation tags (`same_task`, `same_family`, `cross_task`) remain useful where covered.
4. Decision dependence still follows the same architecture: shared measurement/diagnosis with scenario-specific aggregation and policy.

### Representation-specific signals

1. Representation path is artifact-local:
- LoRA: native factors,
- LoHa: shimmed factors/materialized deltas,
- checkpoints: summary-based deltas.

2. Merge execution capability is artifact-local:
- native for LoRA,
- unestablished for LoHa,
- out of scope for checkpoint deltas.

### Open gaps

- LoHa cross-task and same-family behavior is not yet validated in this panel.
- Cross-artifact parity should not be inferred from one LoHa pilot and two checkpoint triage trials.

## Product relevance filter

### Safe to expose in stable workflow language (now)

1. Evidence bootstrap and QA gating as first-class triage steps.
2. Same-task / same-family / cross-task distinction where panel coverage exists.
3. Pairwise compatibility as a narrowing aid (not as a standalone decision).
4. Explicit bounded-scope language for checkpoint triage.

### Keep research-only (for now)

1. Any merge-execution claims outside native LoRA.
2. Cross-artifact parity assumptions beyond tested panels.
3. Representation-specific internal metrics as product-facing primitives.
4. Scenario-level conclusions where only one artifact class has coverage.

--- 

## Route 2 implication

The substrate-level claim that currently holds is:

- shared measurement and triage logic can transfer across artifact classes,
- but representation adapters and execution claims must remain artifact-specific.

That supports disciplined broadening: broaden workflow classes where QA, pairwise narrowing, and follow-through remain useful, while keeping execution support claims explicitly bounded.
