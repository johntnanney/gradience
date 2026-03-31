# Perturbed Signal Summary (Stage C)

## Strongly present

- `qa_evidence_regime`
- `conservative_narrowing`

These remain strong across all three artifact classes under perturbation.

## Moderately present

- `same_task_vs_cross_task_separation`

This remains visible where testable (LoRA and checkpoint delta), with LoHa still indeterminate.

## Weakened under perturbation

- `same_family_intermediate_behavior`

Checkpoint delta still shows strict intermediate ordering, but perturbed LoRA does not (`same_family > same_task > cross_task` in this local panel slice).

## Still inconclusive

- `near_miss_middle_states`

Near-miss remains clearly validated in LoRA only; LoHa and checkpoint delta still do not provide cross-artifact confirmation.

## Local signal rerun takeaway

Representation-local structural metrics remain local. No new cross-artifact structural invariant appears in the perturbed rerun.
