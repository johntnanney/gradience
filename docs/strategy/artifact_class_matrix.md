# Artifact Class Matrix (Route 2)

Date: 2026-03-31  
Status: bounded consolidation snapshot

## Matrix

| Artifact class | Representation for analysis | Current support level | Single-artifact QA | Pairwise comparison | Inventory triage | Merge execution | Validation basis |
|---|---|---|---|---|---|---|---|
| LoRA | Native low-rank factors (`A`, `B`) | stable | yes | yes | yes | yes | Core field trials (`field_trials/inventory_*`), production merge path |
| LoHa | Shimmed factor/materialized low-rank representation | experimental (validated for audit/triage path) | yes (via shim) | yes (via shim) | yes (via shim) | not supported as native LoHa execution | Ring 1 (`experiments/peft_ring1/`, `docs/strategy/ring1_peft_generalization_results.md`) |
| LoKr | Inferred low-rank component or materialized Kronecker delta | inferred / deferred | not validated | not validated | not validated | no | Ring 1 support matrix only (`experiments/peft_ring1/artifact_support_matrix.json`) |
| Full checkpoint delta | Layer-summary checkpoint-delta representation (Representation C) | bounded-supported in tested settings (experimental outside scope) | yes | yes | yes | no (explicitly out of scope) | Ring 2 Stages A-D and checkpoint trials T01/T02 (`docs/design/ring2_stage_d_assessment_memo.md`, `field_trials/checkpoint_inventory_t01/`, `field_trials/checkpoint_inventory_t02/`) |

## Notes

- IA3 remains out of scope for this substrate because its learned object is not low-rank factor geometry and would require a measurement redesign.
- "Supported" in this matrix means support for audit/triage decisions, not universal artifact execution.
- Checkpoint-delta support is intentionally bounded to shared-base, small encoder, classification settings on CPU.
