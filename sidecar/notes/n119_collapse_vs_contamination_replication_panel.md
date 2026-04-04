# n119 -- Collapse vs Contamination Replication Panel

**Type:** replication panel note  
**Date:** 2026-03-31  
**Program:** Route2 Collapse vs Contamination Replication  
**Stage:** B  
**Depends on:** n118, sidecar/results/example_semantics/predictions/*.json  
**Status:** complete

---

## Objective

Define a small, nearby replication panel that can test channel stability with minimal scope expansion.

---

## Panel strategy

The panel uses both replication modes:

1. **Case replication**: a nearby collapse case (`FR-02`) from the same behavioral lineage.
2. **Slice replication**: deterministic even/odd slices from original anchor cases to test slice stability.

This keeps perturbation local while probing robustness beyond a single full-case read.

---

## Final panel (4 targets)

1. `R1_FR02_case` (`collapse_like`, case): nearby collapse case with weaker source-B quality.
2. `R2_FR01_even_slice` (`collapse_like`, slice): even-index slice of original collapse anchor.
3. `R3_CT01_even_slice` (`contamination_like`, slice): even-index slice of original contamination anchor.
4. `R4_CT01_odd_slice` (`contamination_like`, slice): odd-index complement for contamination slice stability.

---

## Why these targets

1. They stay in the same backbone and merge-facing decision context.
2. They reuse existing behavior-rich outputs, avoiding new training/eval campaigns.
3. They represent both channels with small, defensible perturbations.
4. They directly test whether confidence-channel separation survives case/slice variation.

---

## Outputs

- `sidecar/notes/n119_collapse_vs_contamination_replication_panel.md`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/panel_table.json`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/panel_table.md`
