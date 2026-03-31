# n104 -- Aggregation Mixed-Evidence Panel Definition

**Type:** substudy setup note
**Date:** 2026-03-31
**Program:** Route2 Aggregation Mixed-Evidence Triage Perturbation
**Stage:** B
**Depends on:** n103
**Status:** complete

---

## Objective

Construct a small triage-weighted perturbation panel emphasizing mixed-evidence review and same-family optional cases.

---

## Panel design choices

1. Keep one clear retained anchor.
2. Keep blocked anchors (including a same-task QA-override anchor).
3. Overweight mixed-evidence review and same-family optional cases.
4. Keep the panel small (8 cases).

Final composition:

- anchors: 1 clear retained
- blocked controls: 2
- mixed review: 2
- same-family optional: 3

---

## Why this panel is a valid soft-middle stress test

- The middle is intentionally dense: 5/8 cases are mixed or optional.
- Same-family optionality is represented in both clear and mixed evidence regimes.
- The blocked same-task anchor preserves the key QA-vs-structure contradiction.
- Sources come only from existing Route 2 artifacts (T01/T02 and targeted confirmations).

---

## Outputs

- `sidecar/results/route2_stability/aggregation_mixed_evidence/panel_table.json`
- `sidecar/results/route2_stability/aggregation_mixed_evidence/panel_table.md`
- `sidecar/results/route2_stability/aggregation_mixed_evidence/panel_role_table.md`
- `sidecar/notes/n104_aggregation_mixed_evidence_panel.md` (this note)
