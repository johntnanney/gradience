# n114 -- Mixed-Evidence Triage Panel Definition

**Type:** stress-test setup note  
**Date:** 2026-03-31  
**Program:** Route2 Mixed-Evidence Triage Stress Test  
**Stage:** B  
**Depends on:** n113  
**Status:** complete

---

## Objective

Construct a small panel intentionally weighted toward mixed-evidence review and same-family optional cases while preserving clear retained/blocked anchors.

---

## Panel structure

Panel size: 8 cases.

Composition:

- `anchor_retained`: 1
- `anchor_blocked`: 2
- `review`: 2
- `same_family_optional`: 3
- `other_middle`: 0

This composition intentionally overweights the soft middle (5/8 cases are review or same-family optional).

---

## Case-source policy

All cases come from existing Route 2 artifacts only:

- checkpoint triage T02 outputs
- adapter same-family confirmation outputs
- previously audited mixed-evidence review cases

No new training campaign or new artifact class was introduced.

---

## Outputs

- `sidecar/results/route2_stress_tests/mixed_evidence_triage/panel_table.json`
- `sidecar/results/route2_stress_tests/mixed_evidence_triage/panel_table.md`
- `sidecar/results/route2_stress_tests/mixed_evidence_triage/panel_role_table.md`
- `sidecar/notes/n114_mixed_evidence_triage_panel.md`
