# n73 — Decision-Profile Taxonomy

**Type:** taxonomy note  
**Date:** 2026-03-31  
**Depends on:** n70 panel, n71 audit, n72 aggregation analysis  
**Status:** Stage D complete

---

## Goal

Compress scenario comparisons into a small, reusable profile set that is broader than merge-only language but still operational.

---

## Outputs

- `sidecar/results/decision_dependent_compatibility/decision_profile_table.json`
- `sidecar/results/decision_dependent_compatibility/decision_profile_table.md`

Final taxonomy size: **6 profiles** (within 4–7 target).

---

## Profiles

1. `P1_redundant_confusable`  
   Redundant for merge value, confusable for routing.

2. `P2_overlap_needs_disambiguation`  
   Moderate overlap; routing needs explicit disambiguation.

3. `P3_merge_ok_routing_separable`  
   Merge-admissible relation can still be easy to route apart.

4. `P4_qa_blocked_structurally_nontrivial`  
   Structurally interesting relation blocked by weak source QA.

5. `P5_same_family_optional`  
   Same-family relation supports optional review, not auto-retention.

6. `P6_cross_task_low_value_control`  
   Cross-task control: useful as boundary check, low default priority.

---

## Why this taxonomy is useful

It preserves three things at once:

- shared structural substrate,
- scenario-specific aggregation effects,
- policy-level action guidance.

It also prevents a merge-centric collapse into `safe/fragile` only, while avoiding taxonomy sprawl.

---

## Evidence grounding

Representative cases are bound to concrete artifacts in `panel_table.json`, including:

- routing pilot high/moderate/low confusability pairs,
- targeted confirmation merge controls and same-family cases,
- checkpoint triage same-task/same-family/cross-task cases with QA gating.

---

## Caveat

The taxonomy is intended as a **bounded operating vocabulary**, not a universal ontology.  
Profile semantics should be updated only when new scenario evidence materially changes assignments.
