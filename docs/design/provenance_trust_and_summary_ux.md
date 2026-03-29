# Design Note: Provenance/Trust Language Completion (C) + Light Summary UX Pass (D)

**Date:** 2026-03-26
**Status:** In progress
**Roadmap reference:** CPU-Only Roadmap, Phase 1, Projects C and D

## Context

Projects A and B established the Inventory Action Plan and Preflight Run Bundle.
Two gaps remain before Phase 1 can be called "workflow-complete enough" (Decision Point 1):

1. **Provenance / trust language** — the system tracks behavioral evidence
   availability but doesn't make the evidence tier visible everywhere it
   should be, and one spot uses "verified" in a way that conflicts with
   the overall non-verification stance.

2. **Summary UX** — the terminal and markdown renderings are functional but
   could be tighter, more consistent, and better aligned between human-readable
   and machine-readable outputs.

## Project C: Provenance / Trust Language

### Current state

The system already tracks three evidence-tier signals:
- `eval_available` (bool) on each `AdapterQAArtifact`
- `EligibilityStatus` (eligible / uncertain / flagged_weak / unknown_no_behavioral_eval)
- A provenance note in `format_inventory_summary()` with an explicit
  non-verification disclaimer: "Gradience does not independently verify
  claimed evaluation results."

### Gaps

| Gap | Location | Fix |
|-----|----------|-----|
| "verified behavioral quality" language | `qa_report.py` `_confidence_note()` line 481 | Change to "user-reported behavioral quality" |
| No per-adapter evidence tier in action plan MD | `run_bundle.py` `build_action_plan_md()` | Add provenance section |
| No provenance in preflight summary MD | `run_bundle.py` `build_preflight_summary_md()` | Add evidence-tier section |
| No evidence counts in preflight summary JSON | `run_bundle.py` `build_preflight_summary_json()` | Add `behavioral_evidence_count`, `total_source_count` |
| No evidence-tier tracking in comparison MD | `run_bundle.py` `build_comparison_md()` | Compare evidence counts between runs |
| InventorySummary has no evidence-tier count map | `summary.py` `InventorySummary` | Add `evidence_tier_counts` to builder (additive field) |

### Design decisions

1. **Evidence tier vocabulary.** Three tiers, consistently named everywhere:
   - `behavioral_verified` → renamed to `behavioral_reported` (user-provided, not independently verified)
   - `behavioral_uncertain` (eval available but status uncertain)
   - `behavioral_missing` (no eval data)

   The word "verified" is reserved for a future capability where Gradience
   could cross-check evaluation claims. Until then: "reported."

2. **Non-verification disclaimer.** Already present in `format_inventory_summary()`.
   Add to `build_preflight_summary_md()` as well.

3. **No new schema fields on frozen dataclasses.** `InventorySummary.from_dict()`
   already ignores extra keys. We add `evidence_tier_counts` to `to_dict()` output
   and `build_inventory_summary()` but keep `from_dict()` permissive (extra keys
   silently ignored). This is additive-only.

## Project D: Light Summary UX Pass

### Current state

- Terminal output has 4 blocks (OVERVIEW, SOURCE QA SNAPSHOT, STRUCTURAL DETAIL, INTERPRETATION)
- Action plan has 7 blocks (REDUCED CANDIDATE SET, EXCLUDE, SAME-TASK, CROSS-TASK, PROVENANCE, SUMMARY)
- Preflight summary MD has 6 sections
- Action plan MD has 5 sections (missing provenance and per-pair detail)

### Gaps

| Gap | Location | Fix |
|-----|----------|-----|
| Reduced candidate set in MD lacks per-pair risk/strategy | `build_preflight_summary_md()` | Add detail like terminal version |
| Action plan MD lacks provenance section | `build_action_plan_md()` | Add provenance section |
| Action plan MD lacks per-pair detail | `build_action_plan_md()` | Add risk/strategy to same-task items |
| Preflight summary JSON lacks evidence counts | `build_preflight_summary_json()` | Add fields |
| Comparison MD doesn't track evidence changes | `build_comparison_md()` | Add evidence delta |

### Design decisions

1. **Alignment principle.** Every field that appears in the terminal action
   plan should also appear in the markdown action plan and the JSON summary.
   The three outputs should be projections of the same data, not independent
   compositions.

2. **Wording consistency.** All action-plan sections use the same vocabulary:
   - "Exclude / deprioritize" (not "skip" or "remove")
   - "Same-task safe zone" (not "retained" or "priority")
   - "Cross-task caution zone" (not "warning" or "risk")
   - "Evaluate first" (not "prioritize" or "recommend")

3. **Structural detail readability.** The terminal STRUCTURAL DETAIL block
   currently puts all counts on one line per category. Break into
   multi-line when > 3 items for better scanning.

## Completion signals

- **Project C:** A user can tell, quickly and explicitly, what kind of
  evidence supports each adapter — in the terminal, in the markdown
  summary, and in the JSON output. No instance of "verified" remains
  in the non-verification context.

- **Project D:** The markdown summary is a faithful, readable projection
  of the terminal output. All three output formats (terminal, markdown,
  JSON) agree on field names and values.
