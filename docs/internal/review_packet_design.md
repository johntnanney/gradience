# Review Packet — Design Document

**Status:** Draft
**Date:** 2026-03-27

---

## 1. Purpose

The review packet is a single consolidated markdown artifact
(`review_packet.md`) that assembles all key preflight outputs into
one shareable document. It is designed for handoff to collaborators,
future-you, or anyone who needs the full picture of a preflight run
without navigating multiple files.

---

## 2. Sections (fixed order)

The review packet contains exactly seven sections, assembled from
existing stable outputs. No new analysis logic.

| # | Section | Source |
|---|---------|--------|
| 1 | Header / run metadata | run_id, inventory_id, timestamp, adapter/pair counts |
| 2 | Inventory Policy Summary | `inventory_policy_summary` from preflight JSON |
| 3 | Source QA / Trust Snapshot | adapter_status_counts, evidence_tier_counts, provenance |
| 4 | Action Plan | action_plan (reduced candidate set, evaluate-first, exclude, zones) |
| 5 | Drift Summary | `inventory_drift_summary` from preflight JSON (if previous run) |
| 6 | Compare to Previous | comparison highlights (if previous run) |
| 7 | Artifact Links | relative paths to detailed artifacts in the bundle |

---

## 3. JSON Companion

A `review_packet.json` is emitted alongside, containing:

```json
{
  "schema": "gradience.review_packet/v1",
  "inventory_id": "...",
  "run_id": "...",
  "timestamp": "...",
  "sections_present": ["header", "policy_summary", "trust_snapshot", "action_plan", ...],
  "policy_summary": { ... },
  "trust_snapshot": { ... },
  "action_plan_summary": { ... },
  "drift_summary": { ... } or null,
  "previous_run_id": "..." or null
}
```

---

## 4. Guardrails

1. **Assembly only.** No new scoring, no new derivation, no new prose.
2. **Sections omitted cleanly.** If drift/comparison data is absent
   (first run), those sections are simply not included.
3. **No duplication of raw data.** The packet references artifact
   paths; it does not inline full QA artifacts or merge reports.
4. **Stable section titles.** Section headings are frozen.

---

## 5. Implementation Surface

| File | Change |
|------|--------|
| `gradience/vnext/inventory/run_bundle.py` | Add `build_review_packet_md()`, `build_review_packet_json()` |
| `gradience/vnext/inventory/run_bundle.py` | Update `emit_run_bundle()` to emit review_packet.md + review_packet.json |
| `tests/test_inventory_summary.py` | Add review packet test class |
