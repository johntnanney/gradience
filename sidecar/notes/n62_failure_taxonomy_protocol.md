# n62 — Failure Taxonomy Protocol

**Type:** protocol
**Date:** 2026-03-28
**Depends on:** n61 (example behavior findings)
**Status:** Executed. Findings in n63.

---

## Objective

Build a small failure taxonomy from observed example-level patterns. The taxonomy should have 3–6 categories maximum, each with a clear behavioral definition, a geometric signature (where available), and an interpretable relationship to the mechanism ladder.

---

## Method

### Derivation approach

The taxonomy is derived bottom-up from the per-example behavioral categories in n61. Categories are consolidated based on:

1. **Behavioral distinctiveness:** Do the examples in this category share a recognizable prediction pattern?
2. **Class association:** Does the category concentrate in specific merge-quality classes?
3. **Confidence signature:** Does the category have a distinctive confidence profile?
4. **Interpretive utility:** Does knowing an example belongs to this category help explain what happened?

### Candidate categories from n61

The per-example analysis produced 8 raw categories. These are consolidated into the final taxonomy by merging categories that are behaviorally similar and splitting those that hide important distinctions.

### Flip catalog construction

For each taxonomy category, extract representative examples (up to 8 per category per case) showing the text, label, source predictions, merged prediction, and confidence values. The catalog is for interpretive inspection, not statistical analysis.

Script: `sidecar/scripts/build_failure_taxonomy.py`

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This protocol | `sidecar/notes/n62_failure_taxonomy_protocol.md` |
| Taxonomy JSON | `sidecar/results/example_semantics/failure_taxonomy.json` |
| Example flip catalog | `sidecar/results/example_semantics/example_flip_catalog.json` |
| Findings | `sidecar/notes/n63_failure_taxonomy_findings.md` |
