# n82 -- Aggregation Family Implementation Audit

**Type:** analysis note
**Date:** 2026-03-31
**Program:** Aggregation-Sensitive Compatibility (Route 2)
**Stage:** B
**Depends on:** n81 (panel definition), n71 (shared vs specific stack audit)
**Status:** complete

---

## Question

What exactly is each aggregation family doing to local or case-level evidence?

---

## Families formalized

### A. Worst-case (merge-like)

Reduces the pair to its worst structural region. A single high-divergence layer or subspace conflict can determine the verdict. Appropriate for merge because one bad layer can cause catastrophic degradation (the V-module pathology is a single-layer phenomenon). Collapses the distinction between "mostly safe with one bad layer" and "uniformly risky."

### B. Distributional (routing-like)

Summarizes the pair by the spread and concentration of local compatibility scores. Distinguishes confusable (high overlap), needs-disambiguation (moderate), and separable (clear gap). Appropriate for routing because confusability depends on the prevalence of similar subspaces, not just the worst-case overlap. Preserves gradation that worst-case erases.

### C. QA-dominant (triage-like)

Evaluates source evidence status before consulting structural measurements. Blocked = no action regardless of structural compatibility. Clear = structural analysis proceeds. Appropriate for triage because proceeding without evidence is riskier than waiting. Erases structural gradation within the blocked set.

### D. QA-gated distributional (hybrid)

QA gate first, distributional analysis second. Not present in existing code as a single function but appears naturally in the adapter T01 workflow: all sources were eligible, so QA cleared, and the distributional profile became operative. Checkpoint T02 shows the opposite: QA blocks everything, so the distributional layer never fires.

---

## Where each family already appears in the codebase

| Family | Where it appears |
|--------|-----------------|
| Worst-case | `vnext/merge/recommend.py` (pair diagnosis), `vnext/merge/spectral_compat.py` (layer-level verdicts) |
| Distributional | `experiments/routing_pilot/` (confusability scoring from compatibility distributions) |
| QA-dominant | `vnext/inventory/summary.py` (action plan builder), `api.py` (`--strict-qa`) |
| QA-gated distributional | Not a single function; emerges from the workflow when QA clears and distributional data exists |

---

## Expected overlap and divergence

| Condition | Worst-case | Distributional | QA-dominant |
|-----------|-----------|---------------|-------------|
| Same-task, QA-clear, low risk | merge_safe | confusable | qa_clear |
| Same-task, QA-blocked, low risk | merge_safe | confusable | **qa_blocked** |
| Cross-task, QA-clear, high risk | **merge_risky** | separable | qa_clear |
| Cross-task, QA-blocked, high risk | **merge_risky** | separable | **qa_blocked** |
| Same-family, moderate | merge_caution | needs_disambiguation | depends on QA |

The key prediction: the most aggregation-sensitive cases are those where QA regime differs from structural regime (high structure + blocked QA, or low structure + clear QA).

---

## Output artifacts

- `sidecar/results/aggregation_sensitive_compatibility/aggregation_family_specs.json`
- `sidecar/results/aggregation_sensitive_compatibility/aggregation_family_specs.md`
- `sidecar/notes/n82_aggregation_family_audit.md` (this note)
