# n80 -- Cross-Artifact Product Relevance Filter

**Type:** synthesis note
**Date:** 2026-03-31
**Program:** Cross-Artifact Compatibility Research (Route 2)
**Stage:** E
**Depends on:** n76 (panel), n77 (invariants), n78 (local signals), n79 (framework)
**Status:** complete

---

## Question

Which compatibility distinctions are stable enough across artifact classes to deserve stable workflow exposure?

---

## Product relevance classification

### Safe to expose (2 distinctions)

**1. Evidence bootstrap and QA gating as first-class triage steps.**
This is the safest cross-artifact claim. Evidence gating dominates every inventory tested -- LoRA, LoHa, and checkpoint delta alike. It operates on evidence metadata, not on structural measurements, and is representation-agnostic by design. Route 2 product language can state: "Gradience gates candidates by evidence quality regardless of artifact type."

**2. Conservative candidate narrowing.**
The workflow shape (broad input -> narrow output) survives representation change in all three tested classes. The narrowing logic is consistent: QA first, task boundaries second, structural risk third. Route 2 product language can state: "Gradience reduces candidate space to an actionable subset across supported artifact classes."

### Safe with scope guardrail (3 distinctions)

**3. Same-task vs cross-task ordering.**
Confirmed in LoRA and checkpoint delta. Not yet testable in LoHa. Guardrail: "where testable, same-task pairs consistently appear more compatible than cross-task pairs." Do not state this as a universal law.

**4. Same-family intermediate status.**
Confirmed in LoRA and checkpoint delta. Guardrail: "same-family pairs occupy an intermediate position where the task-family registry applies." The registry is still small.

**5. Triage as the cross-artifact decision scenario.**
Triage has complete coverage across all three classes. Merge and routing do not. Guardrail: "triage extends across artifact classes; merge and routing remain LoRA-specific."

### Research only (3 distinctions)

**6. Factor-geometry structural cues.**
The V-module dimensionality ratio (d=3.36) is the sidecar's strongest finding but is representation-locked. It cannot be computed from checkpoint summaries. Do not surface as a cross-artifact primitive.

**7. Merge strategy recommendations.**
Merge execution is restricted to native LoRA. Do not surface strategy strings for non-LoRA artifacts.

**8. Summary-profile-specific checkpoint cues.**
Unique to checkpoint delta Representation C. Do not reference in cross-artifact product language.

### Not stable enough (3 distinctions)

**9. Near-miss as a cross-artifact product category.**
Well-validated in LoRA. Not yet observed elsewhere. Wait for broader panel coverage.

**10. Numeric compatibility scores as cross-artifact comparisons.**
Same name, different scales. Ordinal ranking may transfer; absolute values do not. Do not compare scores across representation families.

**11. Pair risk labels as cross-artifact comparisons.**
Same vocabulary, different calibration. Do not treat as equivalent across classes.

---

## Bottom line

**What broader Route 2 can safely say:**

1. Evidence gating and candidate narrowing are cross-artifact capabilities.
2. Task-relation ordering (same > family > cross) is a recurring structural observation.
3. Triage is the decision scenario that works across artifact classes.
4. The workflow shape transfers; the specific structural metrics do not.

**What it cannot yet say:**

1. That structural compatibility signals are equivalent across artifact classes.
2. That merge or routing extend beyond LoRA.
3. That numeric scores or risk labels are comparable across representation families.
4. That near-miss is a cross-artifact product category.

**Why:**

The cross-artifact substrate is real but narrow. It consists of workflow-level signals and task-relational ordering, not structural measurements. The project's most discriminative signals (V-module ratio, subspace overlap) are representation-locked to factorized artifacts. The most portable signals (QA gating, narrowing) are the least mechanistically specific.

This is consistent with H3 ("artifact broadening preserves workflow shape more than feature parity") and H4 ("cross-artifact science should focus on invariants, not identical metrics"). The product should track the invariants, not the artifact-local details.

---

## Relationship to n75

This note supersedes the product relevance section of n75. The findings are consistent but more precisely layered. n75's "safe to expose" and "keep research-only" lists remain correct at their level of detail; this note adds the framework context, explicit guardrails, and the "not stable enough" category.

---

## Output artifacts

- `sidecar/results/cross_artifact_portability/product_relevance_filter.json`
- `sidecar/notes/n80_cross_artifact_product_relevance.md` (this note)
