# Why The Workflow Looks Like This

**Audience:** practitioner, collaborator, reviewer  
**Status:** stable (research-to-product translation)  
**Purpose:** explain the design logic behind the canonical preflight workflow  
**Canonical for:** rationale linking validated findings to product behavior  
**Supersedes:** implicit rationale spread across report/findings/strategy docs  
**See also:** [`../workflows/canonical_merge_triage_workflow.md`](../workflows/canonical_merge_triage_workflow.md), [`../product_surface.md`](../product_surface.md), [`../claims.md`](../claims.md), [`../technical-report.md`](../technical-report.md)

This page explains why the default workflow is:

1. Adapter QA and eligibility gating
2. Task-relationship-aware pair audit
3. Inventory narrowing and action plan
4. Behavioral evaluation on retained candidates

The goal is principled decision support, not prediction theater.

## Why Gating Comes First

Gating comes first because structurally plausible merges can still be operationally bad when source adapters are behaviorally weak.

Product implication:

1. Run single-adapter QA before pairwise merge interpretation.
2. Use eligibility status to block or deprioritize weak sources.
3. Treat pairwise analysis as conditional on source quality, not a replacement for it.

This is why the front-door workflow starts with `audit-adapter`.

## Why Task Relationship Matters

Task relationship is one of the most reliable practical separators in mixed inventories.

Product implication:

1. Same-task pairs are the primary evaluate-first region.
2. Same-family pairs are a bounded middle region.
3. Cross-task pairs are caution by default unless there is explicit reason to test them.

This is why task-relationship advisory is part of the stable merge report language.

## Why Narrowing Is Prioritized Over Predicting Success

The strongest validated utility is candidate narrowing, not universal success prediction.

Product implication:

1. Optimize the workflow to reduce evaluation burden and increase decision clarity.
2. Present outputs as triage guidance, not guaranteed merge outcomes.
3. Keep action-plan categories operational: retained, monitor, skip.

This is why the inventory summary/action plan is the canonical bundle center.

## Why Structural Audit and Behavioral Evaluation Are Both Required

Structural signals and behavioral outcomes answer different questions:

1. Structural audit answers: "Which pairs are worth evaluating first?"
2. Behavioral evaluation answers: "Which retained pair actually works best on the target outcome?"

Product implication:

1. Structural audit is the prefilter.
2. Behavioral eval is final adjudication.
3. Neither should be presented as a substitute for the other.

This is why the canonical flow ends with behavioral evaluation only on retained candidates.

## Research-to-Product Translation Map

| Research-side finding | Product-side workflow decision |
|---|---|
| Eligibility gating is necessary | QA first, pair analysis second |
| Same-task vs cross-task structure is operationally meaningful | Task-relationship advisory in default pair interpretation |
| Near-miss is a real middle class | Action plans include monitor/near-miss instead of binary keep/drop |
| Threshold revisions reduce ambiguous catch-all behavior | Current default thresholds remain tuned for clearer triage, not overclaiming |
| Some encoder heuristics do not transfer cleanly to decoders | Keep decoder claims bounded and observational unless controlled validation exists |
| Spectral partitioning supports task-relationship grounding | Maintain task relationship as a first-class workflow signal |

## Bottom Line

The workflow is not a pile of tricks. It is a constrained decision architecture:

1. Gate weak sources early.
2. Respect task boundaries.
3. Narrow aggressively.
4. Evaluate behaviorally at the end.

That sequence is what current evidence supports most strongly.
