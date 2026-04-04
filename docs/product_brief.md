# Internal Product Brief

**Audience:** internal team, collaborator onboarding  
**Status:** stable (one-page operating brief)  
**Purpose:** fast, bounded statement of what Gradience is, what it does now, and what comes next  
**Canonical for:** internal product positioning and onboarding alignment  
**See also:** [`claims.md`](claims.md), [`product_surface.md`](product_surface.md), [`workflows/canonical_merge_triage_workflow.md`](workflows/canonical_merge_triage_workflow.md), [`strategy/state-of-program-april-2026.md`](strategy/state-of-program-april-2026.md)

## What Gradience Is

Gradience is a spectral preflight triage system for LoRA adapter composition.  
It helps teams decide which adapter pairs are worth behavioral merge evaluation before spending merge/eval budget.

Its validated product core is:

1. Adapter QA and eligibility gating
2. Task-boundary detection
3. Pairwise merge-risk audit
4. Inventory preflight narrowing
5. Actionable shortlist reports

## Who It Is For

1. Model and platform engineers managing adapter inventories
2. Teams running repeated merge evaluations and needing better candidate prioritization
3. Research collaborators who need a bounded, evidence-based merge triage workflow

## What Problem It Solves Now

Gradience solves the candidate-selection problem, not the full merge-success prediction problem.

In the validated encoder regime, Gradience:

1. Narrows merge candidates aggressively while preserving strong candidates
2. Flags task-boundary risk early in mixed inventories
3. Separates likely-useful, near-miss, and skip/deprioritized candidates before expensive evaluation
4. Enforces source eligibility so structurally plausible but weak sources do not dominate merge plans

## What It Does Not Do Yet

1. It does not replace behavioral evaluation for final merge decisions.
2. It does not provide universal decoder-side or generation-task guarantees yet.
3. It does not treat exploratory diagnostics as policy-authoritative recommendations.
4. It is not a training-control or optimizer-intervention system.

## Recommended Workflow

Use the canonical happy path in order:

1. Single-adapter QA
2. Inventory ingest
3. Task-boundary and family classification
4. Pairwise merge audit
5. Inventory summary and action plan
6. Behavioral evaluation only on retained candidates

Reference: [`workflows/canonical_merge_triage_workflow.md`](workflows/canonical_merge_triage_workflow.md)

## Next Proving Grounds

CPU-side consolidation is substantially complete. The next decisive evidence requires GPU.

Priority proving grounds:

1. DeBERTa adjudication gate for encoder-side mechanism portability
2. Controlled decoder fingerprinting and merge-triage validation
3. Outcome-linked validation of confidence stratification and related bounded companions

Operational stance now:

1. Ship and rely on the validated triage core
2. Keep bounded companions available but clearly secondary
3. Treat GPU-return studies as the path to scope expansion
