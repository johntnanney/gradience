# Product Shipping Surface

**Audience:** maintainer, product owner, release manager  
**Status:** stable (shipping decision note)  
**Purpose:** define what ships by default vs advanced vs research-only  
**Canonical for:** release-surface and default-experience decisions  
**Supersedes:** implicit shipping scope spread across multiple docs  
**See also:** [`product_surface.md`](product_surface.md), [`claims.md`](claims.md), [`workflows/canonical_merge_triage_workflow.md`](workflows/canonical_merge_triage_workflow.md), [`00_start_here/stable-vs-experimental.md`](00_start_here/stable-vs-experimental.md)

This document is about product discipline: what users get by default, what is opt-in, and what stays out of the product path.

## Default (Ships by Default)

These capabilities define the main Gradience experience and should be front-door in CLI/docs/examples:

1. Adapter QA and eligibility gating (`audit-adapter`)
2. Pairwise merge-risk audit (`merge-audit`)
3. Inventory summary and action plan (`summarize-inventory`)
4. Task-boundary detection and task-relationship advisory
5. Near-miss handling in inventory action plans
6. Canonical report outputs (QA artifact, merge report, inventory summary, run bundle report)

Why default:

- Strongest validated operational value in the current evidence base
- Clear user story and low interpretation burden
- Directly supports the canonical happy path

## Advanced (Non-Default, Opt-In)

These capabilities are useful but should be explicitly opt-in and clearly labeled advanced/internal:

1. Verdict-confidence annotations and threshold-tuning overlays
2. Over-accumulation diagnostics
3. Edge-gap and HTSR probe metrics
4. Merge-aware training monitor
5. Direction-aware compatibility companions
6. Telemetry research probes

Why advanced:

- Bounded evidence or narrower utility scope
- Higher interpretation overhead
- Risk of crowding the product front door if enabled by default

Shipping rule:

- Keep these off default paths and main tutorial flows.
- Expose only behind explicit flags or advanced docs sections.

## Research-Only (Not in Product Defaults)

These lines stay in research/theory tracks and should not appear as product promises:

1. Analytical spectral-geometry theorem programs
2. Observational ecosystem studies as standalone decision engines
3. Exploratory sidecar diagnostics not promoted through bounded validation
4. Experimental Route 2 investigations without stable workflow integration

Why research-only:

- Not operationally validated for default product use
- Often hypothesis-generating rather than decision-bearing
- Requires additional controlled evidence before promotion

## Promotion Rule

A feature can move up a tier only when all are true:

1. Clear user-facing decision value in the canonical workflow
2. Reproducible evidence in the intended operating regime
3. Low ambiguity in interpretation for non-expert users
4. No regression to front-door clarity

If these conditions are not met, keep it advanced or research-only.
