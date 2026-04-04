# Product Surface

**Audience:** maintainer, product owner, collaborator  
**Status:** stable (internal scope spec)  
**Purpose:** define what is core product vs advanced companion vs research-only  
**Canonical for:** product-surface scope decisions and messaging boundaries  
**Supersedes:** implicit scope spread across multiple docs  
**See also:** [`product/README.md`](product/README.md), [`00_start_here/stable-vs-experimental.md`](00_start_here/stable-vs-experimental.md), [`strategy/state-of-program-april-2026.md`](strategy/state-of-program-april-2026.md)

## Core Workflow

This is the product surface that should be easiest to understand, demo, and run:

1. Adapter QA and eligibility gating
2. Task-boundary detection
3. Pairwise merge-risk audit
4. Inventory preflight and candidate narrowing
5. Actionable merge triage reports

Operational rule: this workflow is the default user story and the primary product commitment.

## Advanced / Secondary Features

These features are useful but should not define the front door. They are opt-in advanced companions:

1. Threshold tuning and verdict-confidence refinements
2. Over-accumulation diagnostics
3. Edge-gap and HTSR alpha probes
4. Merge-aware training monitoring
5. Direction-aware compatibility companions
6. Telemetry-side research probes

Operational rule: present these as advanced/internal diagnostics, not required steps in normal workflow execution.

## Research-Only Capabilities

These lines are research-facing and should not be framed as product behavior:

1. Theory-development lines (for example, analytical spectral geometry and convergence-bound work)
2. Observational ecology studies (for example, decoder ecosystem census and robustness add-ons)
3. Exploratory Route 2 and sidecar investigations not promoted into stable workflow logic

Promotion rule: research-only capabilities require bounded validation plus outcome-grounded operational evidence before entering the core product surface.
