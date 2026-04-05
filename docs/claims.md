# What Gradience Can Currently Claim

**Audience:** maintainer, writer, collaborator  
**Status:** stable (authoritative claims page)  
**Purpose:** prevent overclaiming and keep outward-facing language aligned with current evidence  
**Canonical for:** validated, bounded, and not-yet-validated claim boundaries  
**Supersedes:** claim language scattered across report, findings, and strategy memos  
**See also:** [`technical-report.md`](technical-report.md), [`strategy/state-of-program-april-2026.md`](strategy/state-of-program-april-2026.md), [`product_surface.md`](product_surface.md), [`workflows/canonical_merge_triage_workflow.md`](workflows/canonical_merge_triage_workflow.md)

Use this page as the source of truth before writing docs, posts, demos, collaborator briefs, or product-facing summaries.

## Validated Claims

These claims are supported strongly enough for operational reliance in the currently validated regime.

1. Gradience is effective at **candidate narrowing** for adapter merge triage in the validated encoder regime.
2. The canonical workflow can remove most candidate pairs while preserving the strongest merge candidates.
3. **Task-boundary detection** is the highest-confidence workflow signal in mixed-task inventories.
4. **Eligibility gating** is necessary before pairwise merge analysis; structurally plausible pairs can still fail if sources are behaviorally weak.
5. The most reliable operational framing is **structural prefiltering + behavioral adjudication**.
6. Adapter-level spectral audit reads stable, nontrivial structure without loading base-model weights.

## Bounded Claims

These are real findings, but only valid within explicit scope limits.

1. The conjunctive failure model is strongly supported on tested encoder backbones, but not yet universal.
2. Spectral partition findings (shared high-SV structure vs lower-SV differentiation) are compelling in bounded encoder settings.
3. Decoder-side ecosystem census shows real non-random structure, but remains observational and confound-aware.
4. Decoder-scale merge evidence is promising but still narrow in pair count and architecture breadth.
5. Threshold calibration results are engineering-stable for the current corpus, not general theorems.
6. Secondary probes (edge-gap, HTSR alpha, direction-aware companions, monitor telemetry) are useful as companions, not front-line decision engines.

## Not Yet Claimed

Do not claim the following today:

1. Universal merge-success prediction from spectral metrics alone.
2. Controlled decoder generalization across architectures/tasks.
3. Generation-task merge-triage validation parity with current encoder evidence.
4. Equivalence to adaptive-rank training methods or state-of-the-art rank-allocation methods.
5. Production policy authority for exploratory signals (for example over-accumulation variants).
6. Training-time control or optimization improvement from merge-aware monitoring.
7. Broad causal claims from observational public-artifact analyses.

## Current Recommended Use

Use Gradience as a **preflight triage system**:

1. Run adapter QA and eligibility gating.
2. Run pairwise merge audit with task-relationship interpretation.
3. Build inventory summary and action plan.
4. Evaluate behaviorally only retained candidates.

What the workflow is for:

- Reduce merge-search space and prioritize evaluation budget.
- Make mixed-task inventories operationally legible.
- Surface where caution is warranted before expensive merge evaluation.

What the workflow is not for:

- Replacing behavioral evaluation.
- Certifying universal merge success.
- Supporting broad decoder claims without controlled follow-on evidence.

## What Still Requires GPU

The highest-value next scope-expansion checks are GPU-gated:

1. DeBERTa adjudication for third-backbone mechanism portability.
2. Controlled decoder fingerprinting and inventory-level merge triage validation.
3. Outcome-linked validation of confidence stratification and related bounded companions.

## Outward-Facing Claim Template

When writing externally, use this structure:

1. Start with validated workflow utility (candidate narrowing + task-boundary triage).
2. State the bounded regime explicitly (encoder-heavy validated core; decoder evidence bounded/observational unless controlled).
3. Separate stable product behavior from exploratory diagnostics.
4. End with what still requires controlled GPU follow-on work.
