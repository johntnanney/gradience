# Core-Space Audit (Optional Diagnostic)

## What this is

The core-space audit is an **optional, pairwise structural diagnostic** for `gradience merge-audit`.

It estimates whether two adapter updates can be represented in a shared low-rank basis with low distortion.

Current classification: **advanced optional diagnostic** (diagnostic-first; not default workflow logic).

## What this is not

- It does **not** change default merge recommendations.
- It does **not** replace `pair_risk`, `dominant_issue`, or `recommended_strategy`.
- It does **not** require GPU; current implementation is CPU-compatible.

## How to run it

```bash
gradience merge-audit \
  --adapter-a ./adapter_a \
  --adapter-b ./adapter_b \
  --compute-core-space \
  --emit-report reports/ab_report.json
```

When `--compute-core-space` is enabled:
- terminal output adds a `CORE-SPACE DIAGNOSTIC` block
- emitted `gradience.merge_qa_report/v1` JSON includes an optional `core_space` section

## JSON shape (optional block)

```json
"core_space": {
  "shared_basis_score": 0.71,
  "basis_distortion": 0.18,
  "effective_shared_rank": 6,
  "status": "compatible"
}
```

If `--compute-core-space` is not enabled, this block is omitted.

Python API helper (advanced optional):

```python
from gradience.api import compute_core_space_diagnostic

core_space = compute_core_space_diagnostic(
    adapter_a="./adapter_a",
    adapter_b="./adapter_b",
)
```

## Metric interpretation

- `shared_basis_score` (higher is better): bounded estimate of shared-basis fitness.
- `basis_distortion` (lower is better): normalized penalty vs. separate bases.
- `effective_shared_rank`: rank needed to hit the configured energy threshold for joint representation.
- `status`: one of `compatible`, `marginal`, `incompatible`, `not_applicable`.

## Status intent

- `compatible`: shared basis represents both updates with low distortion.
- `marginal`: shared-basis fit is mixed; treat as a caution signal.
- `incompatible`: shared-basis representation is notably distorted.
- `not_applicable`: insufficient non-degenerate signal.

## Policy note

Current phase is **diagnostic-first**: core-space output is additive metadata for review and study, not an automatic strategy override.

## Current positioning (2026-03)

Core-space is a selective advanced structural diagnostic. It is **not** the main answer to cross-task merge risk — that role is now filled by the task-relationship advisory, which is part of the stable interpretive layer and addresses the key regime boundary (task identity) directly via metadata.

Verified adjudication showed that core-space is structurally informative but not broadly behaviorally decisive: same-task merges were safe even when core-space flagged them as incompatible, and cross-task degradation was better predicted by task identity than by shared-basis scores. Core-space remains appropriate for selective use in genuinely ambiguous cases where both pair-risk and task metadata are inconclusive.

See `docs/internal/verified_adjudication_implications.md` and `docs/internal/regime_map_after_phase2.md`.
