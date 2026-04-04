# What The CPU Phase Established

## Scope

This memo consolidates what is now settled from the CPU-only phase and what
must wait for GPU or collaborator-side expansion.

## 1) Merge-Side Refinements

- Over-accumulation line remains exploratory (`keep_exploratory`):
  - structural signal exists,
  - but policy-direction validity is weak/mixed.
- Merge-aware monitor is retained as bounded internal diagnostic (`bounded_keep`):
  - callback path works,
  - trajectory interpretability improves with reference choice,
  - same-task is preferred default; same-family fallback; cross-task exploratory.

## 2) Rank-Advisor Bounded Validation

- Bounded positive result in compressible encoder subset:
  - spectral policies are competitive fixed-budget allocation guides.
- Current frozen policy roles:
  - `proxy_gradient`: operational comparator,
  - `proxy_ablation_attenuate`: explanatory companion,
  - `oht`: lead spectral policy.

## 3) Secondary Research Probes

- HTSR alpha and edge-gap probes are retained as secondary observables (`bounded_keep`).
- They add bounded contextual value but are not front-line summary metrics.

## 4) Internal Tools: Kept vs Paused

Kept:

- merge-aware monitor (internal, diagnostic-only)
- phase-probe add-ons (secondary research observables)

Paused / deprioritized:

- rank-reduction ablation expansion in current bounded regime
- over-accumulation escalation into policy/execution use
- additional CPU-only exploratory branches without consolidation value

## 5) What Now Requires GPU

Highest-priority next proving ground:

- decoder-only spectral fingerprinting program
  - roadmap: [`../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md)

## 6) What Can Still Advance On CPU (Theory-Only)

- analytical spectral geometry of merge operations:
  - theorem/bound derivation line with synthetic validation only
  - no training compute required
  - spec: [`../plans/2026-04-03-analytical-spectral-geometry-of-merge-operations-plan.md`](../plans/2026-04-03-analytical-spectral-geometry-of-merge-operations-plan.md)

## 7) What Now Requires Collaborator / External Workflow

- broader external replication and task/architecture diversity checks
- integration into collaborator-facing evaluation workflows
- public claim hardening beyond bounded internal summaries

## Final Strategy Readout

The CPU phase is complete enough to justify consolidation-first posture:

1. canonicalize bounded conclusions,
2. hold exploratory lines at bounded status,
3. direct next major effort to decoder-only GPU proving grounds.
