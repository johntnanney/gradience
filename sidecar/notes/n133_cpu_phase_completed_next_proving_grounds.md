# N133 — CPU Phase Completed / Next Proving Grounds

## Summary

CPU-side exploratory and bounded-validation lines have reached usable decision
states. Main risk is now status drift, not missing experimentation.

## Consolidated Line Status

- rank-proxy validation: bounded positive and strategy-usable
- ablation proxy expansion: resolved enough for bounded policy freeze
- HTSR / edge-gap add-on: bounded_keep (secondary probes)
- merge-aware monitor: bounded_keep (same-task default)
- over-accumulation: keep_exploratory (not policy-ready)

## What This Unlocks

- cleaner internal/external communication
- less branch churn on low-yield CPU expansions
- immediate readiness for GPU-return proving-ground study

## Next Proving Ground

Decoder-only spectral fingerprinting is the highest-leverage next major line.

Spec:

- `docs/plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`

## Recommended Posture

1. freeze and maintain canonical status docs,
2. avoid opening new CPU exploratory branches for now,
3. execute decoder-side study when compute opens.
