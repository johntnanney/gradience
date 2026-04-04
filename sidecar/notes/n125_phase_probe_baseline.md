# n125 - Phase-Probe Add-On Baseline Freeze

## Scope Freeze
This add-on is frozen to CPU-only re-analysis over existing saved checkpoint artifacts.

- No new training runs.
- No new model families.
- Existing baseline toolkit remains:
  - `stable_rank_mean`
  - `effective_rank_mean`
  - `top1_energy_mean`
  - `spectral_decay_alpha_mean`

## Frozen Run Slice
Using `sidecar/scripts/run_phase_probe_addon.py --max-runs 6`, the frozen run slice is:

1. `bench_runs/smoke_test/probe_r16`
2. `test_regular_undertrained/probe_r16`
3. `bench_runs/multiseed_seed42/probe_r32`
4. `bench_runs/cert_v0.1_seed42/probe_r32`
5. `bench_runs/cert_v0.1_seed42/uniform_median_r2`
6. `bench_runs/safety_uniform_r16_extended_seed42/probe_r32`

Snapshot totals:
- run count: `6`
- analyzed timepoints: `32`

## Baseline Transition Reference
Candidate transition reference is frozen as:
- peak absolute change step in `stable_rank_mean` per run.

## Notes
- DFA metrics are not available in this checkpoint slice, so comparison in this pass is against the available stable-rank / energy / decay baseline.
- Canonical machine-readable freeze: `sidecar/results/phase_probe_addon/baseline_snapshot.json`.
