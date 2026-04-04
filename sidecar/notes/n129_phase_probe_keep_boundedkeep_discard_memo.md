# n129 - Keep / Bounded-Keep / Discard Memo (Phase-Probe Add-On)

## What was tested
This pass compared two new probes against the existing spectral toolkit on a CPU-only reanalysis slice of saved checkpoints:
- HTSR-style tail exponent (`htsr_alpha`)
- edge-gap probe (`edge_gap_12 = sigma1/sigma2`)

Baseline comparators in-slice:
- `stable_rank_mean`
- `effective_rank_mean`
- `top1_energy_mean`
- `spectral_decay_alpha_mean`

## What added value appeared
- Both probes showed nontrivial regime sensitivity on this slice.
- Edge-gap had strong bounded correlation with outcome metrics in this slice.
- HTSR alpha showed moderate bounded signal with lower coverage/stability than edge-gap.

## What remained bounded
- Slice size is small (`6` runs, `32` timepoints).
- Candidate-transition lead was not earlier by median for either probe.
- DFA-aligned comparison is not available in this checkpoint slice.
- Edge-gap overlaps substantially with existing concentration summaries (not fully independent).

## Decision
Per probe decision:
- HTSR alpha: `bounded_keep`
- edge-gap: `bounded_keep`

Rationale:
- neither probe cleanly clears a strict \"keep as standing primary observable\" bar in this pass,
- both contribute enough bounded information to retain as secondary research observables.

## Next steps (if continued)
1. Re-run on a larger pre-existing checkpoint slice with stronger regime diversity.
2. Add a matched comparison where DFA-style observables are available at aligned timepoints.
3. Keep probe language explicitly exploratory/secondary in research summaries until replicated.

## Canonical artifacts
- `sidecar/results/phase_probe_addon/baseline_snapshot.json`
- `sidecar/results/phase_probe_addon/phase_probe_timeseries.json`
- `sidecar/results/phase_probe_addon/comparative_analysis.json`
- `sidecar/results/phase_probe_addon/phase_probe_decision_summary.json`
