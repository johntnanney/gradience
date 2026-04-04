# Phase-Probe Add-On Summary (Bounded)

## Scope
This was a CPU-only observables comparison add-on over existing saved checkpoints.
It does not claim a full phase-transition theory result.

## What was added
- HTSR-style tail exponent probe (`htsr_alpha`)
- edge-gap probe (`edge_gap_12 = sigma1/sigma2`)

Implemented in:
- `gradience/research/spectral_extended.py`

## Bounded result
Current decision status:
- HTSR alpha: `bounded_keep`
- edge-gap: `bounded_keep`

Why:
- both probes show bounded regime sensitivity in the analyzed slice,
- but neither currently shows enough robustness/lead behavior to be promoted as a primary standing observable.

## Research-Stack Hierarchy
Core standing observables:
- stable rank / effective rank
- energy concentration
- DFA / regime structure

Secondary bounded probes:
- edge-gap (`edge_gap_12`) as a lightweight companion metric
- HTSR alpha (`htsr_alpha`) when fit quality is adequate

Interpretation policy:
- keep both probes available in research outputs
- do not treat either as a front-line summary metric
- do not override core observables with probe-only movement

## Product implication
No product-surface change is recommended from this pass.
These probes remain research-layer observables.

## Canonical outputs
- `sidecar/results/phase_probe_addon/baseline_snapshot.json`
- `sidecar/results/phase_probe_addon/phase_probe_timeseries.json`
- `sidecar/results/phase_probe_addon/comparative_analysis.json`
- `sidecar/results/phase_probe_addon/phase_probe_decision_summary.json`
- `sidecar/notes/n129_phase_probe_keep_boundedkeep_discard_memo.md`
