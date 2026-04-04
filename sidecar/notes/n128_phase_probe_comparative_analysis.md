# n128 - Phase-Probe Comparative Analysis

## Dataset slice
- Runs: `6`
- Timepoints: `32`
- Source: existing checkpoint artifacts only

Canonical artifacts:
- `sidecar/results/phase_probe_addon/phase_probe_timeseries.json`
- `sidecar/results/phase_probe_addon/comparative_analysis.json`

## Probe validity
- `edge_gap_12` valid coverage: `1.000`
- `htsr_alpha` valid coverage: `0.844`

## Regime discrimination (stable-rank tertiles)
Observed means by regime:
- high-rank regime: `edge_gap_12_mean ~= 3.76`, `htsr_alpha_mean ~= 1.36`
- mid-rank regime: `edge_gap_12_mean ~= 5.35`, `htsr_alpha_mean ~= 1.41`
- low-rank regime: `edge_gap_12_mean ~= 6.36`, `htsr_alpha_mean ~= 1.31`

Interpretation:
- edge-gap shows clear regime sensitivity in this slice (lower stable-rank regimes show stronger edge dominance).
- HTSR alpha varies by regime but with a weaker monotone shape.

## Candidate transition sensitivity
Using stable-rank peak-change step as reference:
- edge-gap median lead: `0` steps (mean lead `33.3`)
- HTSR alpha median lead: `0` steps (mean lead `20.0`)

Interpretation:
- neither probe consistently fires earlier by median,
- both show bounded complementary movement around transition windows.

## Added-value / redundancy snapshot
Outcome correlations (Spearman, bounded slice):
- edge-gap vs eval accuracy: `~0.78`
- HTSR alpha vs eval accuracy: `~0.46`
- strongest baseline vs eval accuracy (`top1_energy_mean`): `~0.37`

Redundancy checks:
- edge-gap vs top1 energy Spearman: `~0.70` (substantial overlap)
- HTSR alpha vs spectral decay alpha Spearman: `~-0.27` (not a simple duplicate)

Interpretation:
- edge-gap carries strong signal in this slice, but overlaps materially with existing concentration measures.
- HTSR alpha appears complementary but weaker and less stable.

## Bounded conclusion for this stage
The probes add nontrivial bounded signal in this CPU checkpoint slice, but early-warning behavior and broader stability are not strong enough yet for unqualified promotion.
