# Rank Proxy Validation Bounded Memo

## Frozen Proxy Policy (Bounded Regime)
- `proxy_gradient` is the primary operational comparator.
- `attenuate` is the companion ablation proxy for structural/explanatory comparison.
- `rank_reduction` expansion is paused in this encoder/compressible regime.

## Frozen Bounded Claim
- In the compressible encoder subset, Gradience's spectral rank policies are competitive fixed-budget compression guides.
- Their allocation structure aligns more with ablation-style importance than gradient-style importance.
- Gradient is currently the stronger operational proxy under the CPU protocol because it is substantially more stable under resampling.

## What Was Evaluated
- Full study cohort: 12 adapters across `sst2`, `imdb`, `tweet_eval/*`, and `ag_news`.
- Primary evaluable subset: compressible families only (`sst2`, `imdb`), where realized budget is materially below 1.0 (mean realized budget ~0.49).
- Non-informative subset: saturated families (`tweet_eval`, `ag_news`) where realized budget remains ~1.0, so allocation-method differences are not behaviorally testable.

## Main Findings (Bounded)
- Lead spectral policy in the current bounded regime: `oht` (best mean `delta_vs_uniform` among spectral policies in the compressible subset).
- Spectral-vs-proxy structure:
  - spectral alignment with `proxy_ablation` is stronger than with `proxy_gradient` (higher mean Spearman and top-k overlap).
  - spectral-vs-gradient agreement is consistently weak/negative in this subset.
- Compression outcome:
  - `proxy_gradient` still achieves the best mean compression outcome in the compressible subset (`delta_vs_uniform`), above `oht` and above `proxy_ablation`.

## Interpretation
- Current evidence supports a competitive spectral advisor story, not a dominant one.
- Spectral policies are viable and non-trivial under matched budgets.
- Spectral signals currently appear closer to ablation-style importance structure than gradient-style structure.
- Outcome leadership remains with gradient in the bounded compressible subset, so stronger advisor claims are not yet justified.

## Non-Informative Context
- `tweet_eval` and `ag_news` are retained for coverage but excluded from primary policy interpretation in this pass because allocation did not materially change effective rank budget.

## Decision Status
- Keep the line active with bounded confidence.
- Continue emphasizing informative-subset analyses for interpretation.
- Treat broader claims as pending until larger compressible cohorts and stronger control slices are added.

## Follow-Up Diagnostic Pass (Ablation vs Gradient)
- A focused informative-subset investigation was run (`sst2` + `imdb`, 6 adapters, 3 resampling repeats, CPU-only).
- Proxy stability under resampling:
  - gradient proxy ranks were substantially more stable than ablation (mean pairwise Spearman ~0.903 vs ~0.400).
  - top-k repeatability was similar but still slightly higher for gradient.
- Outcome concentration:
  - gradient-over-ablation gains were highly concentrated (top-2 adapters contribute ~0.90 of absolute gap mass).
  - family split is mixed: net positive in `sst2`, slightly negative in `imdb`.
- Top-k budget behavior:
  - gradient vs ablation top-k overlap is low-to-moderate, especially in `sst2`.
  - this suggests similar directional structure does not imply similar budget allocation behavior within selected layers.

## Implication for the Main Question
- The current evidence is consistent with: ablation may be conceptually aligned with spectral structure, but in this measurement setup it behaves as a noisier allocation target than gradient.
- This supports keeping ablation-alignment as an explanatory signal while treating gradient as the stronger operational compression target in the current bounded regime.

## Ablation Reliability Follow-Up (Non-Saturated Sweep)
- A follow-up reliability sweep was run on the informative subset with non-saturated panel sizes (`24, 48, 72`), `3` fixed panels, `3` random panels, and modes `hard_zero` + `attenuate`.
- Tie-aware reliability metrics were added:
  - pairwise `kendall_tau_b`
  - pairwise Goodman-Kruskal `gamma`
  - top-k overlap at `q25` and `q50`
- Low-information flags were added:
  - `flat_vector_fraction`
  - `low_information_vector_fraction`
  - `high_tie_vector_fraction`
- Stability improved with larger panel sizes:
  - pairwise Spearman deltas (`72 - 24`) were positive in all mode/panel slices (`+0.267` to `+0.628`).
- Tie-aware metrics moved in the same direction:
  - pairwise Kendall tau-b and gamma deltas (`72 - 24`) were positive in all mode/panel slices.
- `attenuate` at `72` samples gave the strongest observed ablation stability (mean Spearman ~`0.888` random, ~`0.878` fixed).
- Fixed vs random panel consistency remained mixed; neither dominated across all settings.
- Agreement-to-OHT generally decreased as panel size increased, suggesting part of low-sample agreement was noise-amplified.
- Caveat: valid pair fractions remain bounded by tied/flat vectors (Spearman/Kendall/Gamma valid fractions ranged `0.083` to `0.750` by setting).

### Reliability-Pass Decision
- Keep ablation as an explanatory proxy candidate.
- Keep gradient as the operational default proxy under the current CPU protocol.
- If continued, prioritize tie-aware stability diagnostics and optional `rank_reduction` soft-ablation checks.

## Rank-Reduction Soft-Ablation Pilot
- A bounded pilot added `rank_reduction` (`rank_retain_ratio=0.5`) to the same informative cohort/panel/budget protocol.
- Result: `rank_reduction` showed severe low-information collapse (`mean low-information fraction = 1.0`, high flat/tie burden) and low valid-pair coverage (`mean valid-pair fraction = 0.125`), despite nominally high correlation metrics.
- Interpretation: at this retain ratio, `rank_reduction` is not yet a reliable operational ablation target in this regime.
- Decision: keep as exploratory; no proxy-role change and no claim escalation.

## Rank-Reduction Retain-Ratio Rerun (`0.75` / `0.85`)
- A strict rerun was executed with `rank_reduction` only at retain ratios `0.75` and `0.85` under the same cohort/panels/budgets/tie-aware protocol.
- Outcome:
  - degeneracy did not improve; both reruns remained fully low-information in aggregate (`mean low-info = 1.000`),
  - valid pair coverage collapsed to `0.000` for both reruns,
  - tie-aware stability metrics became non-evaluable (`n/a`) due flat vectors.
- Bounded decision from this rerun:
  - stop expanding rank-reduction for now in this regime,
  - keep `gradient` as operational default,
  - keep `attenuate` as explanatory ablation companion,
  - keep `hard_zero` as a sanity probe.

## Consolidation Status
- This ablation branch is now consolidated into bounded policy/docs.
- Further CPU spend on this branch is not recommended unless a narrowly scoped new question is introduced.
