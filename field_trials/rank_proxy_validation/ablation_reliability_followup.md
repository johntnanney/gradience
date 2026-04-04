# Ablation Reliability Follow-Up (CPU, Informative Subset)

## Why This Pass Was Run
- Prior analysis showed ablation-aligned structure but materially lower operational stability than gradient.
- This pass tested whether ablation reliability improves before any scope expansion.

## Protocol
- Subset: informative/compressible families only (`sst2`, `imdb`).
- Adapters: 4.
- Ablation sample grid: `24, 48, 72` (non-saturated).
- Panels: 3 fixed + 3 random per setting.
- Modes: `hard_zero`, `attenuate`.
- Budgets for agreement checks: `0.35, 0.50, 0.65`.
- Tie-aware metrics added:
  - pairwise `kendall_tau_b`
  - pairwise Goodman-Kruskal `gamma`
  - top-k overlap at two levels (`q25`, `q50`)
- Low-information diagnostics added:
  - `flat_vector_fraction`
  - `low_information_vector_fraction`
  - `high_tie_vector_fraction`
- Runtime artifact: `/Users/john/code/gradience/field_trials/rank_proxy_validation/ablation_reliability_sweep.{json,md}`.

## Key Findings
1. Larger ablation sample panels improved rank-order stability.
- Mean pairwise Spearman delta (`72 - 24`) was positive in all mode/panel combinations:
  - `attenuate/fixed`: `+0.628`
  - `attenuate/random`: `+0.267`
  - `hard_zero/fixed`: `+0.272`
  - `hard_zero/random`: `+0.447`
- Tie-aware metrics moved in the same direction:
  - mean Kendall tau-b and mean gamma both increased from `24` to `72` in all 4 mode/panel slices.

2. Top-k stability improved in most combinations, but not all.
- Mean top-k overlap delta (`72 - 24`) at `q25` improved in 3/4 combinations.
- `attenuate/fixed` showed a small top-k decrease (`-0.056`) despite Spearman improvement.
- `q50` top-k overlap increased in all 4 mode/panel combinations.

3. Soft ablation (`attenuate`) is competitive and strongest at higher sample budget.
- At `72` samples, `attenuate` had the highest Spearman stability:
  - random panels: `0.888`
  - fixed panels: `0.878`

4. Fixed-panel vs random-panel consistency was mixed.
- No uniform winner across all sample levels and modes.
- This suggests panelization helps diagnostics, but does not by itself resolve instability.

5. Apparent proxy-policy agreement with OHT weakens as ablation reliability tightens.
- Mean allocation agreement vs OHT generally dropped as ablation sample count increased.
- Interpretation: some low-sample ablation/OHT agreement was likely noise-amplified.

6. Low-information ablation panels are now explicitly measured (and remain common).
- Averaged across mode/panel slices:
  - `low_information_vector_fraction`: `0.833` at 24 samples, `0.813` at 48, `0.563` at 72.
  - `flat_vector_fraction`: `0.542` at 24, `0.438` at 48, `0.313` at 72.
- This supports the same practical read: larger panels reduce degeneracy, but do not eliminate it.

## Reliability Caveat
- Spearman validity remains sparse in some settings because tied/flat vectors still occur.
- With pair-count reporting, valid pair fractions per setting ranged:
  - Spearman: `0.083` to `0.750`
  - Kendall tau-b: `0.083` to `0.750`
  - Gamma: `0.083` to `0.750`
- Treat this pass as directional evidence, not a final proxy verdict.

## Bounded Decision
- Keep ablation as an explanatory proxy candidate.
- For operational proxy use in the current CPU regime, gradient remains the stronger default.
- If this line is continued, prioritize:
  - slightly larger panel budgets around the `72` region,
  - tie-aware validity gates built on the new low-information flags,
  - and optional `rank_reduction` soft-ablation checks.
