# N134 Figure Generation

Three figure scripts. Each reads from `sidecar/results/n134/` and writes matching PDF + PNG (300 dpi) to `papers/n134_workshop/figures/`.

## Run

From repository root:

```sh
python scripts/n134/figures/fig_h1_decision.py
python scripts/n134/figures/fig_four_method_forest.py
python scripts/n134/figures/fig_layer_depth_trend.py
```

Scripts are independent; any order works. Each is idempotent (overwrites its own output). Output is byte-stable modulo font-hinting noise.

## Paper-number / script-name mapping

| Paper figure | Script | Output basename |
|---|---|---|
| Figure 1 | `fig_h1_decision.py` | `h1_decision` |
| Figure 2 | `fig_four_method_forest.py` | `four_method_forest` |
| Figure 3 | `fig_layer_depth_trend.py` | `layer_depth_trend` |

Paper-facing numbering comes from `\begin{figure}` ordering in LaTeX, not from filenames. Script and output names are descriptive so that dropping or reordering figures does not force a rename.

## Style

All three scripts import from `mpl_style.py`:
- `apply_style()` called at the top of `build_figure()`
- `save_figure(fig, basename)` writes both PDF and PNG
- `N134_RESULTS`, `PALETTE`, `COL_SINGLE`, `REF_COLOR`, `ANNOT_GREY` are shared constants

Paul-Tol bright palette chosen over matplotlib's `tab10` default for better colorblind-safe perceptual separation at small figure sizes. Explicit hex codes in `mpl_style.PALETTE` so regeneration is stable regardless of matplotlib default-palette changes.

Font: STIX Two Text preferred (serif, matches common LaTeX body fonts), with a fallback chain down to the system serif. PDF font type 42 (TrueType, editable text in PDF).

## Data dependencies

| Script | JSON input |
|---|---|
| `fig_h1_decision.py` | `sidecar/results/n134/analysis_h1.json` |
| `fig_four_method_forest.py` | `sidecar/results/n134/method_comparison.json` |
| `fig_layer_depth_trend.py` | `sidecar/results/n134/analysis_secondary.json` |

Every annotation number on every figure is extracted from its JSON input. No hardcoded values. If you change an analysis script and the JSON shape changes, the figure script will fail loudly rather than silently produce a wrong plot.

## RNG

`RNG_SEED = 20260420` defined in `mpl_style.py`. None of the three current figure scripts use randomness (all three are deterministic summarizations of the JSON data). The seed is declared in the shared module for any future figure that does need stochastic visualization (jitter, bootstrap resampling for visual CIs, etc.) — using a shared constant rather than an ad-hoc local seed keeps regeneration reproducible across scripts.
