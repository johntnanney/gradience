# OA-v1 vs OA-v2 Heuristic Comparison (Higher-Rank Sweep)

## Scope

- Controlled rank-r synthetic sweep using full left/right angle spectra.
- Ranks: `[4, 8, 16, 32]`
- Coefficients: `[(0.5, 0.5), (0.7, 0.3)]`
- Seeds per setting: `12`
- Cases: `6912`

## Key Results

- Overall Spearman:
  - OA-v1 vs true margin: `0.3957`
  - OA-v2 vs true margin: `0.8261`
- High-overlap positive-alignment Spearman:
  - OA-v1: `0.1799`
  - OA-v2: `0.7181`

## Interpretation

OA-v2 uses rank-r cross-term geometry and should be read as experimental.
OA-v1 remains the production advisory path until strict-naive field cross-check passes.
