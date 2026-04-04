# Spectral Edge-Gap Probe (Bounded, CPU-Only)

## Purpose
Define a cheap edge-dominance probe that captures how strongly the top singular mode dominates nearby leading modes.

## Primary metric
- `edge_gap_12 = sigma_1 / sigma_2`

## Companion metrics
- `edge_gap_13 = sigma_1 / sigma_3` (if available)
- `sigma1_share_top3 = sigma_1 / (sigma_1 + sigma_2 + sigma_3)` (if available)

These remain bounded additions and are not expanded into a large metric family.

## Inputs
- Saved singular values per layer/checkpoint
- Positive values only; sorted descending before computation

## Outputs
- `edge_gap_12`
- `edge_gap_13` (optional)
- `sigma1_share_top3` (optional)
- `edge_valid` boolean
- `warning` for insufficient or degenerate spectra

## Interpretation
- Higher `edge_gap_12` indicates stronger top-mode dominance
- Lower `edge_gap_12` indicates less separation between top two modes

This probe is descriptive. It does not, by itself, prove beneficial or harmful behavior.

## Guardrails
- Use as a complementary observable alongside stable/effective rank and energy concentration
- Keep conclusions bounded to the analyzed regime
