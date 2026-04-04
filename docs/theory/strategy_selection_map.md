# Strategy Selection Map (Analytical Program)

## Goal

Partition observable space:

`(overlap, directional_agreement, frobenius_ratio, concentration)`

into regions where merge strategies are analytically favored for spectral
preservation.

## Current state

Scaffold only. Linear/rank-1 pieces are implemented; TIES/DARE remain in
semi-analytical derivation stage.

## Planned comparison

Map analytical regions against current verdict tree in `verdicts.py` and
identify:

- regions of alignment (tree justified),
- regions of mismatch (candidate threshold refinement),
- regions lacking enough analytical signal (keep heuristic/guarded).
