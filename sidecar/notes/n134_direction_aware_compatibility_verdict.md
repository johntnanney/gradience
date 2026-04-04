# N134 — Direction-Aware Compatibility Verdict

- Generated: `2026-04-04T17:55:28+00:00`
- Cohort: `30` strict-naive pairs

## Result
- Best directional metric in intended slice: `dir_risk_signed_middle_conflict_max` (|rho|=0.3143, lift=0.1667).
- Best coarse metric in intended slice: `coarse_mean_agreement` (|rho|=0.4468, lift=0.1667).
- Preliminary interpretation: coarse summaries remain primary in this slice.

## Implications
- Keep top-energy/coarse summaries as default.
- Retain direction-aware metrics as bounded explanatory companions for fragile/ambiguous cases.
- No policy or threshold changes from this pass alone.
