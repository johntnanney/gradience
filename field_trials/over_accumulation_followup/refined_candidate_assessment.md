# OA Refined Candidate Assessment

- Decision: **keep_exploratory**
- Evidence strength: **suggestive**
- Matched comparisons available: 3
- Best baseline |Spearman|: 0.2195

## Best Refined Candidate
- Feature: `concentration_oa_mass_top3_fraction` (concentration_of_risk)
- Spearman(delta_vs_best): 0.5835
- LOO sign consistency: 1.000
- LOO median |Spearman|: 0.5897
- Median split gap: 0.1120

## Rationale
- Best refined candidate `concentration_oa_mass_top3_fraction` has Spearman=0.5835 (improvement over best baseline |rho|=0.3640).
- Stability: LOO median |rho|=0.5897, sign consistency=1.000, outlier_sensitive=False.
- Median split gap (high minus low) on delta_vs_best=0.1120; risk-direction-consistent=False.
- Refined candidates show some directional signal, but stability/coverage is insufficient for confident promotion.
- Control feature `control_source_score_gap` has materially stronger |Spearman|=0.8468, indicating non-OA confounding remains substantial.

## Decision Options
- `refine`: Adopt a candidate OA v2 feature in next implementation pass.
- `keep_exploratory`: Retain OA line as low-confidence exploratory; collect stronger evidence before promotion.
- `pause`: De-emphasize OA line until better data/theory appears.
