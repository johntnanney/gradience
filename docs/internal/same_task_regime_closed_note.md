# Same-Task Regime — Closed

## Status

**Closed.** No actionable blind spot found. No further same-task rescue logic is justified on small encoder models.

## Evidence

| Study | Stressor | Pairs | Material degradations | Worst delta |
|-------|----------|------:|----------------------:|------------|
| Training-style variation | rank, alpha, dropout, checkpoint | 15 | 0 | 3.4pp |
| Domain shift (sentiment) | movies, restaurants, products | 15 | 0 | 2.2pp |
| Source-strength asymmetry | 19.8pp performance range | 15 | 0 | 2.4pp |
| **Total** | | **45** | **0** | **3.4pp** |

## Interpretation

Same-task pairs on small encoder models (distilbert-base-uncased, roberta-base) are broadly safe across all tested same-task stressors. The worst observed merge degradation was 3.4pp — mild and inconsistent across seeds.

The main predictor of merge failure in the current evidence base is **task identity**, not training style, domain, or source strength. The task-relationship advisory addresses this boundary. No additional same-task signal is warranted.

## What this means for the project

- Same-task blind-spot hunting is no longer an active research line
- Future effort should focus on cross-task regimes, mixed-task inventory interpretation, and larger-scale evidence
- The same-task row in the regime map is marked as "confirmatory / closed"
- This decision can be revisited if future evidence on larger models or non-GLUE tasks produces different results

## Provenance

- `docs/internal/training_style_blind_spot_results.md`
- `docs/internal/domain_shift_blind_spot_results.md`
- `docs/internal/source_strength_blind_spot_results.md`
