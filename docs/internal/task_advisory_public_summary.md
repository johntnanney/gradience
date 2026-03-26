# Task-Relationship Advisory — Public Summary

## The regime boundary

On small encoder models, the most important factor in predicting merge safety is task identity. Same-task pairs are broadly safe — confirmed across 45 pairs spanning training-style variation, domain shift, and source-strength asymmetry, with 0 material degradations. Cross-task pairs are where meaningful failure modes appear.

## The blind spot it addresses

Gradience's spectral pair-risk analysis measures structural compatibility: norm ratios, subspace overlap, directional agreement. When two adapters are structurally similar, pair-risk rates them as compatible — even if they were trained on different tasks. Verified adjudication confirmed this: cross-task pairs rated "redundant" by pair-risk degraded the weaker task by 5-12pp after merging.

## The fix

A metadata check that is now part of the stable interpretive layer. When two adapters have different `eval_dataset` values in their QA artifacts, the merge report includes a `task_relationship_advisory`. This is additive — it does not change pair-risk, strategies, or recommendation logic.

## The evidence

| Metric | Result |
|--------|--------|
| Total advisory checks | 108 |
| Same-task pairs with advisory | 0/33 (0%) |
| Different-task pairs with advisory | 75/75 (100%) |
| False positives | 0 |
| Backbones tested | 2 (distilbert, roberta) |
| Same-task blind-spot studies | 3 (45 pairs, 0 material degradations) |

## The interpretation

The advisory is strongest at the inventory level. In mixed-task pools, it partitions the pair matrix into same-task safe zones and cross-task caution zones. In observation testing, it collapsed 11 medium-risk candidates to 2 actionable same-task pairs. Its silence on same-task pairs is itself evidence — confirmed by 3 blind-spot studies showing same-task merges are robustly safe.

## Recommended public framing

Task identity is the key regime boundary for LoRA merge safety on small encoder models. A simple metadata check — already present in Gradience's QA artifacts — cleanly separates the safe same-task regime from the cross-task regime where structural analysis alone is insufficient. This advisory is now part of the stable interpretive layer.
