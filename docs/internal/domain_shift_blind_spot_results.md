# Domain-Shift Blind Spot Study — Results

## Study setup

6 binary sentiment adapters on distilbert-base-uncased, 3 domains x 2 seeds:

| Adapter | Domain | Seed | Own-domain acc | Cross-domain range |
|---------|--------|------|---------------|-------------------|
| sst2_s42 | SST-2 (movies) | 42 | 0.836 | 0.854–0.874 |
| sst2_s7 | SST-2 (movies) | 7 | 0.846 | 0.850–0.864 |
| yelp_s42 | Yelp (restaurants) | 42 | 0.888 | 0.810–0.862 |
| yelp_s7 | Yelp (restaurants) | 7 | 0.892 | 0.810–0.862 |
| amazon_s42 | Amazon (products) | 42 | 0.874 | 0.820–0.884 |
| amazon_s7 | Amazon (products) | 7 | 0.876 | 0.830–0.888 |

All verified above base (margins 0.31–0.39). All 15 pairwise merges evaluated on all 3 domains.

## Main results

| Category | Pairs | Safe | Mildly degraded | Materially degraded | Avg max delta |
|----------|-------|------|-----------------|--------------------|--------------:|
| Same-domain | 3 | 3 | 0 | 0 | +0.006 |
| Cross-domain | 12 | 9 | 3 | 0 | +0.010 |
| **All** | **15** | **12** | **3** | **0** | **+0.009** |

## Key finding: blind_spot_not_found

Same-task, different-domain merges are overwhelmingly safe for binary sentiment classification across movies, restaurants, and products. **No materially degraded merges.** The 3 mildly degraded pairs (1.8–2.2pp) all involve sst2_s7 — likely a seed-specific effect rather than a domain-shift pattern.

## Detailed results

| Pair | Type | SST-2 | Yelp | Amazon | Max delta | Outcome |
|------|------|-------|------|--------|-----------|---------|
| sst2_s42 × sst2_s7 | same | 0.836 | 0.866 | 0.848 | +0.010 | safe |
| yelp_s42 × yelp_s7 | same | 0.810 | 0.890 | 0.866 | +0.002 | safe |
| amazon_s42 × amazon_s7 | same | 0.832 | 0.882 | 0.874 | +0.006 | safe |
| sst2_s42 × yelp_s42 | cross | 0.822 | 0.888 | 0.870 | +0.014 | safe |
| sst2_s42 × amazon_s42 | cross | 0.826 | 0.882 | 0.880 | +0.010 | safe |
| yelp_s42 × amazon_s42 | cross | 0.826 | 0.892 | 0.876 | -0.002 | safe |
| yelp_s42 × amazon_s7 | cross | 0.834 | 0.892 | 0.876 | +0.000 | safe |
| yelp_s7 × amazon_s42 | cross | 0.822 | 0.892 | 0.866 | +0.008 | safe |
| yelp_s7 × amazon_s7 | cross | 0.828 | 0.892 | 0.872 | +0.004 | safe |
| sst2_s42 × yelp_s7 | cross | 0.824 | 0.888 | 0.862 | +0.012 | safe |
| sst2_s42 × amazon_s7 | cross | 0.830 | 0.882 | 0.880 | +0.006 | safe |
| sst2_s7 × amazon_s7 | cross | 0.834 | 0.878 | 0.874 | +0.012 | safe |
| sst2_s7 × yelp_s42 | cross | 0.828 | 0.896 | 0.872 | +0.018 | mildly |
| sst2_s7 × yelp_s7 | cross | 0.824 | 0.884 | 0.870 | +0.022 | mildly |
| sst2_s7 × amazon_s42 | cross | 0.832 | 0.866 | 0.880 | +0.018 | mildly |

## Why domain shift didn't matter here

Binary sentiment classification learns a broadly shared feature: positive vs negative polarity. The feature transfers well across movies, restaurants, and products — as shown by the high cross-domain performance of individual adapters (0.81–0.89 on out-of-domain eval).

When two adapters already represent similar features in similar subspaces, merging them is safe regardless of training domain. This is exactly what the structural pair-risk analysis predicted (all 15 pairs rated medium, no high-risk pairs).

## Task advisory behavior

| Metric | Count |
|--------|-------|
| Same-domain pairs (no advisory) | 3/3 ✓ |
| Cross-domain pairs with advisory | 12/12 ✓ |

The advisory correctly fired on all cross-domain pairs because they have different `eval_dataset` values (sst2_dev, yelp_test, amazon_test). But in this case, the advisory is **overcautious** — it warns about cross-domain merges that are actually safe.

This is the first regime where the advisory produces **false caution**: structurally and behaviorally safe merges that get flagged because eval_dataset differs.

## What this means for the advisory

The advisory remains useful as a general-purpose inventory-level partitioning signal. But this study shows a regime where it overcalls: **same broad task, different narrow domain, high cross-domain transfer.** The advisory cannot distinguish "different task" from "different domain of the same task."

This is not a bug — the advisory makes no task-semantic claims. It reports a metadata fact (eval_dataset differs). But it means the advisory's caution should be treated as "worth checking" rather than "likely degraded."

## Pair-risk behavior

All 15 pairs received `pair_risk=medium`. Pair-risk did not distinguish same-domain from cross-domain pairs, and it didn't need to — all pairs were safe.

## Recommendation

**`blind_spot_not_found`**

Domain shift within binary sentiment classification does not create a blind spot. Cross-domain merges are nearly as safe as same-domain merges in this regime. The task's learned features transfer well across review domains.

### Important caveat

This result is specific to binary sentiment, which is a broad-feature, high-transfer task. Domain shift may matter more for:
- tasks with domain-specific decision boundaries (e.g., medical NER vs legal NER)
- tasks where domain-specific vocabulary is critical
- tasks where label semantics shift across domains

Those would be worth testing in a future study, but they are not the same-task-different-domain regime — they are closer to different tasks with shared names.

## Implication for the regime map

No new row needed. The same-task row is further strengthened:

> Same-task, all eligible → workflow confirmatory. Neither training-style variation nor domain shift within high-transfer tasks creates a blind spot.

The advisory's overcaution in this regime is noted but does not warrant a logic change — it correctly reports a metadata difference, and practitioners can decide whether to heed the caution.
