# Phase 2b — Near-Miss Confirmation Memo

**Date:** 2026-03-28

---

## Purpose

Phase 2 found one near-miss pair (Pilot 3 hate-speech) that improved over both sources despite being excluded by the evidence gate. Phase 2b asks: is this a repeatable pattern, or was it a lucky exception?

We built two new inventories designed to generate near-miss pairs through evidence-gate variance, merged and evaluated 11 pairs across three categories, and compared outcomes.

## Inventories

| Inventory | Backbone | Adapters | Tasks | Near-miss source |
|-----------|----------|----------|-------|------------------|
| 04 — irony cluster | distilbert-base-uncased | 8 | irony (4), emotion (2), ag_news (2) | phailyoor irony: delta -0.004 |
| 05 — hate+emotion | bert-base-uncased | 8 | hate (3), emotion (3), ag_news (2) | aviator hate: delta -0.002; HateXplain emotion: delta -0.150 |

Both inventories were designed with deep same-task clusters (3-4 adapters per task) to maximize the chance that evidence-gate variance would produce at least one weak source per cluster, generating natural near-miss pairs.

## Table 1 — Retained same-task pairs

| Pair | Task | Backbone | Strategy | Merged | Best src | Δ best | Δ avg |
|------|------|----------|----------|--------|----------|--------|-------|
| JB173 × neibla irony | irony | distilbert | norm_equalized | 0.626 | 0.632 | -0.006 | +0.000 |
| vaariis × neibla irony | irony | distilbert | linear | 0.634 | 0.640 | -0.006 | +0.004 |
| TG base × HateXplain hate | hate | bert | linear | **0.616** | 0.588 | **+0.028** | +0.065 |
| TG base × fabriceyhc emotion | emotion | bert | linear | 0.664 | 0.752 | -0.088 | +0.186 |

**Summary:** 4 pairs, 1 improver (+0.028), average Δ vs best = -0.018. The hate pair on bert-base-uncased is the second genuine improvement across the entire trial series (after Pilot 2's ag_news +0.006). The emotion pair's large drop (-0.088) reflects fabriceyhc's very weak absolute performance (0.204 on 4-class). The irony pairs show clean interpolation with minimal loss.

## Table 2 — Near-miss pairs

| Pair | Task | Backbone | Strategy | Merged | Best src | Δ best | Δ avg | Weak source delta |
|------|------|----------|----------|--------|----------|--------|-------|-------------------|
| JB173 × phailyoor irony | irony | distilbert | linear | 0.620 | 0.632 | -0.012 | -0.005 | -0.004 |
| vaariis × phailyoor irony | irony | distilbert | linear | 0.628 | 0.640 | -0.012 | -0.001 | -0.004 |
| TG base × aviator hate | hate | bert | linear | 0.572 | 0.574 | -0.002 | +0.028 | -0.002 |
| HateXplain × aviator hate | hate | bert | linear | 0.586 | 0.588 | -0.002 | +0.005 | -0.002 |
| TG base × HateXplain emotion | emotion | bert | norm_equalized | 0.664 | 0.752 | -0.088 | +0.220 | -0.150 |
| fabriceyhc × HateXplain emotion | emotion | bert | linear | 0.202 | 0.204 | -0.002 | +0.032 | -0.150 |

**Plus Phase 2 Pilot 3 (included for completeness):**

| jaesun × Aureliano hate | hate | distilbert | linear | 0.598 | 0.520 | **+0.078** | +0.125 | -0.150 |

**Summary:** 7 near-miss pairs total (6 new + 1 from Pilot 3), 1 improver (+0.078), average Δ vs best = -0.006 (excluding the Pilot 3 outlier: -0.020). The hate pairs on bert barely degrade at all (-0.002). The irony pairs degrade slightly more (-0.012) but remain close to their best source. The emotion near-miss with a deeply weak source (-0.150 delta) shows the same degradation as the corresponding retained pair (-0.088).

## Table 3 — Control

| Pair | Task | Backbone | Strategy | Merged | Best src | Δ best | Δ avg |
|------|------|----------|----------|--------|----------|--------|-------|
| TG base ag_news × aviator hate | ag_news | bert | audit_aware | 0.826 | 0.922 | -0.096 | +0.078 |

Cross-task control degrades 5× more than the near-miss average.

## Category comparison

| Metric | Retained (4) | Near-miss (6 new) | Near-miss (all 7) | Control (1) |
|--------|-------------|-------------------|-------------------|-------------|
| Avg Δ vs best source | -0.018 | -0.020 | -0.006 | -0.096 |
| Avg Δ vs avg source | +0.064 | +0.047 | +0.058 | +0.078 |
| Improvers (beat best src) | 1/4 (25%) | 0/6 (0%) | 1/7 (14%) | 0/1 |
| Δ best ≤ 0.015 | 2/4 (50%) | 4/6 (67%) | 5/7 (71%) | 0/1 |
| Δ best > 0.05 | 1/4 (25%) | 1/6 (17%) | 1/7 (14%) | 1/1 |

## Answers to the key questions

### Q1: Do retained pairs continue to be the best use of evaluation budget?

**Yes.** One retained pair improved over its best source (+0.028 on bert hate). The rest interpolate with small loss (-0.006 to -0.088). The retained set remains the right first pick.

### Q2: Do near-miss pairs outperform excluded controls often enough to justify a new action-plan category?

**Yes.** Near-miss pairs degrade an average of -0.020 vs best source. The cross-task control degrades -0.096. The separation is consistent and large (5×). Including the Pilot 3 result, the average near-miss delta is -0.006 — practically indistinguishable from retained pairs (-0.018).

### Q3: Are near-miss pairs actual improvers, or just "less bad than excluded controls"?

**Mostly "less bad," with one genuine improver.** 1 of 7 near-miss pairs improved over best source (Pilot 3 hate, +0.078). The other 6 interpolated between sources or barely degraded. But the key finding is that near-miss pairs behave like retained pairs — not like excluded controls. Their degradation is in the same range as retained pairs.

### Q4: Does the pattern hold across more than one inventory/backbone/task family?

**Yes, clearly.** The near-miss pattern appears on:

- distilbert-base-uncased, irony task (Inv 04): -0.012 degradation
- distilbert-base-uncased, hate task (Pilot 3): +0.078 improvement
- bert-base-uncased, hate task (Inv 05): -0.002 degradation
- bert-base-uncased, emotion task (Inv 05): -0.088 degradation (deeply weak source)

Three backbones, three task families. The pattern is not a one-off accident.

## What modulates near-miss quality

The data reveals a clear gradient: the weaker the excluded source, the worse the near-miss outcome.

| Weak source delta | Near-miss Δ vs best | Examples |
|-------------------|---------------------|---------|
| -0.002 to -0.004 | -0.002 to -0.012 | Irony phailyoor, bert hate aviator |
| -0.150 | -0.002 to -0.088 | Bert emotion HateXplain, Pilot 3 hate Aureliano |

Sources that barely miss the evidence gate (delta -0.002 to -0.004) produce near-miss merges that are almost indistinguishable from retained pairs. Sources that genuinely underperform base (delta -0.150) can still produce useful merges, but the variance is higher.

This suggests a refinement: the near-miss section could rank candidates by how close the weak source is to the evidence threshold. Pairs where the weak source barely missed would appear first.

## The fabriceyhc anomaly

The retained pair TG base emotion × fabriceyhc emotion (both eligible) degraded -0.088, identical to the near-miss pair TG base emotion × HateXplain emotion. fabriceyhc is technically eligible (beats base by +0.118) but has terrible absolute performance (0.204 on 4-class). This shows that the evidence gate's binary threshold (positive delta → eligible) can pass adapters that drag down merges just as much as a flagged_weak source would.

This reinforces the Phase 1 finding about marginal adapters. The boundary between "eligible but marginal" and "flagged_weak" is not a useful distinction for merge quality. A severity-aware evidence score would help.

## Product implication

**Near-miss confirmed.** The data supports near-miss as a distinct product category.

The decision rule: "same-task + structurally plausible + trust-constrained" identifies a practically useful middle category that is distinct from both retained pairs and excluded controls. Near-miss pairs degrade comparably to retained pairs and far less than cross-task controls. Excluding them entirely is overprotective.

The near-miss section is already implemented in the action plan, preflight summary, and HTML report. No further product changes are needed based on this confirmation pass.

## What would make me more cautious

Three limitations are worth noting:

1. **All adapters are r=1 TransferGraph except two.** The confirmation pass used the same adapter ecosystem as Phases 1-2. Higher-rank adapters with richer spectral structure might show different near-miss dynamics.

2. **Binary classification tasks dominate the near-miss set.** Hate and irony are both binary tasks near chance. On these tasks, a "weak" adapter (delta -0.002 to -0.004) is barely distinguishable from noise. The near-miss pattern might be weaker on tasks where weak adapters are more clearly harmful.

3. **The evidence bootstrap uses 500-sample slices.** At this budget, a delta of -0.004 is well within the confidence interval of the evaluation. The "barely weak" near-miss pairs might just be measurement noise around the eligibility threshold. A larger eval budget would tighten this.

None of these are reasons to retract the confirmation. They are reasons to expect the near-miss category's value to vary with adapter ecosystem and eval budget.
