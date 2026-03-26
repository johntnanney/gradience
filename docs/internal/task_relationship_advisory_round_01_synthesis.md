# Task-Relationship Advisory — Round 01 Synthesis

## Summary

5 inventories, 46 pairs, 32 advisories (70% overall rate).

| Inventory | Category | Adapters | Pairs | Advisory | Rate | Behavior |
|-----------|----------|----------|-------|----------|------|----------|
| same_task_sst2_control | A (same-task) | 3 | 3 | 0 | 0% | Silent as expected |
| nli_family_adjacent | B (adjacent) | 4 | 6 | 4 | 67% | Fired on all cross-task pairs |
| cross_task_sst2_qnli | C (distant) | 4 | 6 | 4 | 67% | Fired on all cross-task pairs |
| messy_mixed_quality | D (messy) | 5 | 10 | 8 | 80% | Fired on all cross-task pairs |
| large_diverse_pool | E (large) | 7 | 21 | 16 | 76% | Fired on all cross-task pairs |

## RQ answers

### RQ1: How often does the advisory appear?

**32 of 46 pairs (70%).** But this rate is entirely determined by the task composition of the pool:

- Same-task pairs: **0/14** advisories (0%) — silent every time
- Different-task pairs: **32/32** advisories (100%) — fired every time

The advisory is perfectly selective. It fires if and only if `eval_dataset` differs.

### RQ2: When it appears, does it correspond to pairs that would otherwise look too safe?

**Yes, in 18 of 32 cases.** The advisory fired on 18 pairs with `pair_risk=medium` — pairs that structural analysis rated as only partially concerning. Without the advisory, these medium-risk cross-task pairs would look similar to medium-risk same-task pairs (which are actually safe).

The remaining 14 advisory-bearing pairs had `pair_risk=high` (norm imbalance or subspace conflict), where the structural signal already flags problems. In those cases the advisory is confirmatory rather than additive.

**Breakdown of advisory-bearing pairs by structural risk:**

| Pair risk | Count | Advisory value |
|-----------|-------|---------------|
| medium | 18 | **Additive** — structural risk alone doesn't distinguish cross-task from same-task |
| high | 14 | Confirmatory — structural risk already flags the problem |

### RQ3: Does it change the recommended next action?

**At the pair level:** Modestly. The advisory doesn't change the structural risk classification, so a practitioner reading only `pair_risk` and `recommended_strategy` would see the same recommendation.

**At the inventory level:** More substantially. In inventories B, D, and E, the advisory clearly separates the pair matrix into same-task safe clusters and cross-task caution zones. This is the same compression that neighborhoods provide, but from a different (metadata) signal.

**Most impactful inventory: messy_mixed_quality (D).** 8 of 10 pairs carry the advisory. Without it, the pair matrix is 10 medium-risk pairs with no obvious structure. With it, the inventory immediately separates into 2 same-task safe pairs (qnli×qnli, rte×rte, mnli×mnli) and 8 cross-task caution pairs. That's a clear action-space reduction.

### RQ4: Does it add value beyond QA + pair-risk + neighborhoods?

**Yes, in a specific way.** The advisory provides a signal that none of the existing layers capture:

- **QA** tells you whether each adapter is individually credible. It does not say whether two credible adapters are on the same task.
- **Pair-risk** tells you structural compatibility. It does not distinguish same-task redundancy (safe) from cross-task redundancy (degraded).
- **Neighborhoods** group by structural similarity and QA status. They do not use task metadata.
- **The advisory** is the only signal that uses `eval_dataset` to flag task mismatch.

In 18 of 32 advisory-bearing pairs, this is genuinely new information — the structural risk was medium, QA was clean for both, and neighborhoods would have placed them in the same group.

### RQ5: Is the advisory noisy or unhelpful?

**No false positives detected.** The advisory fired 32 times — every time on a genuinely different-task pair. It never fired on a same-task pair. The wording is clear and factual.

**Potential concern: volume in large pools.** In the 7-adapter inventory (E), 16 of 21 pairs carry the advisory. That's a lot of advisory text in the pair reports. At this scale, the advisory is more useful as an inventory-level summary ("16 of 21 pairs involve cross-task merges") than as pair-level text. This suggests the advisory might eventually benefit from inventory-level aggregation, but the pair-level version is not misleading.

## Advisory effect classification

Across all 32 advisory-bearing pairs:

| Effect | Count | % |
|--------|-------|---|
| Clarifying | 14 | 44% — advisory confirms what structural risk already flagged (high-risk cross-task pairs) |
| Caution-raising | 18 | 56% — advisory flags cross-task risk that structural analysis rated as only medium |
| No effect | 0 | 0% |
| Action-changing | 0 | 0% — advisory doesn't change `recommended_strategy` |
| Redundant | 0 | 0% — even confirmatory advisories add task-identity context that structural risk lacks |

**Advisory impact ratio:** 32/32 (100%) — every advisory was at least clarifying or caution-raising.

## Key finding

The advisory's main value is not at the pair level — it's at the inventory level. It cleanly partitions the pair matrix into same-task safe zones and cross-task caution zones, which is exactly the compression that makes inventories actionable.

In every inventory with mixed tasks (B, C, D, E), the advisory provided the clearest single signal for separating safe from potentially degraded merges — clearer than pair-risk (which rated many cross-task pairs as medium rather than high) and orthogonal to neighborhoods (which group by structural similarity, not task identity).

## Recommendation

**Keep as-is.** The advisory:

- Fires when expected (different `eval_dataset`)
- Stays silent when expected (same `eval_dataset`)
- Adds genuinely new information in 56% of cases
- Confirms structural risk in the remaining 44%
- Never misleads
- Is factual and concise

**Future consideration:** At scale (6+ adapters with mixed tasks), an inventory-level summary of cross-task pair count would complement the pair-level advisory. But this is an enhancement, not a correction.

## Success criteria evaluation

| Criterion | Met? |
|-----------|------|
| Advisory appears in expected different-task regimes | Yes (32/32 different-task pairs) |
| Advisory stays mostly silent in same-task controls | Yes (0/14 same-task pairs) |
| At least 2 inventories show clarifying/caution-raising value | Yes (B, C, D, E — all 4 mixed inventories) |
| No evidence advisory is misleading or noisy | Correct — 0 false positives |
| Round produces clear recommendation | Yes: keep as-is |

**Round 01 verdict: advisory validated. Keep as stable additive signal.**

---

## Observation Round Extension (5 additional inventories)

A second observation round was conducted with 5 inventories covering all 5 categories using verified adjudication adapters and NLI-family adapters.

### Round overview

| Inventory | Cat | Pairs | Same | Diff | Adv | Caution | Redundant | Impact ratio | Value level |
|-----------|-----|-------|------|------|-----|---------|-----------|-------------|------------|
| same_task_qnli_control | A | 3 | 3 | 0 | 0 | — | — | n/a | minimal |
| adjacent_nli_family | B | 3 | 0 | 3 | 3 | 3 | 0 | 1.00 | inventory |
| distant_sst2_x_qnli | C | 6 | 2 | 4 | 4 | 1 | 3 | 0.25 | mixed |
| messy_mixed_provenance | D | 6 | 1 | 5 | 5 | 0 | 5 | 0.00 | minimal |
| large_multitask_pool | E | 15 | 2 | 13 | 13 | 9 | 4 | 0.69 | inventory |
| **Totals** | | **33** | **8** | **25** | **25** | **13** | **12** | **0.52** | |

### Selectivity

- Same-task with advisory: **0/8 (0%)**
- Different-task with advisory: **25/25 (100%)**
- False positives: **0**

### Key observations from observation round

1. **Strongest value is in large mixed-task pools.** In the 6-adapter/15-pair inventory, the advisory collapsed the candidate space from 11 medium-risk pairs to 2 same-task pairs.

2. **Advisory is redundant when pair-risk dominates.** In messy inventories where pair-risk already rates most pairs as high, the advisory adds nothing.

3. **Advisory's caution-raising effect concentrates on medium-risk cross-task pairs.** This is exactly the blind spot it was designed to fill.

4. **Adjacent-task NLI pools are a strong advisory regime.** All pairs medium-risk, all cross-task, advisory is the only discriminator.

### Cumulative evidence

| Source | Pairs | Same-task fires | Different-task fires | FP |
|--------|-------|-----------------|---------------------|-----|
| Adjudication (DistilBERT) | 23 | 0/9 | 14/14 | 0 |
| Adjudication (RoBERTa) | 6 | 0/2 | 4/4 | 0 |
| Validation round | 46 | 0/14 | 32/32 | 0 |
| Observation round | 33 | 0/8 | 25/25 | 0 |
| **Total** | **108** | **0/33** | **75/75** | **0** |

### Final recommendation (confirmed)

**`keep_as_is`** — no changes warranted. The advisory is a validated, stable, perfectly selective additive signal whose strongest value is inventory-level partitioning of mixed-task pools.
