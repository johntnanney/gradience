# Field Note — Pilot 3: Large Mixed-Task (DistilBERT)

## Context

Nine LoRA adapters on `distilbert-base-uncased` across four task types: sentiment (IMDB ×2, SST-2 ×2), emotion (tweet_eval/emotion ×1), news topic classification (ag_news ×2), and hate speech (tweet_eval/hate ×2). This is the largest inventory in the pilot — 36 theoretical candidate pairs (though muneeb-ai's PEFT incompatibility reduces the working set to 8 adapters / 28 pairs).

All TransferGraph adapters are r=1/alpha=1 (q+v only). The community adapters have varied configs: jmeneu r=1/alpha=32, myselfmankar r=16/alpha=32, NightPrince r=8/alpha=8. This is the most structurally heterogeneous inventory in the pilot series.

The inventory was designed to test whether Gradience handles a larger candidate space gracefully — producing useful task-boundary partitions, region groupings, and an actionable evaluate-first subset.

## Evidence bootstrap results

| Adapter | Dataset | Adapter score | Base score | Delta | Beats base? |
|---------|---------|--------------|------------|-------|-------------|
| muneeb-ai (IMDB, r=4) | — | ERROR | — | — | — |
| jmeneu (IMDB, r=1) | imdb | 0.836 | 1.000 | -0.164 | no |
| myselfmankar (SST-2, r=16) | sst2 | 0.886 | 0.470 | +0.416 | yes |
| NightPrince (SST-2, r=8) | sst2 | 0.714 | 0.528 | +0.186 | yes |
| TG emotion (r=1) | tweet_eval/emotion | 0.772 | 0.178 | +0.594 | yes |
| TG/JB173 ag_news (r=1) | ag_news | 0.912 | 0.218 | +0.694 | yes |
| TG/jaesun hate (r=1) | tweet_eval/hate | 0.520 | 0.472 | +0.048 | yes |
| TG/Aureliano hate (r=1) | tweet_eval/hate | 0.426 | 0.576 | -0.150 | no |
| TG/phailyoor ag_news (r=1) | ag_news | 0.884 | 0.208 | +0.676 | yes |

**Key observations:**
- **muneeb-ai** fails again with the same PEFT modules_to_save incompatibility from Pilot 1. Excluded.
- **jmeneu** triggers the same misleading base-score artifact (1.000 on IMDB) as in Pilot 1. Flagged_weak — correctly, given the reported negative delta, but for a spurious reason.
- **Aureliano hate** genuinely underperforms base (0.426 vs 0.576) — a real flagged_weak finding. This adapter's training lineage (source: Aureliano/distilbert-base-uncased-if) may not have been effective for hate detection.
- **jaesun hate** barely beats base (+0.048) — the same marginal pattern as the hate adapter in Pilot 2.

## Gradience stance (with evidence)

### Preflight summary

- 8 adapters (muneeb excluded), 28 pairs, **2 retained**, 93% reduction
- 6 adapters `eligible`, 2 `flagged_weak` (jmeneu, Aureliano)
- 2 excluded: jmeneu (weak, negative delta), Aureliano (weak, underperforms base)
- Same-task safe zones: 2 pairs retained (AG News pair, SST-2 pair)
- Cross-task caution: 6 task-boundary regions (AG_NEWS × SST-2, AG_NEWS × EMOTION, AG_NEWS × HATE, SST-2 × EMOTION, SST-2 × HATE, EMOTION × HATE)
- Evaluate-first subset: 2 pairs
- Summary: "QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 2."

### Same-task pairs retained

| Adapter A | Adapter B | Risk | Issue | Strategy |
|-----------|-----------|------|-------|----------|
| TG/JB173 ag_news | TG/phailyoor ag_news | low | none | linear |
| myselfmankar SST-2 | NightPrince SST-2 | medium | norm_imbalance | norm_equalized |

### Same-task pair excluded (weak source)

| Adapter A | Adapter B | Risk | Issue | Strategy | Note |
|-----------|-----------|------|-------|----------|------|
| TG/jaesun hate | TG/Aureliano hate | low | none | linear | Aureliano flagged_weak |

### QA detail

| Adapter | Status | Confidence | Adapter score | Base score | Delta |
|---------|--------|------------|--------------|------------|-------|
| jmeneu | flagged_weak | high | 0.836 | 1.000 | -0.164 |
| myselfmankar (SST-2) | eligible | high | 0.886 | 0.470 | +0.416 |
| NightPrince (SST-2) | eligible | high | 0.714 | 0.528 | +0.186 |
| TG emotion | eligible | high | 0.772 | 0.178 | +0.594 |
| TG/JB173 ag_news | eligible | high | 0.912 | 0.218 | +0.694 |
| TG/jaesun hate | eligible | high | 0.520 | 0.472 | +0.048 |
| TG/Aureliano hate | flagged_weak | high | 0.426 | 0.576 | -0.150 |
| TG/phailyoor ag_news | eligible | high | 0.884 | 0.208 | +0.676 |

### Aggregate pair statistics (28 pairs)

| Dimension | Breakdown |
|-----------|-----------|
| Risk | high: 10, medium: 12, low: 6 |
| Dominant issue | norm_imbalance: 21, none: 6, partial_redundancy: 1 |
| Strategy | audit_aware: 11, norm_equalized: 11, linear: 6 |
| Task type | cross-task: 25, same-task: 3 |

## What Gradience got right

1. **Two clean same-task pairs correctly retained.** The AG News pair (JB173 × phailyoor) and the SST-2 pair (myselfmankar × NightPrince) are exactly the pairs a practitioner should evaluate first. The AG News pair gets `low` risk / `linear` merge — the cleanest possible recommendation. The SST-2 pair gets `medium` risk / `norm_equalized` because r=16/alpha=32 vs r=8/alpha=8 creates a norm mismatch. Both correct.

2. **Weak adapters correctly excluded from the retain set.** The Aureliano hate adapter genuinely underperforms base, and Gradience excludes it. This means the hate-speech same-task pair (jaesun × Aureliano) drops out — not because of structural issues, but because one source is weak. Correct gatekeeping.

3. **93% candidate reduction is the best product outcome in the pilot.** From 28 candidate pairs to 2 evaluate-first candidates. A practitioner facing 28 merge possibilities gets a clear two-item priority list. This is the value proposition in action.

4. **Task-boundary partitioning produces 6 cross-task regions.** Every cross-task pair gets an advisory. The 6 regions correspond to the 4 task types' pairwise combinations (minus the same-task diagonal). The partitioning is clean and correct.

5. **The jmeneu base-score artifact is consistent across pilots.** Pilot 1 and Pilot 3 both flag jmeneu as weak with base=1.000. This is a repeatable finding — the same adapter produces the same misleading eval in both inventories. Gradience handles it consistently (flagged_weak both times).

## What Gradience got wrong or where it's limited

1. **IMDB and SST-2 treated as cross-task, but they're the same task.** Both are binary sentiment classification. The task-boundary detection fires because the eval_dataset labels differ ("imdb" vs "sst2"), but these are genuinely the same task with different data sources. A practitioner would know this. Gradience's metadata-level task detection can't see task equivalence across different datasets. In this inventory, if jmeneu were eligible, the jmeneu(IMDB) × myselfmankar(SST-2) pair would be flagged cross-task when it should arguably be same-task.

2. **The hate-speech pair is a missed opportunity.** jaesun × Aureliano is structurally clean (low risk, no dominant issue, linear merge) and same-task. But Aureliano is flagged_weak, so the pair drops out. The practitioner's action plan shows only 2 candidates, with no mention that a third pair was structurally promising but blocked by a weak source. The exclusion list says "Aureliano: weak source — low confidence" but doesn't connect this to the downstream impact on the hate-speech same-task pair.

3. **Norm imbalance dominates yet again (21 of 28 pairs).** The community adapters (r=8, r=16) vs TransferGraph (r=1) configuration gap produces the same norm_imbalance signal across almost every mixed pair. This is by now the third pilot in a row where norm_imbalance is the majority finding. It's a real signal — these adapters genuinely have different magnitude profiles — but it crowds out other structural features. A practitioner seeing "21 of 28 pairs: norm_imbalance" would learn relatively little from it.

4. **The jaesun hate adapter's `eligible` status is marginal.** At 0.520 on a binary task with base at 0.472, the delta is +0.048 — barely distinguishable from noise on a 500-sample eval. Same pattern as the Pilot 2 hate adapter. The evidence gate accepts any positive delta, which means marginal adapters pass consistently.

5. **No structural flags surfaced on any adapter.** Despite the structural heterogeneity (r=1 through r=16, different layer counts, different target modules), none of the 8 QA artifacts have structural flags. In Pilot 1, RAJESH got `high_rank_waste` and `low_utilization`. The absence here suggests the flag thresholds may not trigger on the particular rank/utilization profile of these adapters, or that the flag logic depends on adapter configurations that aren't present in this inventory.

## Notable structural findings

### The AG News pair: the cleanest merge candidate across all three pilots

TG/JB173 × TG/phailyoor: both r=1/alpha=1, same target modules, same task (ag_news), same backbone. Low risk, no dominant issue, linear merge recommended. Both adapters score >0.88 on ag_news. This is the one pair across all three pilots where Gradience's recommendation is unambiguously correct and actionable — and where a practitioner would most benefit from proceeding directly to evaluation.

### The SST-2 pair: norm mismatch on same-task adapters

myselfmankar (r=16/alpha=32) × NightPrince (r=8/alpha=8) both do SST-2 sentiment, but with very different capacity. The norm_equalized strategy accounts for the 2× rank and 4× alpha difference. The `partial_redundancy` finding (1 pair in the inventory) would ideally tell the practitioner whether the redundancy is in the task-relevant subspace — i.e., whether both adapters learned the same sentiment features or complementary ones.

### Cross-task pair risk variation is meaningful

Not all cross-task pairs are equally risky. SST-2 × hate pairs range from `low` to `high` risk depending on the specific adapters involved. The TG-to-TG cross-task pairs (same rank, same alpha) tend to be lower risk than the community-to-TG pairs (different rank/alpha). This separation is real: structural compatibility is orthogonal to task compatibility, and Gradience correctly separates the two signals.

## Product usefulness ratings

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Search reduction | **high** | 28 → 2 (93% reduction). Clear, actionable. Best in pilot series. |
| Interpretive clarity | **high** | Action plan is well-structured: exclude 2 weak, evaluate 2 safe, caution on 6 regions. |
| Trust usefulness | **high** | Evidence gate correctly identifies 2 weak adapters and 6 eligible. One marginal pass (jaesun). |
| Report usefulness | **high** | HTML report and action plan both informative. At 28 pairs, the summary layer adds real navigability value. |
| Large-inventory usefulness | **medium** | 28 pairs exercises the pair-counting logic, but doesn't trigger large-inventory-specific features (region maps, etc.) because the code threshold is higher. Still, the action plan format scales well to this size. |

## Key takeaways for the trial

1. **Gradience's core pipeline works at 28-pair scale.** The complete workflow — evidence bootstrap, adapter audit, pairwise merge audit, inventory summary, action plan, HTML report — runs end-to-end and produces actionable output. 93% candidate reduction with 2 clear priority pairs.

2. **The evidence gate is the most valuable product feature observed so far.** It correctly separates eligible from weak across the full range: genuine failures (Aureliano), misleading evals (jmeneu base artifact), marginal passes (jaesun), and strong performers (myselfmankar, TG ag_news adapters). Without it, Pilot 1 showed that everything gets excluded.

3. **Same-dataset task matching works; same-task-family matching doesn't.** IMDB ≠ SST-2 in Gradience's view, even though both are binary sentiment. This is a known limitation (metadata-driven, not content-driven) and is worth flagging for product development.

4. **Norm imbalance as a dominant signal reflects real configuration heterogeneity, but its diagnostic value saturates.** When 75% of pairs show the same issue, the signal stops being informative. A practitioner would benefit from knowing *which* pairs have the most severe imbalance, not just that most pairs have some.

5. **The "missed pair" problem — structurally clean but evidence-blocked pairs — deserves surface-level treatment in the action plan.** The hate-speech pair was the third same-task pair in this inventory, and it was structurally clean. But because one adapter is weak, it vanishes from the action plan entirely. Adding a "near-miss" section would help practitioners understand what they're losing and whether sourcing a better adapter could unlock additional merge candidates.
