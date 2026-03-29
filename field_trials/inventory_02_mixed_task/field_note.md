# Field Note — Pilot 2: Mixed-Task (RoBERTa-base)

## Context

Five LoRA adapters on `roberta-base` across four distinct tasks: hate speech (tweet_eval/hate), news topic classification (ag_news ×2), irony detection (tweet_eval/irony), and natural language inference (MNLI). This is the core mixed-task use case — the scenario Gradience is designed for. The two AG News adapters have different training lineage (one direct, one from a formality-pretrained source), providing a same-task pair to anchor the evaluation.

All adapters are r=1/alpha=1 (TransferGraph series) except yuuhan's MNLI adapter, which is r=8/alpha=16 — the only higher-rank adapter in the inventory. The irony adapter targets only 12 layers (6 transformer layers × q+v), while others target 24 layers (12 transformer layers × q+v).

Evidence bootstrap ran successfully on all 5 adapters. No PEFT incompatibilities.

## Evidence bootstrap results

| Adapter | Dataset | Adapter score | Base score | Delta | Beats base? |
|---------|---------|--------------|------------|-------|-------------|
| TG hate (r=1) | tweet_eval/hate | 0.486 | 0.426 | +0.060 | yes |
| TG ag_news (r=1) | ag_news | 0.938 | 0.218 | +0.720 | yes |
| TG irony (r=1) | tweet_eval/irony | 0.622 | 0.622 | +0.000 | no |
| yuuhan MNLI (r=8) | mnli | 0.842 | 0.350 | +0.492 | yes |
| TG ag_news-formality (r=1) | ag_news | 0.902 | 0.254 | +0.648 | yes |

**Note on irony adapter:** Delta is exactly 0.000 — the adapter produces identical accuracy to the base model on this 500-sample slice. Gradience classified it as `uncertain` (not `flagged_weak`), which is the correct intermediate status: we cannot confirm it helps, but we cannot confirm it hurts either.

**Note on hate adapter:** The delta is small (+0.060), and the absolute accuracy is low (0.486 on a binary task — barely above chance). Gradience still classified it as `eligible` because it does beat the base. This is worth watching — a marginal adapter that passes the evidence gate.

**Note on base scores:** The random-initialized classifier head on `roberta-base` scores 0.218–0.426 on the various tasks (depending on label distribution vs random class predictions). These are more reasonable than the Pilot 1 anomaly (base=1.000 on IMDB).

## Gradience stance (with evidence)

### Preflight summary

- 5 adapters, 10 pairs, **1 retained**, 90% reduction
- 4 adapters `eligible`, 1 `uncertain` (irony)
- 0 excluded
- Same-task safe zone: 1 pair (AG News × AG News-formality)
- Cross-task caution: 9 pairs across 6 task-boundary regions
- Evaluate-first subset: 1 pair (the same-task AG News pair)
- Summary: "Inventory is mostly explained by task boundary. Candidate space reduced from 10 pairs to 1 (90% reduction)."

### Pair-level detail (10 pairs, 5 adapters)

| Adapter A | Adapter B | Risk | Issue | Strategy | Advisory |
|-----------|-----------|------|-------|----------|----------|
| TG hate × TG irony | — | low | none | linear | cross-task |
| TG hate × TG ag_news | — | high | norm_imbalance | audit_aware | cross-task |
| TG hate × TG ag_news-formality | — | high | norm_imbalance | audit_aware | cross-task |
| TG hate × yuuhan MNLI | — | medium | norm_imbalance | norm_equalized | cross-task |
| TG ag_news × TG irony | — | high | norm_imbalance | audit_aware | cross-task |
| TG ag_news × yuuhan MNLI | — | high | norm_imbalance | audit_aware | cross-task |
| **TG ag_news × TG ag_news-formality** | — | **medium** | **partial_redundancy** | **norm_equalized** | **same-task** |
| TG irony × yuuhan MNLI | — | medium | norm_imbalance | norm_equalized | cross-task |
| TG irony × TG ag_news-formality | — | high | norm_imbalance | audit_aware | cross-task |
| yuuhan MNLI × TG ag_news-formality | — | high | norm_imbalance | audit_aware | cross-task |

### QA detail

| Adapter | Status | Confidence | Rank | Layers | Flags |
|---------|--------|------------|------|--------|-------|
| TG hate | eligible | high | 1 | 24 | none |
| TG ag_news | eligible | high | 1 | 24 | none |
| TG irony | uncertain | medium | 1 | 12 | none |
| yuuhan MNLI | eligible | high | 8 | 24 | none |
| TG ag_news-formality | eligible | high | 1 | 24 | none |

## What Gradience got right

1. **Same-task pair correctly identified and retained.** The two AG News adapters (TG ag_news × TG ag_news-formality) are correctly recognized as a same-task pair. No cross-task advisory. This is the retained pair — the only one Gradience recommends pursuing. This is exactly the right call.

2. **The retained pair's diagnosis is informative.** Medium risk, `partial_redundancy` (8 of 24 layers show redundancy). This makes sense: two adapters trained on the same task from related-but-different source models should share some learned structure. The recommended strategy (norm_equalized) is reasonable — handle the overlap without throwing it away.

3. **Cross-task boundaries detected on all 9 non-same-task pairs.** Every pair that crosses a task boundary gets the advisory. The task-boundary partition exactly matches the ground truth: hate ≠ ag_news ≠ irony ≠ MNLI.

4. **Irony adapter correctly classified as `uncertain`.** A 0.000 delta is genuinely ambiguous. Gradience doesn't exclude it (it might work), but doesn't confidently endorse it either. This is the right epistemic posture.

5. **The hate × irony pair is rated `low` risk with `none` as dominant issue.** Both are tweet-domain binary classifiers from the same TransferGraph pipeline, same rank, same alpha. Structurally they're similar. Gradience sees this correctly — but still applies the cross-task advisory because they were evaluated on different datasets. This is the right separation of concerns: structural compatibility is high, but task compatibility is uncertain.

6. **90% candidate reduction is a useful product outcome.** A practitioner facing 10 candidate pairs is told: "evaluate this one first." That's actionable workflow guidance.

## What Gradience got wrong or where it's limited

1. **Norm imbalance dominates again, but for the wrong reason.** 8 of 10 pairs show `norm_imbalance` as dominant issue. In Pilot 1, this was driven by rank/alpha heterogeneity (r=1/alpha=1 vs r=4/alpha=32). Here it's different: the yuuhan MNLI adapter is r=8/alpha=16, while all TransferGraph adapters are r=1/alpha=1. So every pair involving yuuhan shows norm_imbalance. But pairs between TransferGraph adapters with different layer counts (24 vs 12) also show norm_imbalance — the hate × irony pair is the only TG-vs-TG pair without it, because they happen to share the same 12-layer overlap structure. The question: is Gradience genuinely detecting problematic magnitude mismatches, or is it just flagging any configuration difference as norm_imbalance?

2. **The hate adapter's `eligible` status may be overgenerous.** 0.486 accuracy on a binary task is essentially chance. The adapter barely beats a random-initialized base (+0.060). A practitioner seeing "eligible, high confidence" might trust it more than warranted. The evidence gate currently has a simple threshold: beat the base by any margin → eligible. A more nuanced approach might downgrade marginal improvements to `uncertain`.

3. **The redundancy finding on the AG News pair is real but shallow.** 8 of 24 layers redundant, mean overlap 0.420. This tells the practitioner "these adapters overlap a third of the time." Useful, but a practitioner would want to know *which* layers are redundant and whether the non-redundant layers are complementary. The per-layer breakdown in the full pair report has this information, but the summary doesn't surface it.

4. **Task-boundary detection is metadata-driven, not content-driven.** Gradience flags cross-task pairs based on `eval_dataset` mismatch. This works correctly here. But it means the hate × irony pair gets the same advisory as the AG News × MNLI pair, even though hate/irony are much more task-adjacent (both tweet-domain binary classification) than news classification vs. NLI. A "task distance" metric would add nuance.

## Notable structural findings

### The hate × irony pair: low risk but cross-task

This is the most interesting pair in the inventory. Structurally, it's the cleanest: same rank, same alpha, same TransferGraph pipeline. Gradience rates it `low` risk with `none` as dominant issue and recommends `linear` merge. But the cross-task advisory fires because the datasets differ. In a real workflow, this would be a pair worth investigating — two tweet-domain binary classifiers might compose well. Gradience correctly does not block it; it just adds a caution note.

### The MNLI adapter as structural outlier

yuuhan's MNLI adapter is r=8/alpha=16 — 8× the rank and 16× the alpha of every other adapter. Every pair involving it shows heavy norm imbalance. This is the same dynamic as Pilot 1 (TransferGraph vs community adapters), now with a single outlier rather than half the inventory. It's a realistic scenario: a practitioner might have one high-capacity adapter and several lightweight ones.

## Product usefulness ratings

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Search reduction | **high** | 10 → 1 (90% reduction). Clear, actionable. |
| Interpretive clarity | **high** | Same-task pair identified, cross-task regions mapped, action plan coherent. |
| Trust usefulness | **high** | Evidence correctly changed irony adapter to `uncertain`. Marginal hate adapter may be overgenerous. |
| Report usefulness | **high** | HTML report and action plan both informative. First inventory where the output is genuinely useful to a practitioner. |
| Large-inventory usefulness | n/a | Not a large inventory |

## Key takeaways for the trial

1. **This is Gradience's best result so far.** The same-task detection, cross-task partitioning, candidate reduction, and action plan all work as designed. A practitioner would save time and avoid 9 unnecessary evaluations.

2. **The evidence bootstrap is essential.** Without behavioral evidence, all 5 adapters would be excluded (as in Pilot 1's v1 run). With evidence, Gradience can distinguish eligible from uncertain and make meaningful retention decisions.

3. **Norm imbalance is still the dominant structural signal, but it conflates several sources.** Configuration differences (r=1 vs r=8), layer-count mismatches (12 vs 24), and genuine magnitude differences all produce the same `norm_imbalance` label. Decomposing this signal would add interpretive value.

4. **The same-task pair exhibits partial redundancy — the first non-trivial structural finding.** Pilot 1 had no same-task pairs survive the evidence gate. Here, the AG News pair shows 33% layer redundancy. This is the kind of finding that justifies the spectral analysis: it tells the practitioner something they couldn't know without examining the weight geometry.

5. **The irony adapter's `uncertain` status demonstrates the evidence gate's calibration.** A 0.000 delta is genuinely ambiguous, and Gradience's three-way classification (eligible/uncertain/flagged_weak) handles it correctly. This is a meaningful improvement over binary pass/fail.

## Decision for remaining pilots

Proceed to Pilot 3 (large mixed-task, 9 distilbert adapters, 36 candidate pairs). This inventory tests whether Gradience scales — both computationally and in terms of interpretive value — when the candidate space is substantially larger.
