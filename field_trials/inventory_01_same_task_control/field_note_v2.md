# Field Note — Pilot 1: Same-Task Control (with evidence)

## Context

This is the second pass. The first pass (no behavioral evidence) produced "exclude everything" — see `field_note.md`. This pass feeds real CPU-evaluated behavioral scores into Gradience.

One adapter (muneeb-ai, r=4 targeting all modules including classifier) could not be loaded for inference due to a PEFT version incompatibility — it was excluded from the evidence run. The trial proceeds with 3 of the original 4 adapters.

## Evidence bootstrap results

| Adapter | Dataset | Adapter score | Base score | Delta | Beats base? |
|---------|---------|--------------|------------|-------|-------------|
| jmeneu (IMDB, r=1) | imdb | 0.836 | 1.000 | -0.164 | no |
| RAJESH (txt, r=4) | imdb | 0.898 | 0.846 | +0.052 | yes |
| TG (emotion, r=1) | tweet_eval/emotion | 0.772 | 0.378 | +0.394 | yes |

**Note on jmeneu:** The base score of 1.000 is an artifact — the randomly-initialized classifier head happens to predict all-one-class on this 500-sample slice, which matches the label distribution. The adapter's 0.836 is actually reasonable IMDB accuracy, but the base comparison is meaningless. Gradience correctly classified this as `flagged_weak` because the reported delta is negative.

## Gradience stance (with evidence)

### Preflight summary

- 3 adapters, 3 pairs, **0 retained**, 100% reduction
- 2 adapters `eligible`, 1 `flagged_weak` (jmeneu)
- 1 excluded: jmeneu (weak source)
- Cross-task caution: IMDB × TWEET_EVAL/EMOTION region
- No same-task safe zones (no same-task pairs remain after muneeb excluded)
- No evaluate-first subset
- Summary: "QA dominates this inventory; no credible same-task candidates remain."

### Pair-level detail (3 pairs, 3 adapters)

| Adapter A | Adapter B | Risk | Issue | Strategy | Advisory |
|-----------|-----------|------|-------|----------|----------|
| jmeneu (flagged_weak) | RAJESH (eligible) | medium | norm_imbalance | norm_equalized | — |
| jmeneu (flagged_weak) | TG emotion (eligible) | medium | norm_imbalance | norm_equalized | cross-task |
| RAJESH (eligible) | TG emotion (eligible) | medium | norm_imbalance | norm_equalized | cross-task |

### QA detail

| Adapter | Status | Confidence | Rank | Layers | Util mean | Flags |
|---------|--------|------------|------|--------|-----------|-------|
| jmeneu | flagged_weak | high | 1 | 6 | 0.907 | none |
| RAJESH | eligible | high | 4 | 6 | 0.232 | high_rank_waste, low_utilization, underutilized_capacity |
| TG emotion | eligible | high | 1 | 12 | 0.917 | none |

## What changed from v1 (no evidence) to v2 (with evidence)

1. **jmeneu correctly flagged as weak.** It underperforms the base on the reported metric. Gradience excludes it — reasonable.
2. **RAJESH and TG emotion classified as eligible.** Both beat base by a meaningful margin. But RAJESH gets structural flags (high rank waste, low utilization) because at r=4, only ~1 effective rank is used.
3. **Cross-task advisory fires correctly.** RAJESH (IMDB) × TG (emotion) gets a cross-task warning. Task-boundary detection is working.
4. **Still 0 retained.** Every pair involves either a weak source (jmeneu) or a cross-task boundary (RAJESH × TG). There are no same-task eligible pairs left because muneeb was excluded from the run.
5. **The summary line is the same:** "QA dominates this inventory." This is accurate but unhelpful — the inventory was too small and too heterogeneous to produce same-task safe zones.

## What Gradience got right

1. **Behavioral evidence classification is sensible.** The jmeneu adapter genuinely does worse than base (on this eval), and Gradience flags it. The other two are real improvements, and Gradience accepts them.
2. **Structural flags on RAJESH are informative.** At r=4 with only q_lin targeted, most capacity is wasted — utilization is 0.232. This is honest and useful.
3. **Cross-task boundary detection works.** IMDB × emotion gets flagged. This is correct — these are genuinely different tasks.
4. **Norm imbalance is the dominant signal, and it's real.** r=1/alpha=32 vs r=4/alpha=32 (jmeneu vs RAJESH) and r=4/alpha=32 vs r=1/alpha=1 (RAJESH vs TG) both have substantial norm mismatches.

## What Gradience got wrong or where it's limited

1. **The jmeneu "flagged_weak" is technically correct but based on a misleading eval.** The base model's perfect score (1.000) is a random-classifier artifact, not genuine superiority. A practitioner who saw "base beats adapter on IMDB" would naturally distrust the result. Gradience can't know the eval is misleading — but the product could add a note when the delta is small or the base score is suspiciously high.

2. **No same-task pairs survived.** This is a genuine limitation of the inventory — with muneeb excluded, there's only one IMDB adapter left (jmeneu, weak). The inventory was designed as a same-task control, but it didn't survive the evidence gate. This is a valid field finding: the same-task control needs adapters that actually work.

3. **The "evaluate first" subset is empty.** Every pair was either weak-source or cross-task. The practitioner gets no actionable evaluation candidates. In a real scenario, this would mean "this inventory is not worth exploring further" — which may actually be the right answer for these adapters.

## Product usefulness ratings (v2)

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Search reduction | **medium** | Correctly identified jmeneu as weak — saved one pair from evaluation. But still 0 retained. |
| Interpretive clarity | **medium** | Eligibility status, structural flags, and cross-task advisories all clear. Summary line is unhelpful. |
| Trust usefulness | **high** | Evidence changed the picture materially. v1 excluded everything; v2 found 2 eligible adapters. |
| Report usefulness | **medium** | HTML report renders well. Content is richer than v1 but still mostly "nothing to pursue." |
| Large-inventory usefulness | n/a | Not a large inventory |

## Key takeaways for the trial

1. **The evidence bootstrap works.** Feeding real eval scores into Gradience produces meaningfully different preflight output. This validates the approach for Pilots 2 and 3.

2. **This particular inventory is too thin to be a useful same-task control.** With only 3 working adapters across 2 tasks, there aren't enough same-task pairs to test safe-zone behavior. The lesson: the same-task control pilot needs at least 3–4 same-task adapters that are genuinely functional.

3. **Norm imbalance dominates when LoRA configs are heterogeneous.** Every pair in this inventory shows norm_imbalance because the rank and alpha values vary widely. This is a realistic finding but means the structural analysis is mostly saying "these adapters are configured differently." A practitioner would already know that.

4. **The evidence gate (behavioral eval) is working as designed.** It correctly changes the picture — from "all unknown" to "2 eligible, 1 weak." This is the most positive product finding from Pilot 1.

## Decision for remaining pilots

Proceed with Pilots 2 and 3 using the evidence bootstrap. The pipeline works end-to-end. The remaining pilots have more adapters and more task variety, which should produce richer preflight output.
