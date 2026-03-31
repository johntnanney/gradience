# Field Note — Campaign A, Inventory A-01

**Inventory:** `campaign-a01-sentiment-family-distilbert`
**Date:** 2026-03-29
**Backbone:** distilbert-base-uncased
**Question:** Is exact task identity too strict for practically similar sentiment tasks (SST-2 vs IMDB)?

---

## Adapter Pool

| Adapter | Task | Dataset | Score | Delta vs Base | Status |
|---------|------|---------|-------|---------------|--------|
| myselfmankar/distilbert-base-sst2-lora | sentiment | sst2 | 0.884 | +0.382 | eligible |
| rambodazimi/distilbert-base-uncased-finetuned-LoRA-SST2 | sentiment | sst2 | 0.902 | +0.424 | eligible |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | sentiment | imdb | 0.876 | +0.398 | eligible |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | sentiment | imdb | 0.856 | +0.358 | eligible |

All four adapters eligible with high confidence. Data sampling uses shuffled split (seed=42) after discovering IMDB test[:500] is label-sorted.

**Dropped:** muneeb-ai/distilbert-base-uncased-lora-imdb-sentiment — PEFT modules_to_save incompatibility (all 8 modules targeted including classifier).

## Gradience Preflight Stance

4 adapters, 6 pairs. Gradience correctly triggers cross-task advisory on all 4 SST-2 x IMDB pairs (different `eval_dataset` strings: "sst2" vs "imdb").

| Pair Type | Count | Risk Distribution | Advisory |
|-----------|-------|-------------------|----------|
| SST-2 x SST-2 (same-task) | 1 | medium | none |
| SST-2 x IMDB (cross-dataset) | 4 | low(1), medium(1), high(2) | cross-task |
| IMDB x IMDB (same-task) | 1 | low | none |

Risk variation within cross-dataset pairs tracks structural properties (rank mismatch, norm imbalance), not task identity. The cross-task advisory is purely string-match driven.

## Merge Evaluation Results

| Pair | Role | Eval On | Strategy | Merged | Best Src | Delta |
|------|------|---------|----------|--------|----------|-------|
| SST-2 x SST-2 | retained | sst2 | audit_aware | 0.876 | 0.902 | -0.026 |
| SST-2(m) x IMDB(d) | family_test | sst2 | audit_aware | 0.880 | 0.902 | -0.022 |
| SST-2(m) x IMDB(d) | family_test | imdb | audit_aware | 0.868 | 0.876 | -0.008 |
| SST-2(r) x IMDB(w) | family_test | sst2 | uniform_linear | 0.852 | 0.902 | -0.050 |
| SST-2(r) x IMDB(w) | family_test | imdb | uniform_linear | 0.870 | 0.876 | -0.006 |
| IMDB x IMDB | retained | imdb | uniform_linear | 0.868 | 0.876 | -0.008 |

Abbreviations: (m)=myselfmankar, (d)=dipanjanS, (r)=rambodazimi, (w)=wt-golf.

## Key Observations

**1. Same-family cross-dataset pairs behave like retained pairs, not like cross-task controls.**

Category averages:
- Retained (same-task/dataset): avg Δ = -0.017 (n=2)
- Family test (SST-2 x IMDB): avg Δ = -0.022 (n=4)
- Phase 2 cross-task controls: avg Δ = -0.047 (reference)

The family-test average (-0.022) falls squarely within the retained range, not the cross-task control range. The gap between family-test and retained averages is only 0.005 — well within noise.

**2. Eval-dataset direction matters less than structural compatibility.**

When evaluated on IMDB (the simpler binary sentiment task with longer texts), cross-dataset merges show only -0.006 to -0.008 degradation. On SST-2 (shorter, more ambiguous sentences), degradation ranges wider (-0.022 to -0.050). The worst case (-0.050 on SST-2) comes from the rambodazimi x wt-golf pair, where wt-golf is a q_lin-only, 1k-sample-trained adapter — the degradation tracks adapter quality, not task mismatch.

**3. The cross-task advisory fires correctly but is overprotective for this case.**

Gradience's task-boundary detection is accurate: SST-2 and IMDB are different datasets. But the advisory treats this pair as equivalent in risk to genuine cross-task merges (sentiment x AG News, sentiment x NLI), which Phase 2 showed produce much larger degradation (avg Δ = -0.047).

## What This Inventory Suggests

SST-2 and IMDB are practically interchangeable for merge purposes. The current strict task-identity boundary misclassifies these as cross-task when they behave as same-task. A task-family taxonomy — even a simple one covering the most common NLP task families — would reduce unnecessary caution for users with sentiment-family adapter pools.
