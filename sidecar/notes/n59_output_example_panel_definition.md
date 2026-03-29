# n59 — Output Example Panel Definition

**Type:** panel definition
**Date:** 2026-03-28
**Depends on:** Phase 2b field trial results, n51 research synthesis
**Status:** Defines the case panel for the Output Example Semantics program.

---

## Purpose

This note defines the panel of merge cases whose per-example predictions will be analyzed for behavioral patterns. The goal is a small, information-rich panel spanning safe, fragile, control, and near-miss categories — sufficient to test whether example-level failure structure differs across merge quality classes.

---

## Case selection rationale

All cases are drawn from the Phase 2b field trial, which has merged adapters on disk and known aggregate accuracy scores. Source adapters are cached in the inventory directories. No new merges are required.

The panel is designed to maximize behavioral contrast with minimum cases:

- **Group A (safe retained):** Two same-task pairs with mild degradation, both on different backbones, to establish the preservation baseline.
- **Group B (fragile):** Two cases with noticeable degradation — one from a weak-partner retained pair, one from a deeply-weak near-miss — to characterize structured failure.
- **Group C (control):** One cross-task pair with substantial degradation, providing the contrast endpoint.
- **Group D (near-miss):** Two near-miss pairs with minimal degradation, to test whether near-miss behavior is closer to safe retained or fragile.
- **Group E (anchor):** The emotion-TGbase×hatexplain near-miss is the strongest pathology case in the panel (Δ=-0.088, deeply weak source at 0.136). It serves as the anchor for "what does severe near-miss degradation look like at example level."

---

## Panel

| Case ID | Pair | Backbone | Task | Class | Δ vs best | Source A score | Source B score | Merged score |
|---------|------|----------|------|-------|-----------|----------------|----------------|-------------|
| SR-01 | irony_JB173 × neibla | DistilBERT | irony (binary) | safe_retained | -0.006 | 0.632 | 0.620 | 0.626 |
| SR-02 | bert_hate_TGbase × hatexplain | BERT | hate (binary) | safe_retained | +0.028 | 0.514 | 0.588 | 0.616 |
| FR-01 | bert_emotion_TGbase × fabriceyhc | BERT | emotion (4-class) | fragile | -0.088 | 0.752 | 0.204 | 0.664 |
| FR-02 | bert_emo_TGbase × hatexplain_NM | BERT | emotion (4-class) | fragile | -0.088 | 0.752 | 0.136 | 0.664 |
| CT-01 | bert_cross_agnews × aviator_hate | BERT | ag_news/hate (cross) | control | -0.096 | 0.922 | 0.574 | 0.826 |
| NM-01 | irony_JB173 × phailyoor_NM | DistilBERT | irony (binary) | near_miss | -0.012 | 0.632 | 0.618 | 0.620 |
| NM-02 | bert_hate_TGbase × aviator_NM | BERT | hate (binary) | near_miss | -0.002 | 0.514 | 0.498 | 0.572 |
| AN-01 | bert_emo_fab × hatexplain_NM | BERT | emotion (4-class) | anchor | -0.002 | 0.204 | 0.136 | 0.202 |

### Notes on case selection

**SR-01** and **SR-02** provide the safe baseline on two different backbones and tasks (binary irony on DistilBERT, binary hate on BERT). SR-02 is the positive case — the merge actually improved.

**FR-01** is formally a retained pair (both sources are `eligible`) but the partner is very weak (accuracy 0.204 on a 4-class task, barely above chance). It degraded -0.088. This makes it the best available "fragile but not cross-task" case.

**FR-02** is a near-miss pair where the weak source is deeply below base (0.136 vs base 0.286). It also degraded -0.088. Together FR-01 and FR-02 provide two fragile cases with the same magnitude of degradation but different evidence profiles.

**CT-01** is the sole cross-task control: ag_news adapter × hate adapter on BERT. Δ=-0.096. This is evaluated on ag_news (the stronger source's task), so we see how a cross-task merge degrades a strong adapter's predictions.

**NM-01** and **NM-02** are near-miss pairs with minimal degradation (-0.012 and -0.002), on two different backbones/tasks. They test whether near-miss example-level behavior is closer to safe retained or fragile.

**AN-01** is the unusual case: both sources are very weak (0.204 and 0.136 on 4-class emotion), and the merge barely changes anything (Δ=-0.002). This is the "floor" case — when both sources are near chance, what does the merge do at example level?

---

## Artifact availability

| Case ID | Source A adapter | Source B adapter | Merged adapter | Labels | Logits/confidence |
|---------|-----------------|-----------------|----------------|--------|-------------------|
| SR-01 | yes (inv04 cache) | yes (inv04 cache) | yes (phase2b output) | yes (tweet_eval/irony test) | will collect |
| SR-02 | yes (inv05 cache) | yes (inv05 cache) | yes (phase2b output) | yes (tweet_eval/hate test) | will collect |
| FR-01 | yes (inv05 cache) | yes (inv05 cache) | yes (phase2b output) | yes (tweet_eval/emotion test) | will collect |
| FR-02 | yes (inv05 cache) | yes (inv05 cache) | yes (phase2b output) | yes (tweet_eval/emotion test) | will collect |
| CT-01 | yes (inv05 cache) | yes (inv05 cache) | yes (phase2b output) | yes (ag_news test) | will collect |
| NM-01 | yes (inv04 cache) | yes (inv04 cache) | yes (phase2b output) | yes (tweet_eval/irony test) | will collect |
| NM-02 | yes (inv05 cache) | yes (inv05 cache) | yes (phase2b output) | yes (tweet_eval/hate test) | will collect |
| AN-01 | yes (inv05 cache) | yes (inv05 cache) | yes (phase2b output) | yes (tweet_eval/emotion test) | will collect |

All 8 cases are fully analyzable at the `full_example_panel` level (predictions + confidence + labels).

---

## Evaluation slice

All cases will use the first 500 examples from the relevant test split, matching the Phase 2b evaluation. The same slice is used for source A, source B, and merged model evaluation. This ensures:

- Reproducibility (fixed slice, deterministic evaluation)
- Per-example alignment (same examples across all three models)
- Labels available for all examples
- Logits/softmax probabilities collected for confidence analysis

---

## Success criteria

- At least 2 safe retained cases analyzable: **met** (SR-01, SR-02)
- At least 2 fragile/control cases analyzable: **met** (FR-01, FR-02, CT-01)
- Labels available for the selected slice: **met** (all 8 cases)
- At least one case supports confidence/logit analysis: **met** (all 8 cases will have logits)

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This panel definition | `sidecar/notes/n59_output_example_panel_definition.md` |
| Panel table (JSON) | `sidecar/results/example_semantics/panel_table.json` |
| Panel table (MD) | `sidecar/results/example_semantics/panel_table.md` |
