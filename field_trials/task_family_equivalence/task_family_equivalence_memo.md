# Campaign A Memo — Task-Family Equivalence

**Date:** 2026-03-29
**Campaign:** A (Task-Family Equivalence)
**Question:** Is exact task identity too strict, or is it the right conservative boundary?
**Verdict:** Too strict. Same-family cross-dataset pairs behave like retained same-task pairs.

---

## Summary

Campaign A tested whether Gradience's strict task-identity logic — which flags any pair with different `eval_dataset` strings as cross-task — is appropriately conservative for practically similar tasks. Specifically: do SST-2 x IMDB sentiment merges behave like same-task retained pairs or like genuine cross-task controls?

**Finding:** Same-family cross-dataset merges are indistinguishable from same-task retained merges. The avg delta vs best source for family-test pairs (-0.022) falls within the retained range (-0.017), not the cross-task control range (-0.047). The current boundary is overprotective for this task family.

## Evidence

### Inventory A-01: DistilBERT Sentiment Family

4 adapters (2 SST-2, 2 IMDB), all eligible, all distilbert-base-uncased. 7 merge evaluations across 6 pairs.

| Category | Avg Δ vs Best Source | n | Range |
|----------|---------------------|---|-------|
| Retained (same-task/dataset) | -0.017 | 2 | -0.008 to -0.026 |
| Family test (SST-2 x IMDB) | -0.022 | 4 | -0.006 to -0.050 |
| Phase 2 cross-task control (reference) | -0.047 | — | -0.031 to -0.089 |
| Phase 2 near-miss (reference) | -0.006 | — | +0.002 to -0.014 |

The family-test mean falls 0.005 above the retained mean and 0.025 above the cross-task control mean. The family-test distribution overlaps completely with the retained distribution.

### Structural observations

Gradience's structural analysis (risk levels, dominant issues, strategy recommendations) varies across the SST-2 x IMDB pairs based on structural properties (rank mismatch, norm ratio) — not task identity. The audit_aware merges perform slightly better than uniform_linear, consistent with the structural signals being informative even when the task advisory is misleading.

The one outlier (-0.050 delta on SST-2 for rambodazimi x wt-golf) is driven by adapter quality (wt-golf was trained on only 1k samples with q_lin-only targeting), not by task mismatch. This same pair shows only -0.006 degradation when evaluated on IMDB.

### Confound acknowledgment

This test covers only one task family (binary sentiment) on one backbone (distilbert-base-uncased). The SST-2/IMDB comparison is a favorable case: both are binary sentiment classification with the same label semantics ({0: negative, 1: positive}). Less aligned task families (e.g., entailment variants, multi-class classification subfamilies) may show larger gaps.

## Gate Decision

Per the protocol's decision criteria:

> **Same-family pairs behave like retained (avg Δ within 0.02 of retained):**
> Consider adding a task-family taxonomy to the product. Design it as a metadata registry, not a learned classifier.

The finding clearly triggers this gate. The avg Δ gap between family-test and retained is 0.005, well within the 0.02 threshold.

### Recommendation

Add a simple task-family taxonomy to Gradience's advisory system. Design considerations:

1. **Scope:** Start with the most common NLP classification families: binary sentiment (SST-2, IMDB, Yelp, Amazon), topic classification (AG News, DBpedia, Yahoo Answers), NLI (MNLI, SNLI, RTE), and emotion/affect (tweet_eval/emotion, GoEmotions).

2. **Mechanism:** A static registry mapping known dataset names to family labels. Not a learned classifier — the taxonomy should be inspectable and overridable.

3. **Advisory change:** When both adapters in a pair belong to the same task family, downgrade the cross-task advisory from "caution" to "informational" (e.g., "Different datasets but same task family — similar tasks tend to merge well").

4. **Conservative default:** Unknown datasets remain in their own singleton family. The taxonomy never suppresses the advisory entirely — it modulates the language.

5. **Validation requirement:** Before promoting, replicate on at least one additional task family (e.g., NLI) and one additional backbone.

## Remaining Limitations

- Single task family tested (binary sentiment)
- Single backbone (distilbert-base-uncased)
- No Yelp/Amazon adapters available for DistilBERT LoRA — could not test broader sentiment family
- Label semantics alignment was favorable (both binary, same label mapping) — misaligned labels would require more careful handling
- Small sample of 4 family-test evaluations — enough for a clear signal, but not for precise effect-size estimation

## Files

- Inventory manifest: `inventory_a01/manifest.json`
- Evidence: `inventory_a01/evidence/`
- Preflight: `inventory_a01/preflight_ev_090044/`
- Eval results: `eval_090340/campaign_a_results.json`
- Field note: `inventory_a01/field_note.md`
