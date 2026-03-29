# Phase 1 Pilot — Cross-Pilot Comparison

## Summary table

| Dimension | Pilot 1 (same-task control) | Pilot 2 (mixed-task) | Pilot 3 (large mixed-task) |
|-----------|---------------------------|---------------------|---------------------------|
| Backbone | distilbert-base-uncased | roberta-base | distilbert-base-uncased |
| Adapters (working) | 3 of 4 | 5 of 5 | 8 of 9 |
| Task types | 2 (IMDB, emotion) | 4 (hate, ag_news, irony, MNLI) | 4 (sentiment, emotion, ag_news, hate) |
| Candidate pairs | 3 | 10 | 28 |
| Retained pairs | 0 | 1 | 2 |
| Reduction | 100% | 90% | 93% |
| Eligible adapters | 2 | 4 | 6 |
| Uncertain | 0 | 1 (irony) | 0 |
| Flagged weak | 1 (jmeneu) | 0 | 2 (jmeneu, Aureliano) |
| Same-task pairs found | 0 | 1 | 3 |
| Same-task pairs retained | 0 | 1 | 2 |
| Cross-task regions | 1 | 6 | 6 |
| Dominant structural issue | norm_imbalance | norm_imbalance | norm_imbalance |

## Progression narrative

The three pilots form a natural progression — from a degenerate case to a working product demonstration.

**Pilot 1** was designed as a same-task control but failed that purpose. With only 3 working adapters across 2 tasks and no surviving same-task pairs after the evidence gate, it produced a null result (0 retained). Its contribution is methodological: it validated the evidence bootstrap pipeline and demonstrated that the evidence gate materially changes preflight output.

**Pilot 2** is the first inventory where Gradience produces actionable output. The same-task AG News pair is correctly retained as the sole evaluate-first candidate, with a meaningful structural diagnosis (partial redundancy in 8 of 24 layers). The 90% reduction (10 → 1) is the kind of result a practitioner would find useful. The irony adapter's `uncertain` status and the hate × irony pair's low structural risk add interpretive nuance.

**Pilot 3** tests scale. At 28 pairs, the summary and action-plan layers add genuine navigability value. The 93% reduction (28 → 2) is sharp, with two structurally distinct candidates: one low-risk linear merge (AG News pair) and one medium-risk norm-equalized merge (SST-2 pair). The evidence gate correctly removes two weak adapters, and the exclusion of one same-task pair (hate speech, blocked by a weak source) reveals the "near-miss" problem.

## Consistent findings across pilots

### 1. The evidence gate is the most impactful product feature

Without behavioral evidence: everything excluded (Pilot 1 v1). With evidence: meaningful eligibility classification across all three inventories. The three-way classification (eligible / uncertain / flagged_weak) is calibrated correctly — it handles genuine failures (Aureliano, delta -0.150), misleading evals (jmeneu, base artifact), marginal passes (hate adapters, delta ~0.05), ambiguous ties (irony, delta 0.000), and strong performers (ag_news adapters, delta >0.65).

### 2. Norm imbalance dominates structural analysis

Norm_imbalance is the dominant issue in all three pilots: 100% of Pilot 1 pairs, 80% of Pilot 2, 75% of Pilot 3. This is driven by the configuration heterogeneity in public LoRA adapters — TransferGraph (r=1/alpha=1) vs community adapters (r=4 to r=16, alpha=8 to 32). The signal is real, but its diagnostic value saturates when it applies to the majority of pairs. In a real product deployment, practitioners would benefit from norm-imbalance severity ranking rather than a uniform label.

### 3. Task-boundary detection is metadata-correct but task-family-blind

Cross-task advisories fire correctly based on eval_dataset metadata mismatch. This works well for genuinely different tasks (ag_news vs MNLI, hate vs emotion). But it misses task-family equivalence: IMDB and SST-2 are both binary sentiment but treated as cross-task. This is a product-level design decision — metadata matching is simple, reliable, and conservative. Task-family inference would add nuance but requires a task taxonomy.

### 4. Same-task pairs, when found, produce the clearest recommendations

The AG News pair in Pilot 2 (partial redundancy, norm_equalized) and Pilot 3 (low risk, linear) are the most useful recommendations across the entire pilot. Same-task pairs bypass the task-boundary concern entirely, letting structural analysis speak directly. This validates the product's core value: within same-task inventories, spectral analysis provides actionable merge guidance.

### 5. The "marginal adapter" problem

Adapters that barely beat base (hate speech adapters: delta ~0.05 on binary tasks) pass the evidence gate as `eligible`. A practitioner might not want to invest in merging a 0.52-accuracy hate classifier. The current gate is binary: positive delta → eligible. A graduated confidence score or a minimum-delta threshold would help.

## Product feature priorities (informed by pilot findings)

1. **Evidence bootstrap integration** — The most impactful improvement would be built-in support for running lightweight evals. Currently, evidence generation is a separate step (`evidence_bootstrap.py`). A built-in `gradience eval --adapter-dir ... --dataset ...` command would lower the barrier to generating behavioral evidence, which the pilot demonstrated is essential for useful preflight output.

2. **Norm-imbalance severity ranking** — When 75% of pairs show the same issue, ranking by severity (e.g., magnitude ratio) would help practitioners triage. The raw data is already computed (e.g., "4.7× mean magnitude ratio across 22 layers"); surfacing it as a sortable metric in the action plan would add value.

3. **Near-miss reporting** — When a same-task pair is blocked by one weak source, the action plan should mention it: "The hate-speech pair was structurally clean but blocked because Aureliano underperforms base. Consider sourcing a better hate-speech adapter." This turns an exclusion into an acquisition signal.

4. **Task-family equivalence** — A lightweight task taxonomy (sentiment = {imdb, sst2, yelp, ...}) would let Gradience recognize that IMDB × SST-2 is same-task-family even though the datasets differ. This could be metadata-driven (a registry of known task families) rather than learned.

5. **Marginal-adapter advisory** — When an adapter's delta is very small relative to the task's difficulty (e.g., +0.048 on binary), add a note: "This adapter shows marginal improvement over base. Consider whether the improvement is sufficient to justify inclusion in merge candidates."

## What the pilot cannot tell us

These pilots are limited to minimal-rank TransferGraph adapters (r=1) and small community adapters (r=4 to r=16). The spectral analysis features that would be most interesting — rank-dependent structure, multi-rank interactions, energy-rank medians, utilization patterns — are largely invisible at r=1. A richer-adapter control inventory (r≥8, broader target modules, well-trained adapters) is needed to test whether the spectral layer adds value beyond what the evidence gate and task-boundary detection already provide.

The pilot also uses only classification tasks with accuracy as the metric. Generation tasks, regression tasks, and tasks with more nuanced metrics (F1, BLEU, perplexity) would test different aspects of the evidence pipeline.

Finally, no actual merges were evaluated. The pilot tests Gradience's *recommendations*, not whether following those recommendations produces better merges. A validation step — actually merging the retained pairs and evaluating the result — would close the loop.
