# Execution Plan — CPU-Only Field Research Protocol

**Companion to:** `cpu_field_research_protocol.md`
**Purpose:** Turn the protocol into concrete steps — which adapters, which datasets, which scripts, what to run in what order, and what decisions to make at each gate.

---

## Pre-work: infrastructure readiness

Before any micro-campaign, confirm the following are working:

| Item | How to check | Current status |
|------|-------------|----------------|
| Evidence bootstrap script | `python field_trials/evidence_bootstrap.py --help` | Working (used in Phases 1–2b) |
| Phase 2 eval script | `python field_trials/run_phase2_eval.py --help` | Working (used in Phase 2) |
| Gradience preflight CLI | `gradience summarize-inventory --help` | Working |
| HuggingFace Hub access | `python -c "from datasets import load_dataset; load_dataset('sst2', split='validation[:5]')"` | Requires network |
| PEFT adapter loading | `python -c "from peft import PeftConfig"` | Working |

**New datasets needed for Campaign A.** The current `DATASET_REGISTRY` in `evidence_bootstrap.py` covers: imdb, tweet_eval/emotion, tweet_eval/hate, tweet_eval/irony, ag_news, mnli. Campaign A requires adding:

- `sst2` — `glue/sst2`, split `validation`, text_col `sentence`, label_col `label`, 2 labels
- `yelp_polarity` — `yelp_polarity`, split `test`, text_col `text`, label_col `label`, 2 labels
- `amazon_polarity` — `amazon_polarity`, split `test`, text_col `content`, label_col `label`, 2 labels

These should be added to `DATASET_REGISTRY` in both `evidence_bootstrap.py` and `run_phase2_eval.py` before Campaign A begins. All three are standard HuggingFace datasets, CPU-friendly for 500-sample slices.

---

## Campaign A — Task-Family Equivalence

### The question

Is exact task identity (eval_dataset string match) too strict for practically similar tasks, or is it the right conservative boundary?

### Why this matters most

Task-family blindness is called out in `docs/product-validation.md` §6 as a known limitation. If IMDB × SST-2 merges perform like retained same-task pairs, the product should recognize that. If they perform like cross-task controls, the current conservatism is correct. This is the single most interesting unresolved product question.

### Inventory design

**Inventory A-01: Sentiment family (DistilBERT)**

| Adapter | Task | Source | Expected |
|---------|------|--------|----------|
| myselfmankar SST-2 (r=16) | sst2 | HuggingFace Hub, already verified | eligible |
| NightPrince SST-2 (r=8) | sst2 | HuggingFace Hub, already verified | eligible |
| muneeb-ai IMDB (r=4) | imdb | HuggingFace Hub, already verified | eligible |
| New: find 1 IMDB DistilBERT adapter | imdb | HuggingFace Hub search | eligible or uncertain |
| New: find 1 Yelp or Amazon DistilBERT adapter | yelp/amazon | HuggingFace Hub search | eligible or uncertain |

*Goal: 5 adapters, 2 tasks minimum (SST-2 + IMDB), ideally 3 (+ Yelp/Amazon). 10 pairs. Include exact same-task (SST-2 × SST-2), same-family cross-dataset (SST-2 × IMDB), and truly cross-task control (sentiment × non-sentiment if a non-sentiment adapter is available, otherwise use the cross-task control data from Phase 2).*

**Adapter sourcing note.** Search HuggingFace Hub for: `distilbert-base-uncased lora imdb`, `distilbert-base-uncased lora yelp`, `distilbert-base-uncased lora amazon`. If DistilBERT adapters are scarce for Yelp/Amazon, try BERT-base as a second backbone — this also tests whether the family-equivalence finding is backbone-dependent.

**Inventory A-02: Sentiment family (RoBERTa or BERT)**

Same design as A-01 but on a different backbone. This tests whether the family-equivalence result (whichever direction) transfers across backbones.

**Fallback:** If public adapters are too scarce for a second backbone, run A-01 only and note the single-backbone limitation in the memo.

### Required pair classes per inventory

| Class | Example | Purpose |
|-------|---------|---------|
| Exact same-task retained | SST-2 × SST-2 | Baseline: how good are retained pairs? |
| Same-family cross-dataset | SST-2 × IMDB | **The test pair.** Does this behave like retained or like cross-task? |
| Cross-task control | Sentiment × AG News (from Phase 2 data) | Anchor: how bad are genuine cross-task merges? |

### Workflow

1. **Source adapters.** Search Hub, verify loadability with `PeftConfig.from_pretrained()`, record configs.
2. **Build manifest.** Use the same JSON schema as existing inventories.
3. **Add datasets** to `DATASET_REGISTRY` (sst2, yelp_polarity, amazon_polarity).
4. **Run evidence bootstrap.** 500 samples per adapter. Record scores.
5. **Run Gradience preflight.** Lock stance before evaluation.
6. **Evaluate 3–5 merges:**
   - 1 exact same-task retained pair
   - 1–2 same-family cross-dataset pairs (the key test)
   - 1 cross-task control (reuse Phase 2 data if backbones match, otherwise run one new control)
7. **Write field note** per inventory.
8. **Write memo:** `task_family_equivalence_memo.md`.

### Decision gate after Campaign A

| Finding | Implication |
|---------|------------|
| Same-family pairs behave like retained (avg Δ within 0.02 of retained) | Consider adding a task-family taxonomy to the product. Design it as a metadata registry, not a learned classifier. |
| Same-family pairs are intermediate (worse than retained but better than cross-task) | Document the gradient. The current conservatism is defensible but slightly overprotective. A "same-family advisory" (weaker than cross-task advisory) may be worth considering. |
| Same-family pairs behave like cross-task controls | The current boundary is correct. Exact task identity is the right practical line. Close the question. |

### Estimated effort

- Adapter sourcing: 1–2 hours (Hub search, loadability verification)
- Evidence bootstrap: ~30 min per inventory (CPU)
- Preflight: ~5 min per inventory
- Merge evaluation: ~1–2 hours per inventory (CPU inference)
- Field note + memo: 1 hour
- **Total: ~1 day of hands-on work per inventory**

---

## Campaign B — Marginal-Adapter Behavior

### The question

Do barely-weak adapters (delta -0.002 to -0.010 vs base) behave more like near-miss candidates or like genuinely excluded sources?

### Why this is second

Phase 2b already showed the pattern qualitatively (barely-weak near-miss pairs behave like retained), but the sample size is thin and the split between "barely weak" and "deeply weak" was observational, not designed. Campaign B makes it deliberate.

### Inventory design

**Inventory B-01: Marginal irony/hate cluster (DistilBERT)**

Reuse the inventory_04 (irony cluster) adapter pool but add or replace adapters to ensure at least:
- 2 adapters with delta in the -0.002 to -0.010 range ("barely weak")
- 1 adapter with delta below -0.050 ("deeply weak")
- 2+ adapters that are clearly eligible

If the existing adapters don't provide enough marginal cases, search Hub for additional adapters on the same task/backbone that might fall near the threshold. Alternatively, use a different eval budget (100 samples instead of 500) on the same adapters — this naturally shifts some adapters closer to the threshold due to higher sampling variance.

**Inventory B-02: Marginal hate/emotion cluster (BERT)**

Similar design on bert-base-uncased, reusing the inventory_05 adapter pool with possible additions.

### Required pair classes

| Class | Definition | Purpose |
|-------|-----------|---------|
| Retained | Both sources eligible | Baseline |
| Near-miss (barely weak) | One source delta -0.002 to -0.010 | **Test band 1** |
| Near-miss (deeply weak) | One source delta below -0.050 | **Test band 2** |
| Excluded control | Cross-task | Anchor |

### Key comparison

| Metric | Barely weak NM | Deeply weak NM | Retained | Control |
|--------|---------------|----------------|----------|---------|
| Avg Δ vs best source | ? | ? | -0.024 (Phase 2) | -0.047 (Phase 2) |
| Variance | ? | ? | baseline | baseline |

If barely-weak near-miss avg Δ falls within the retained range (above -0.030), and deeply-weak near-miss falls measurably worse, the current near-miss treatment is correct but would benefit from a severity ranking within the near-miss section (barely-weak pairs first).

### Decision gate after Campaign B

| Finding | Implication |
|---------|------------|
| Barely weak ≈ retained, deeply weak ≈ intermediate | Add a "confidence" or "proximity to threshold" indicator in the near-miss section. Rank barely-weak pairs above deeply-weak. |
| Both bands ≈ retained | The current near-miss treatment is sufficient. No product change needed. |
| Both bands ≈ control | Near-miss is less useful than Phase 2b suggested. Tighten the exclusion. (This would be surprising.) |

### Estimated effort

- Adapter sourcing: 1 hour (mostly reusing existing adapters)
- Evidence bootstrap: ~30 min per inventory
- Merge evaluation: ~1 hour per inventory
- **Total: ~1 day across both inventories**

---

## Gate: Continue or stop?

After Campaigns A and B, assess:

- If task-family equivalence clearly does not help AND marginal-adapter behavior is already resolved by Phase 2b data, **stop**. Write the synthesis memos and close.
- If either campaign produced a product-actionable finding, continue to C and/or D.
- If you're uncertain, Campaign D (robustness) is more informative than C (stress test) for most practical purposes.

---

## Campaign C — Large-Inventory Stress Test

### The question

Do ergonomics hold at 10–14 adapters / 40–90 pairs?

### Inventory design

**Inventory C-01: 12-adapter sentiment/classification pool (DistilBERT)**

Combine adapters from multiple existing inventories plus new sourcing to reach 12 adapters across 3–4 tasks. This generates 66 pairs — well above the 28-pair validated ceiling.

The goal is not merge quality (only evaluate 2–3 merges). The goal is: can you understand the preflight output? Is the HTML report scannable? Does the action plan make sense? Are neighborhoods useful?

**Inventory C-02 (optional): 10-adapter mixed pool (RoBERTa)**

Only if C-01 reveals interesting strain points worth testing on a second backbone.

### Evaluation focus

Primarily qualitative. Rate each output artifact (region summary, candidate map, action plan, HTML report, review packet) as high / medium / low usefulness. Note where presentation starts to strain.

Evaluate only 1–2 retained pairs + 1 control to keep the outputs grounded.

### Estimated effort

- Adapter sourcing: 1–2 hours
- Pipeline run: ~30 min
- Qualitative assessment: 1 hour
- **Total: half a day**

---

## Campaign D — Public-Ecosystem Robustness

### The question

How gracefully does Gradience handle messy real-world adapters?

### Inventory design

**Inventory D-01: Deliberately messy pool**

Search Hub specifically for adapters with unusual properties:
- Non-standard target modules (e.g., all 6 transformer sublayers, or only `k`)
- Transfer-chain bases (already identified in the landscape assessment — TransferGraph adapters use intermediate fine-tuned models as base)
- Very high or very low alpha (alpha=1 vs alpha=64)
- Sparse or missing metadata (no eval results, no task description)
- Adapters that partially fail to load

The existing TransferGraph adapters already exercise the transfer-chain pattern. Add 2–3 genuinely odd Hub adapters to push the boundaries.

**Inventory D-02: Partially broken pool**

Include at least one adapter that fails to load entirely, one that loads but produces garbage predictions, and one with a mismatched label count. The goal is to test whether the pipeline fails gracefully (clear error messages, honest reporting) rather than silently producing bad output.

### Evaluation focus

Merge evaluation is secondary. The primary measures are: load success rate, bootstrap success rate, manual intervention count, error message quality, whether the pipeline tells you what's wrong.

### Estimated effort

- Adapter sourcing: 2 hours (finding weird adapters is slower)
- Pipeline runs + diagnostics: 1–2 hours
- **Total: half a day to one day**

---

## Synthesis

After all completed campaigns, write two documents in `field_trials/synthesis/`:

**`cross_campaign_summary.md`** — What was confirmed, what remains ambiguous, which product questions are now settled, which are still open. Reference specific inventory results.

**`product_implications.md`** — Rank three categories:
1. Immediate product changes worth making (with implementation notes)
2. Things confirmed good enough already (with evidence references)
3. Things that should wait for more evidence (with what that evidence would need to look like)

Update `docs/product-validation.md` §6 (limitations) to reflect any newly resolved or newly discovered issues.

Update `sidecar/notes/n69_settled_open_next.md` to move any resolved questions from Open to Settled.

---

## Overall sequencing and timeline

| Phase | Campaign | Inventories | Estimated effort | Cumulative |
|-------|----------|-------------|-----------------|------------|
| 1 | A (task-family equivalence) | A-01, A-02 | 2 days | 2 days |
| — | **Gate decision** | — | — | — |
| 2 | B (marginal-adapter behavior) | B-01, B-02 | 1 day | 3 days |
| — | **Gate decision** | — | — | — |
| 3 | D (public robustness) | D-01, D-02 | 1 day | 4 days |
| 4 | C (large-inventory stress) | C-01 | 0.5 day | 4.5 days |
| — | **Synthesis** | — | 0.5 day | **5 days total** |

This is ~5 working days of hands-on effort spread across as many calendar days as needed. The gate decisions after A and B may shorten this — if both questions resolve cleanly, Campaigns C and D become optional.

---

## Checklist before starting Campaign A

- [x] Confirm HuggingFace Hub network access (2026-03-29)
- [x] Add sst2, yelp_polarity, amazon_polarity to `DATASET_REGISTRY` in both scripts (2026-03-29; also fixed shuffled sampling)
- [x] Search Hub for DistilBERT LoRA adapters (2026-03-29; no Yelp/Amazon LoRA adapters available)
- [x] Confirm SST-2 adapters available (myselfmankar confirmed; NightPrince ID was stale, rambodazimi used instead)
- [x] Create inventory manifest: `task_family_equivalence/inventory_a01/manifest.json` (2026-03-29)
- [x] Run evidence bootstrap on all adapters (2026-03-29; muneeb-ai dropped due to PEFT incompatibility)
- [x] Confirm Gradience preflight runs cleanly (2026-03-29)
- [x] Run Campaign A merge evaluations — 7 pairs (2026-03-29)
- [x] Campaign A memo written (2026-03-29; result: task-family equivalence confirmed)
- [x] Campaign B memo written (2026-03-29; result: barely-weak ≈ retained, deeply-weak ≈ controls)
- [x] Cross-campaign synthesis written (2026-03-29)
- [x] Product implications written (2026-03-29)
