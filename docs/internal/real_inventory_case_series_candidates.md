# Real Inventory Case Series — Candidate Selection

## Purpose

Build 2–4 additional real-inventory case studies that collectively show where the current Gradience workflow stays strong, where it is merely helpful, and where it begins to strain.

This is evidence-building, not benchmarking. No threshold or logic changes during selection or execution.

## Target count

- Minimum: 2
- Preferred: 3
- Maximum: 4

## Freeze note

The following are frozen for the duration of this series:

- default workflow pipeline
- strict-QA semantics
- pair-risk thresholds
- neighborhood logic
- core-space formulas

---

## Target Inventory Categories

### Category 1 — Messier mixed-quality inventory

An inventory where adapter QA statuses are genuinely varied (eligible, uncertain, unknown, possibly weak). Pair reports will not be uniformly clean.

Qualifies if:
- at least 2 distinct eligibility statuses present
- at least one adapter that source QA should flag or downgrade
- pair-risk landscape is not uniformly safe or uniformly dangerous

Purpose: show whether source QA still does major early narrowing when the pool is messy.

### Category 2 — Larger pool with neighborhood opportunity

An inventory with 5–7 adapters — enough for neighborhoods to become structurally informative rather than trivially echoing the pair matrix.

Qualifies if:
- 5+ adapters
- likely to produce more than one local group or at least one meaningful boundary warning
- candidate density is high enough that flat-pool exploration would be impractical

Purpose: show whether neighborhoods remain useful at moderate inventory size.

### Category 3 — Cross-task or cross-style but structurally compatible

Adapters that are not all same-task clones, but share a base model family and are plausibly merge-compatible.

Qualifies if:
- adapters differ in task, rank policy, or training regime
- same base model family
- enough structural difference to produce nontrivial pair structure

Purpose: test whether the workflow helps when the pool is less homogeneous.

### Category 4 — Core-space-relevant inventory

An inventory likely to produce at least one pair with ordinary pair_risk=low but plausible core-space disagreement.

Qualifies if:
- at least one low-risk pair exists
- structural similarity is high enough that core-space inspection is worth running
- the pair would otherwise stay in the active candidate set

Purpose: track where core-space matters in realistic selection.

---

## Raw Candidate List

### Candidate A — RoBERTa mixed-evidence triplet

- **Inventory ID:** `roberta_mixed_evidence_4`
- **Base model:** roberta-base
- **Adapter count:** 3 (possibly 4 with extension)
- **Category:** 1 (messier mixed-quality) + 3 (cross-style)
- **Adapters:**
  - `roberta_uniform_elig` — eligible, sst2_dev 0.88 (beats base 0.85), rank_90=3.0
  - `roberta_probe_uncertain` — uncertain, sst2_dev 0.85 (ties base), rank_90=6.5
  - `roberta_per_layer_unknown` — unknown_no_behavioral_eval, rank_90=4.0
- **Adapter source:** `results/real_inventory_runs/20260317/cycle03_roberta_mixed_evidence_triplet/adapters/`
- **QA source:** `results/real_inventory_runs/20260317/cycle03_roberta_mixed_evidence_triplet/qa/`
- **Why interesting:**
  - Different base model from first case study (roberta vs distilbert)
  - All three QA statuses are different — genuine messy pool
  - Prior pair reports show: 1 low-risk pair, 1 medium (norm_imbalance 11x), 1 high (norm_imbalance 6x)
  - Neighborhoods found 2 clusters with a boundary warning
  - Source screening should meaningfully narrow: uncertain adapter ties base, unknown has no eval
- **Expected result:** QA excludes or flags 1–2 adapters early; the one low-risk pair (uniform_elig × probe) becomes the focus; neighborhoods should separate per_layer into its own cluster.
- **Not redundant because:** first case study was all-distilbert, all-QNLI, 4 adapters with only 1 weak. This is RoBERTa, SST2, 3 adapters, 2 problematic QA statuses.

### Candidate B — Large distilbert pool (6 adapters)

- **Inventory ID:** `distilbert_large_pool_6`
- **Base model:** distilbert-base-uncased
- **Adapter count:** 6
- **Category:** 2 (larger pool / neighborhood opportunity)
- **Adapters:**
  - `final_per_layer_ckpt50` — unknown, rank_90=10.0
  - `final_probe_r16_ckpt50` — unknown, rank_90=9.5
  - `final_uniform_median_r16_ckpt50` — unknown, rank_90=10.0
  - `final_uniform_p90_r16_ckpt50` — unknown, rank_90=10.0
  - `qnli_probe_elig` — eligible, qnli_dev 0.87, rank_90=18.0 (high utilization)
  - `qnli_uniform_elig` — eligible, qnli_dev 0.87, rank_90=18.0 (high utilization)
- **Adapter sources:**
  - cycle02_final_test_quartet adapters: `results/real_inventory_runs/20260317/cycle02_final_test_quartet/adapters/`
  - cycle03_qnli_all_eligible adapters: `results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/`
- **QA sources:**
  - `results/real_inventory_runs/20260317/cycle02_final_test_quartet/qa/`
  - `results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/qa/`
- **Why interesting:**
  - 6 adapters = 15 pairs. Flat-pool exploration is impractical without narrowing.
  - Two clear quality tiers: 4 unknown (no eval) vs 2 eligible (with behavioral validation)
  - Prior data shows cycle02_final has high norm-imbalance issues (per_layer vs others = 6–7x)
  - QNLI adapters have much higher effective rank (18 vs 10) — structural mismatch with cycle02 final adapters
  - Neighborhoods should find at least 2 distinct groups (QNLI vs final), possibly 3
  - Core-space could be relevant: some low-risk pairs exist within the final_test set (probe vs uniform_median, probe vs uniform_p90)
- **Expected result:** strict-QA blocks all 4 unknown adapters (drastic narrowing); non-strict still has interesting pair structure; neighborhoods should cluster QNLI adapters separately from final-test adapters.
- **Not redundant because:** first case study was 4 adapters (manageable by hand). This inventory is large enough that manual pair-by-pair triage is genuinely tedious.

### Candidate C — Cross-task DialoGPT triplet

- **Inventory ID:** `dialogpt_cross_task_3`
- **Base model:** microsoft/DialoGPT-small
- **Adapter count:** 3
- **Category:** 3 (cross-task / cross-style) + 4 (core-space relevant)
- **Adapters:**
  - `btgenbot` — unknown, rank_90=7.5
  - `magicoder` — unknown, rank_90=7.5
  - `metamath` — unknown, rank_90=7.5
- **Adapter source:** `results/real_inventory_runs/20260317/study17_cache_triplet/adapters/`
- **QA source:** `results/real_inventory_runs/20260317/study17_cache_triplet/qa/`
- **Why interesting:**
  - Genuinely different tasks: chat, code generation, math
  - All adapters have identical spectral rank — structural similarity is high
  - All pairs show medium risk / high_redundancy (12 layers each)
  - Core-space on metamath × magicoder returned compatible (shared_basis_score=0.993)
  - The "too similar" pattern is different from prior case studies
  - No QA differentiation (all unknown) — source screening does nothing here
- **Expected result:** QA does not help (all unknown). All pairs medium risk. Core-space says compatible. Neighborhoods produce singletons. The workflow adds little here — this is a "merely helpful" or "starts to strain" case.
- **Not redundant because:** only candidate where the workflow may honestly not add much. Documents the boundary. Different base model family entirely.

### Candidate D — Distilbert core-space-focused quartet

- **Inventory ID:** `distilbert_core_space_4`
- **Base model:** distilbert-base-uncased
- **Adapter count:** 4
- **Category:** 4 (core-space relevant)
- **Adapters:**
  - `final_probe_r16_ckpt50` — unknown, rank_90=9.5
  - `final_uniform_median_r16_ckpt50` — unknown, rank_90=10.0
  - `final_uniform_p90_r16_ckpt50` — unknown, rank_90=10.0
  - `qnli_per_layer_elig` — eligible, rank_90 varies
- **Why interesting:**
  - Prior pair data: probe vs uniform_median = low risk, probe vs uniform_p90 = low risk, uniform_median vs uniform_p90 = medium (high_redundancy 24 layers)
  - Two low-risk pairs are core-space targets: they look safe ordinarily but structural inspection might disagree
  - Adding qnli_per_layer_elig (eligible, different task) creates cross-task contrast within the same inventory
- **Expected result:** source QA flags 3 unknown adapters but the eligible one survives. Core-space on the low-risk pairs tests whether "safe" really means "compatible at depth."
- **Not redundant because:** only candidate specifically designed for core-space disagreement testing. First case study used core-space on an ambiguous pair; this uses it on a "safe" pair.

---

## Wave 1 Selection

**Selected inventories for immediate execution:**

### selected_wave_1

1. **Candidate A — `roberta_mixed_evidence_4`** (Category 1: messier mixed-quality)
   - Maximum contrast with first case study: different base model, different task, genuine QA variation
   - Source screening should do real work here
   - Tests the workflow on a genuinely messy pool

2. **Candidate B — `distilbert_large_pool_6`** (Category 2: larger pool)
   - Tests neighborhood utility at scale (15 pairs)
   - Structural mismatch between adapter groups should produce informative clustering
   - Core-space opportunity on low-risk pairs within the final-test subset

**Why this pair:**
- Maximizes contrast: 3-adapter RoBERTa (messy QA, small pool) vs 6-adapter distilbert (large pool, neighborhood stress test)
- Neither duplicates the first case study's story (4 distilbert QNLI adapters with one weak)
- Together they cover categories 1, 2, and partially 3

**Deferred to wave 2 (if needed):**
- Candidate C (`dialogpt_cross_task_3`) — boundary/strain case
- Candidate D (`distilbert_core_space_4`) — core-space focused
