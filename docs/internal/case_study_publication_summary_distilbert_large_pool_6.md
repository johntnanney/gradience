# Publication Summary — distilbert_large_pool_6

Series: real_inventory_case_series, wave 1
Category: 2 (larger pool / neighborhood opportunity)
Date: 2026-03-22

## One-line summary

A 6-adapter inventory with 15 pairs collapses to one defensible neighborhood after source QA, neighborhoods, and core-space each contribute materially to narrowing.

## Setup

- Base model: distilbert-base-uncased
- 6 adapters from two distinct sources:
  - 4 from cycle02 final-test quartet (generic fine-tuning, no behavioral eval, r=2/16)
  - 2 from cycle03 QNLI all-eligible (QNLI entailment, behavioral eval passed, r=32)
- 15 pairwise merge reports

## What happened

1. **Source QA** split the pool into 2 eligible + 4 unknown. Strict-QA blocks 14 of 15 pair reports — the most dramatic narrowing in the series.

2. **Pair audit** found 5 high-risk (all per_layer norm imbalance), 4 medium-risk (redundancy), 6 low-risk. The per_layer adapter (r=2) is incompatible with everything. uniform_med and uniform_p90 are near-identical (compat=1.0).

3. **Core-space** on the most interesting cross-group low-risk pair (probe_r16 × qnli_probe) returned **incompatible** (shared_basis_score=0.867). This is a cross-task pair where structural incompatibility is consistent with verified adjudication findings (cross-task merges degrade, and ordinary pair-risk already separates them from safe same-task pairs).

4. **Neighborhoods** produced 5 groups: 4 singleton caution clusters (one per cycle02 adapter) + 1 likely-safe neighborhood ({qnli_probe, qnli_uniform}). 10 boundary warnings make the cross-group incompatibility visually obvious.

5. **Net result:** 15-pair flat pool → 1 defensible merge (qnli_probe × qnli_uniform). The cycle02 adapters are structurally isolated and behaviorally unvalidated.

## Where the workflow was strong

- **Source QA dominated.** 4 of 6 adapters had no behavioral evaluation; strict-QA alone reduces 15 pairs to 1. Even without strict-QA, the QA stratification immediately highlights the quality gap.
- **Neighborhoods scaled.** With 6 adapters and 15 pairs, the pair matrix is hard to parse manually. Neighborhoods compressed it into one actionable insight: the QNLI pair is the only safe neighborhood.
- **Core-space flagged structural incompatibility on a cross-task pair.** The probe_r16 × qnli_probe pair looked safe at the layer level but was incompatible at the basis level. Verified adjudication later showed this cross-task structural divergence is consistent with degraded merge behavior, but ordinary pair-risk already separated cross-task from same-task pairs in the tested regime.

## Where the workflow was merely helpful

- The pair-risk distribution (5 high, 4 medium, 6 low) is informative but doesn't by itself tell the practitioner which low-risk pairs are truly safe vs superficially safe. Neighborhoods and core-space were needed to resolve that.

## Where the workflow starts to strain

- The neighborhood output fragmented heavily: 4 singleton caution clusters is not structurally useful — it means "every cycle02 adapter is alone." With fewer adapters per group, neighborhood characterization becomes less informative.
- All cycle02 adapters lack behavioral eval, so the workflow cannot distinguish between them except spectrally. A practitioner who cares about those adapters still needs to run downstream eval.

## Inventory-level lesson

**What the inventory-level view changed:** Without the inventory view, a practitioner sees 6 low-risk pairs and might try multiple cross-group merges. The inventory view shows that the only safe neighborhood is {qnli_probe, qnli_uniform}, and core-space confirms that cross-group "safety" is illusory at depth. The action space collapsed from "explore 6 safe-looking pairs" to "merge the QNLI pair or invest in better adapters."

**Was the inventory mistake resolved before pairwise detail became central?** Yes, strongly. Source QA identified that 4/6 adapters have no behavioral evidence. This is the primary insight. Pairwise analysis and neighborhoods confirm and refine it, but the inventory-level QA stratification did the heavy lifting.

**Did the neighborhood result materially reduce the candidate space?** Yes. Neighborhoods compressed 15 pairs into 5 groups with clear boundaries. The likely-safe neighborhood ({qnli_probe, qnli_uniform}) was the key takeaway — something not obvious from the pair matrix alone at this inventory size.
