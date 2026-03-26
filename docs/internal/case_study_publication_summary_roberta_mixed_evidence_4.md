# Publication Summary — roberta_mixed_evidence_4

Series: real_inventory_case_series, wave 1
Category: 1 (messier mixed-quality) + 3 (cross-style)
Date: 2026-03-22

## One-line summary

A small RoBERTa inventory with genuinely mixed QA shows the workflow doing meaningful narrowing at every stage, including a core-space disagreement on the only low-risk pair.

## Setup

- Base model: roberta-base
- Task: SST-2 sentiment classification
- 3 adapters with 3 distinct eligibility statuses (eligible, uncertain, unknown)
- Rank policies: uniform r=8, probe-based r=32, per-layer with alpha scaling

## What happened

1. **Source QA** immediately stratified the pool: 1 eligible (beats base on SST-2), 1 uncertain (ties base), 1 unknown (no eval). Strict-QA would collapse to 1 adapter.

2. **Pair audit** found 1 low-risk, 1 medium-risk, 1 high-risk pair. The per-layer adapter with custom alpha scaling creates 6–11x norm imbalance with both other adapters.

3. **Core-space** on the low-risk pair (uniform_elig × probe_uncertain) returned **marginal** (shared_basis_score=0.878). This is the target case class: ordinary pair_risk=low but deeper inspection disagrees. The pair moved to a caution track.

4. **Neighborhoods** separated per_layer_unknown into a caution cluster and grouped the other two as likely-safe, with a boundary warning between clusters.

5. **Net result:** no straightforward merge target survives. The inventory is genuinely thin — the best available action is to proceed cautiously with the one low-risk pair (now cautioned) or invest in collecting stronger adapters.

## Where the workflow was strong

- Source QA did the most important work: it identified that 2 of 3 adapters lack credible behavioral evidence
- The workflow correctly surfaced that this inventory has no safe, confident merge — a practitioner would waste time discovering this through trial merges
- Core-space caught a case where layer-level safety was not confirmed at depth

## Where the workflow was merely helpful

- Neighborhoods echoed what the pair matrix already showed (per_layer is isolated). With only 3 adapters, neighborhoods add confirmation but not new structure.

## Where the workflow starts to strain

- With only 3 adapters, the inventory pipeline produces a lot of machinery for a small decision. The overhead is justified if source QA or core-space actually changes action — which it did here — but would not be if all adapters were eligible and all pairs were safe.

## Inventory-level lesson

**What the inventory-level view changed:** The combination of QA stratification + core-space disagreement turned a "3 adapters, one easy merge" story into "no straightforward merge, rethink the adapter set." This is a stronger conclusion than any single pair report would have produced.

**Was the inventory mistake resolved before pairwise detail became central?** Partially. Source QA identified the thin pool before pair analysis. But core-space (a deeper pairwise check) was needed to fully resolve the one remaining pair.

**Did the neighborhood result materially reduce the candidate space?** Modestly. Neighborhoods confirmed per_layer_unknown's isolation but did not add new information beyond the pair matrix at this inventory size.
