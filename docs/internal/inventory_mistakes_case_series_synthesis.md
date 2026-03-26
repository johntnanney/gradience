# Inventory Mistakes Before Pair Mistakes — Case Series Synthesis

Aggregate synthesis across 3 case-study inventories (1 published + 2 series wave 1).

## Summary of evidence

### How often did source QA do early meaningful narrowing?

**Every time (3/3).**

| Inventory | What QA did | Magnitude |
|-----------|-------------|-----------|
| case_study_qnli4 | Excluded 1 flagged_weak adapter | Removed 3 of 6 pairs |
| roberta_mixed_evidence_4 | Stratified 3 distinct statuses (eligible/uncertain/unknown) | Strict-QA collapses to 1 adapter |
| distilbert_large_pool_6 | Identified 4/6 adapters with no behavioral eval | Strict-QA blocks 14 of 15 pairs |

Source QA was the single most impactful narrowing step in all three inventories. In the largest inventory (6 adapters), source QA alone reduced the candidate space by >90%.

### How often did neighborhoods reduce the candidate space?

**Scaled with inventory size.**

| Inventory | Adapters | Neighborhood contribution |
|-----------|----------|--------------------------|
| case_study_qnli4 | 4 | Modest — confirmed pair-matrix structure |
| roberta_mixed_evidence_4 | 3 | Modest — confirmed per_layer isolation |
| distilbert_large_pool_6 | 6 | **Strong** — compressed 15 pairs into 5 groups, identified single safe neighborhood |

At 3–4 adapters, neighborhoods confirm what the pair matrix already shows. At 6 adapters, they add genuine compression — the pair matrix becomes hard to parse manually, and neighborhoods make the answer obvious.

### Does the phrase have enough evidence?

**Yes, for a follow-up blog post or paper subsection.**

The pattern is consistent across all three inventories:
1. Source QA does the heaviest lifting
2. The inventory-level view always changed the practical next step
3. Skipping source screening would have led to wasted exploration in every case

The evidence is strong enough to support:
- **A follow-up blog post** — yes, with concrete examples from all three inventories
- **A paper subsection** — yes, as a short empirical finding (not a full study)
- **A later focused small study** — not yet needed; the current evidence is already persuasive for the scope of the claim

### Caveats

- All inventories used distilbert or roberta base models. Larger models may behave differently.
- "Unknown" adapters dominate the narrowing. In an inventory where all adapters have behavioral eval, source QA would do less.
- The claim is about inventory *workflow*, not about thresholds. Different thresholds would shift the details but not the pattern.
- 3 inventories is a small sample. The pattern is consistent but not yet robust to adversarial construction.
