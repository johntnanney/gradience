# Task-Relationship Advisory — Validation Evidence Card

## Signal

`task_relationship_advisory` field on `MergeQAReport`. Present when source QA artifacts have different `eval_dataset` values. Additive — does not alter pair-risk, core-space, or recommendation logic.

## Validation summary (combined across 2 backbones)

| Metric | DistilBERT | RoBERTa | Combined |
|--------|-----------|---------|----------|
| Same-task pairs with advisory | 0/14 (0%) | 0/2 (0%) | **0/16 (0%)** |
| Different-task pairs with advisory | 32/32 (100%) | 4/4 (100%) | **36/36 (100%)** |
| False positives | 0 | 0 | **0** |
| Task identity as merge-safety discriminator | 23/23 | 6/6 | **29/29 (100%)** |
| Main effect | Inventory-level partitioning | Same | Same |

## Evidence base

- **Verified adjudication — DistilBERT (23 pairs):** Task identity was a perfect discriminator of merge safety. 9/9 same-task pairs safe, 14/14 different-task pairs degraded. Neither pair-risk (65% accuracy) nor core-space (~57%) achieved this separation.
- **Advisory validation round — DistilBERT (46 pairs, 5 inventories):** Advisory fired on all 32 different-task pairs, stayed silent on all 14 same-task pairs. Zero false positives across all regimes.
- **Replication round — RoBERTa-base (6 pairs):** Same pattern replicated on a second backbone (125M params vs 66M). Same-task merges safe (<=2pp). Cross-task merges degraded QNLI by 7-12pp. Advisory fired correctly on all 4 cross-task pairs, silent on both same-task pairs. Pair-risk blind spot was actually worse on RoBERTa — 3 of 4 cross-task pairs rated `low` risk despite substantial degradation.

## Interpretation

The advisory's primary value is inventory-level, not pair-level. It cleanly partitions a mixed-task pair matrix into same-task safe zones and cross-task caution zones. This is information that pair-risk, QA, and neighborhoods do not provide.

The blind spot it addresses — pair-risk rating cross-task pairs as structurally compatible when they actually degrade the weaker task — is backbone-independent within the small encoder regime. The advisory is the only signal that catches it.

## Backbones tested

- distilbert-base-uncased (66M params)
- roberta-base (125M params)

## Observation round (5 additional inventories, 33 pairs)

Advisory behavior in real inventory use:
- Caution-raising: 13/25 (52%) — concentrated on medium-risk cross-task pairs
- Redundant: 12/25 (48%) — pair-risk already high
- Strongest single result: 6-adapter/15-pair inventory where advisory collapsed 11 medium-risk candidates to 2 actionable same-task pairs

## Known overcaution regime

**Same-task, different-domain, high cross-domain transfer.** A 15-pair domain-shift study across 3 binary sentiment domains (movies, restaurants, products) found that the advisory fires on all 12 cross-domain pairs — but all 12 merges are actually safe (0 materially degraded, max delta 2.2pp). The advisory correctly reports a metadata fact (eval_dataset differs) but the underlying task features transfer well enough that domain shift does not cause degradation.

This is acceptable behavior. The advisory makes no behavioral prediction — it flags a metadata difference for the practitioner to consider. In high-transfer task families, the caution is overcautious but not misleading.

## Status

**Established stable interpretive layer.** The advisory is canonical infrastructure for mixed-task inventories. Its silence on same-task pairs is correct behavior confirmed by 3 blind-spot studies (45 pairs, 0 material degradations). Its strongest value is inventory-level partitioning of cross-task pools. One known overcaution regime documented (same-task domain shift with high transfer). See:
- `docs/internal/task_relationship_advisory_round_01_synthesis.md`
- `docs/internal/task_advisory_replication_implications.md`
- `docs/internal/roberta_replication_results.md`
- `docs/internal/domain_shift_blind_spot_results.md`
