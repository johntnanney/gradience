# Merge Neighborhoods Design Note (Internal)

## Scope

This design note freezes first-pass semantics for the rule-based neighborhood suggester.

Goal: provide conservative inventory guidance using existing preflight artifacts without introducing graph UI or optimizer framing.

## Frozen terms

### candidate adapter

Adapter eligible to participate in neighborhood construction after exclusion policy is applied.

### excluded adapter

Adapter removed from neighborhood construction because policy indicates it should not be grouped.

### compatibility edge

Pairwise relation derived from `MergeQAReport` fields:
- `pair_risk`
- `recommended_strategy`
- `compatibility_score`
- adapter eligibility statuses

### merge neighborhood

Conservative adapter group formed from high-compatibility relations, with limited moderate-compatibility expansion.

### boundary warning

Cross-group warning emitted when incompatible or conditional relations exist between groups.

## Exclusion policy (v1)

Default exclusion:
- `flagged_weak`

Strict-QA exclusion (`--strict-qa`):
- `flagged_weak`
- `unknown_no_behavioral_eval`
- missing QA status in pair reports (`eligibility_status == null`)

Optional exclusion (`--exclude-unknown`):
- unknown/missing QA statuses even without strict mode

## Compatibility buckets (v1)

- `high_compatibility`
- `moderate_compatibility`
- `conditional_compatibility`
- `incompatible`

Mapping is deterministic and explainable from report fields; no hidden model scores are used.

## Grouping algorithm (v1)

1. Build candidate set and exclusion set.
2. Build compatibility edges.
3. Create initial groups from connected components of `high_compatibility` edges.
4. Attempt moderate-edge merges only when no cross-group `incompatible` or `conditional_compatibility` edge would be introduced.
5. Emit singleton groups for remaining candidates.
6. Emit boundary warnings for risky cross-group edges.

## Non-goals

- force-directed layouts
- spectral clustering
- probabilistic memberships
- UI graph rendering
