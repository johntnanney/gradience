# Checkpoint Inventory Trials Summary (T01 + T02)

Date: 2026-03-31  
Scope: CPU-only, shared-base `distilbert-base-uncased`, classification checkpoints

## Trial coverage

| Trial | Checkpoints | Pairs | Relationship coverage | Dominant driver | Retained count |
|---|---|---|---|---|---|
| T01 | 4 | 6 | same-task + cross-task | source_qa | 0 |
| T02 | 5 | 10 | same-task + same-family + cross-task | source_qa | 0 |

## What the two trials establish together

1. End-to-end checkpoint-inventory triage is operational in a normalized bundle format.
2. Evidence bootstrap and source QA remain the practical gate outside adapter-only workflows.
3. Same-family branching in checkpoint space is usable and legible (T02).
4. Workflow stays conservative (frequent QA blocks) while still useful (clear review/evaluation priorities).

## Follow-through highlights

- T01: same-task near-miss probe and control checkpoints produced interpretable sanity checks despite zero retained pairs.
- T02: same-family probe showed asymmetric behavior across datasets (SST-2 weak, Yelp strong), reinforcing that same-family is a review relation, not interchangeability.

## Canonical checkpoint trial bundle shape

Each checkpoint inventory trial should include:

- `manifest.json`
- `evidence/bootstrap_results.json`
- `qa_artifacts/*.json`
- `pairwise/pairwise_results.json`
- `preflight/` (inventory summary/action plan/review packet)
- `eval_results.json`
- `field_note.md`
- `trial_memo.md`

## Current Route 2 judgment

Checkpoint inventory triage is a credible broader workflow class in bounded scope. It should remain explicitly constrained to shared-base small-encoder classification settings on CPU until additional inventories broaden coverage.
