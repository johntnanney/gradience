# Checkpoint Triage Language and Boundaries

Date: 2026-03-31  
Status: Route 2 wording guide for checkpoint workflows

## Purpose

This note defines which terms transfer from adapter workflows, which need checkpoint-specific wording, and which merge-centered terms should be avoided in checkpoint triage outputs.

## Terms that transfer unchanged

- `evidence bootstrap`
- `QA status` (`eligible`, `uncertain`, `flagged_weak`)
- `same_task`, `same_family`, `cross_task`
- `evaluate-first`, `near-miss`, `optional review`, `exclude`
- `conservative narrowing`

These terms remain operationally correct in checkpoint inventories.

## Terms that require checkpoint-specific translation

| Adapter-era term | Checkpoint-triage wording |
|---|---|
| adapter | checkpoint |
| LoRA factors | checkpoint-delta summaries |
| merge report | pairwise compatibility summary |
| merge candidate | evaluation priority candidate |
| adapter inventory | checkpoint inventory |

## Terms to avoid in checkpoint triage artifacts

Avoid these unless merge execution is truly being performed:

- `merge-safe`
- `merge-ready`
- `merge now`
- `strategy execution` language (`ties`, `dare_ties`, etc.)

For checkpoint triage, use decision language such as:

- `evaluate-first`,
- `review before spending evaluation budget`,
- `blocked by source QA`.

## Same-family definition in checkpoint space

In checkpoint inventories, `same_family` means:

- non-identical tasks,
- with declared task-family relation,
- under one shared base model,
- treated as informationally related but not interchangeable.

T02 example: SST-2 and Yelp were treated as same-family (`sentiment_binary`) with cautious review routing, not automatic retention.

## Boundary statements to keep explicit

- Checkpoint triage is evidence-aware prioritization, not checkpoint merging.
- Representation C supports audit and pairwise triage decisions; it does not imply full checkpoint execution support.
- Same-family does not imply same-dataset transferability or interchangeable performance.

## Recommended sentence templates

- `This inventory is QA-dominated; structural plausibility is secondary until source evidence improves.`
- `Same-family pairs are routed to optional review, not automatically retained.`
- `No merge execution recommendation is issued for checkpoint triage outputs.`
