# Corpus Identity Hardening Note

Status: implemented  
Date: 2026-03-17

## Why this note exists

Cycle-01 showed a corpus accounting caveat: multiple adapter QA artifacts can share the same human-readable `adapter_name` (for example checkpoint directories all named `checkpoint-50`).  
When this happens, manifest-level adapter identity can collapse and undercount instance-level coverage.

This is a corpus metadata hardening task, not a policy calibration task.

## Problem statement

Current corpus manifests encode:
- `adapter_names` (string set)
- artifact paths

This is useful for readability, but not sufficient for robust instance identity in all workflows.

Observed risk:
- inventory-level adapter instance counts can be misleading when names collide
- downstream review metrics can appear cleaner than underlying sample diversity

## Scope

In scope:
- improve adapter instance identity representation in corpus manifests
- preserve current human-readable naming fields
- keep compatibility with existing cycle artifacts where possible

Out of scope:
- any change to strict-QA semantics
- any change to recommendation logic
- any change to neighborhood/core-space algorithms
- any feature expansion beyond corpus metadata contracts

## Implemented approach (Cycle-03 hardening patch)

Implemented in `scripts/summarize_corpus.py`:

1. deterministic adapter-instance key resolution
2. identity-safe unique counting across manifests
3. separate human-readable display-name counting

Identity source precedence (first available wins):

1. manifest-level `adapter_instance_ids[index]` (future-compatible optional field)
2. QA artifact-level `adapter.instance_id` (future-compatible optional field)
3. canonicalized QA `adapter.path`
4. canonicalized QA artifact reference path
5. stable hash fallback of canonical QA path payload

This keeps display labels (`adapter_names`) for readability while ensuring corpus counts do not depend on label uniqueness.

## Counting semantics

Corpus summary now uses:

- `adapter_instance_count`: unique adapter instances across all manifests by identity key
- `unique_adapter_count`: alias of the same identity-safe unique count (backward compatibility)
- `unique_adapter_display_name_count`: unique human-readable adapter labels

Deduplication rule:
- if the same underlying instance key appears in multiple manifests, it counts once in corpus-level adapter-instance totals.

## Acceptance criteria (for future implementation)

Implemented and validated:

- corpus summary reports both identity-safe instance counts and display-name counts
- no change to strict-QA, recommendation logic, neighborhoods, or thresholds
- existing manifests continue to strict-load without schema changes
- regression tests cover duplicate display-name, repeated-reference dedupe, and mixed explicit/fallback identity scenarios

## Tracking

- Implementation window: completed during Cycle-03 review hardening.
- Related docs:
  - `docs/internal/corpus.md`
  - `docs/internal/corpus-review-memo-2026-05.md`
  - `docs/internal/selective-calibration-decision-2026-05.md`
