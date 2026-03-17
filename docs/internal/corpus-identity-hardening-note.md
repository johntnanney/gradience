# Corpus Identity Hardening Note

Status: planned  
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

## Candidate design directions

### Option A — Add explicit `adapter_instance_ids`

Add a new manifest field with one unique id per QA artifact, generated from stable local evidence such as:
- normalized adapter path
- optional content hash fingerprint

Pros:
- clear separation of display name vs identity
- additive field, easy to audit

Cons:
- requires migration/compat handling in summary scripts

### Option B — Use QA artifact paths as canonical identity

Treat normalized `qa_artifact_paths` as the authoritative instance keys in `summarize_corpus.py`, while retaining `adapter_names` for display only.

Pros:
- minimal schema churn
- immediate counting fix at summary layer

Cons:
- identity is path-dependent
- harder to compare moved/copied artifacts

### Option C — Hybrid (`adapter_instance_id` + canonical path)

Store both:
- stable instance id
- canonical artifact path

Pros:
- strongest long-term traceability

Cons:
- slightly more implementation overhead

## Recommended near-term approach

For the next hardening pass, prefer:

1. implement Option B in summary logic first (low-risk correction),
2. then add Option A additively in manifest schema handling when ready.

This keeps current cycle momentum while preventing misleading adapter-instance counts.

## Acceptance criteria (for future implementation)

- corpus summary reports both:
  - adapter display-name counts
  - adapter instance counts from identity-safe keys
- no change to existing policy semantics
- existing manifests continue to load (or migrate with explicit tooling)
- tests cover checkpoint-name collision cases

## Tracking

- Suggested implementation window: after Cycle-02 review, before Cycle-03 collection.
- Related docs:
  - `docs/internal/corpus-review-memo-2026-03.md`
  - `docs/internal/selective-calibration-decision-2026-03.md`
