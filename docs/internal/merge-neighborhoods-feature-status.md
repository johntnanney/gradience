# Merge Neighborhoods Feature Status

## Purpose

Rule-based inventory-level grouping aid built from existing `AdapterQAArtifact` and `MergeQAReport` inputs.

## Current status

- classification: **advanced workflow extension (practitioner-usable)**
- CLI exposure: `suggest-neighborhoods`
- schema surface: `gradience.merge_neighborhoods/v1`
- public API export: **not promoted yet**

## Why it is considered usable now

- outputs are conservative and explainable from existing artifact fields.
- weak-adapter exclusion behavior is reliable.
- boundary warnings are actionable for cross-group risk checks.
- human review indicates output is understandable and useful without graph tooling.

## Promotion gate (evidence required)

Promote to broader public/API surfaces after:

1. multiple real inventories confirm stable utility and low false confidence.
2. wording and operational guidance are clear for practitioners.
3. no pressure to introduce non-explainable grouping logic.

## Near-term guidance

- keep current conservative grouping logic
- polish wording only when it improves clarity
- treat as advanced inventory workflow, not default onboarding path
