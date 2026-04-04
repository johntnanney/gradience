# N131 — Merge-Aware Training Monitor Prototype Decision Memo

## Decision

`bounded_keep`

## What Was Built

- HF callback extension with optional `merge_target`
- per-log compatibility snapshots against one fixed reference adapter
- run-end trend summarization with transparent rule set
- bounded demo run and manifest artifacts

Implemented in:

- `gradience/vnext/integrations/hf.py`
- `gradience/vnext/integrations/merge_aware_monitor.py`

## What the Traces Showed

In the bounded demo:

- init + per-step + summary events were emitted as designed
- snapshot payloads were interpretable and compact
- run-level trend resolved to `inconclusive` (acceptable under conservative rules)

## What Remains Bounded

- no optimizer intervention
- no claim of training improvement
- no merge-policy automation
- evidence is from prototype-scale demonstration, not broad empirical validation

## Keep / Bounded-Keep / Discard Rationale

Why not `discard`:

- technical path works
- telemetry signal is structured and reviewable

Why not full `keep`:

- evidence is still narrow and mostly plumbing-focused
- interpretability under diverse real runs is not yet established

Hence: keep as internal exploratory capability with explicit bounds.

## Recommended Use

- internal diagnostics for compatibility drift inspection
- research-side telemetry experiments in small bounded runs
- not for automated training control or public claims
