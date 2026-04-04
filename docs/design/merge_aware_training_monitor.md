# Merge-Aware Training Monitor (Diagnostic Prototype)

## Scope

This design defines a diagnostic-only training callback extension that reuses
Gradience pairwise compatibility measurements during training.

It is intentionally:

- measurement-only
- CPU-friendly
- optional and low-overhead
- non-interventionist

It is not an optimizer, rollback policy, or loss-term controller.

## Monitoring Contract

### Reference mode

`GradienceCallbackConfig` supports an optional single reference adapter:

- `merge_target: str | Path | None`

If unset, the callback remains a no-op for merge-aware monitoring.

### Sampling event

Sampling is aligned to existing callback log events:

- at each `on_log` step where `step % merge_monitor_every == 0`

No extra training-loop hooks are introduced.

### Measurement bundle

Per snapshot (current training adapter vs reference), emit:

- overlap summary (`mean_overlap`)
- directional agreement (`mean_agreement`)
- conflict fraction
- imbalance fraction
- compatibility score
- overall verdict
- shared-layer counts/status fields

These are computed by reusing existing merge-audit substrate.

### Output shape

Telemetry emits:

1. init event
   - `metrics(kind="merge_aware_monitor")`
2. snapshot events
   - `metrics(kind="merge_aware_compatibility")`
3. run-end trend summary
   - `metrics(kind="merge_aware_summary")`

Snapshot payloads are compact and JSON-safe.

## Guardrails

- No optimizer/gradient/model-control changes.
- Failures are non-fatal and surfaced as warning alerts.
- Trend summary is conservative (`toward_compatibility`, `away_from_compatibility`, `mixed`, `inconclusive`).
- No merge-policy recommendation is implied by this callback.

## Current Prototype Status

Implemented in:

- `gradience/vnext/integrations/merge_aware_monitor.py`
- `gradience/vnext/integrations/hf.py` (optional callback integration)

Initial bounded demo artifacts:

- `field_trials/merge_aware_monitor_demo/demo_runs_manifest.json`
- `field_trials/merge_aware_monitor_demo/demo_summary.md`
