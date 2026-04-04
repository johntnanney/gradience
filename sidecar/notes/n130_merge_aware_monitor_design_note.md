# N130 — Merge-Aware Training Monitor Design Freeze

## Snapshot

This note freezes the v1 prototype contract for merge-aware training monitoring.

Design intent:

- reuse merge-audit-compatible measurements during training
- emit diagnostic telemetry snapshots relative to a fixed reference adapter
- avoid any optimization intervention

## v1 Contract

- input: optional single `merge_target`
- cadence: callback log cadence (`merge_monitor_every`)
- core measurements:
  - overlap
  - directional agreement
  - conflict fraction
  - imbalance fraction
  - compatibility score
  - overall verdict
- outputs:
  - `merge_aware_monitor` init metric
  - `merge_aware_compatibility` per-step metrics
  - `merge_aware_summary` run-end trend summary

## Boundaries

- CPU-only
- diagnostic-only
- no optimizer/loss control
- no merge-policy actioning from this signal

## Canonical design docs

- `docs/design/merge_aware_training_monitor.md`
- `docs/design/merge_aware_monitor_trend_rules.md`
