# Merge-Aware Training Monitor (Bounded Strategy Summary)

## Status

`bounded_keep` (internal diagnostic prototype)

## What Is Supported Now

- optional HF callback mode that compares current adapter state against one
  fixed reference adapter during training
- periodic compatibility snapshots using existing merge-audit-compatible metrics
- conservative run-end trend labels (`toward`, `away`, `mixed`, `inconclusive`)

## What Is Not Supported

- optimizer feedback or loss shaping
- auto-stop / auto-steer behavior
- claims of improved downstream training outcomes
- broad product-level training guidance

## Operational Guidance

- keep this in research/internal workflow only
- use as visibility tooling for compatibility drift, not control logic
- require additional real-run evidence before any elevation
- for reference selection, prefer same-task first, then same-family fallback
  (`cross-task` remains exploratory)

## Canonical References

- `docs/design/merge_aware_training_monitor.md`
- `docs/design/merge_aware_monitor_trend_rules.md`
- `field_trials/merge_aware_monitor_demo/demo_summary.md`
- `sidecar/notes/n131_merge_aware_monitor_keep_boundedkeep_discard_memo.md`
- `field_trials/merge_aware_monitor_reference_choice/reference_type_comparison.md`
- `sidecar/notes/n132_merge_aware_monitor_reference_choice_memo.md`
