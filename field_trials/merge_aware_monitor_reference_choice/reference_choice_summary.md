# Reference-Choice Summary

## Run Snapshot

- study: `merge_aware_monitor_reference_choice`
- training run: `tiny_encoder_reference_choice_demo_v1`
- trace source: `runs/*/run.jsonl`
- total reference conditions: 3

## Per-Condition Summary

| reference_type | relation | snapshots | overlap_trend | score_trend | run_level_label | interpretability |
|---|---|---:|---|---|---|---|
| `same_task` | same_task | 6 | `increasing` | `increasing` | `toward_compatibility` | `interpretable` |
| `same_family` | same_family | 6 | `mixed_unstable` | `mixed_unstable` | `mixed` | `mixed` |
| `cross_task` | cross_task | 6 | `mixed_unstable` | `decreasing` | `inconclusive` | `inconclusive` |

## Quick Read

- Same-task gave the cleanest trajectory in this bounded demo.
- Same-family produced a partially readable but mixed pattern.
- Cross-task was mostly not interpretable under current trend rules.

## Boundedness

This result is from a tiny synthetic relation-labeled demo. It is useful for
internal usage guidance only, not predictive or product claims.
