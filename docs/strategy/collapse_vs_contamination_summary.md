# Collapse vs Contamination Replication Summary

Date: 2026-03-31  
Status: complete (bounded replication pass)

## Purpose

Test whether the collapse-vs-contamination behavioral distinction survives modest replication strongly enough to treat it as stable but bounded.

## Study outputs

- `sidecar/notes/n118_collapse_vs_contamination_baseline.md`
- `sidecar/notes/n119_collapse_vs_contamination_replication_panel.md`
- `sidecar/notes/n120_collapse_vs_contamination_rerun.md`
- `sidecar/notes/n121_collapse_vs_contamination_verdict.md`
- `sidecar/notes/n122_collapse_vs_contamination_replication_memo.md`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/`

## Main result

Overall verdict: `replicated_with_guardrails`.

What replicated:

1. Collapse-like targets remained uncertainty-dominant (higher confidence-collapse, near-zero high-confidence wrong).
2. Contamination-like targets remained confident-wrong-dominant (higher high-confidence wrong, low confidence-collapse).
3. Neither-source rates stayed relatively close across channels, reinforcing that confidence-channel metrics carry the distinction.

## Why this matters

This strengthens one of Route 2's highest-value behavioral explanations:

- similar novel-failure pressure can still correspond to different operational channels.

## Guardrails

Keep bounded:

1. merge-facing decision context
2. tested case family/backbone scope
3. no universal cross-context channel law claim yet

## Route 2 language update

Safe to say:

- "Collapse-like and contamination-like failures are behaviorally distinct channels in tested merge-facing settings."
- "Confidence-channel metrics (collapse vs confident-wrong) are the key discriminators."

Still guarded:

- broader scenario/artifact-class portability without further replication.

## Bottom line

Collapse-vs-contamination is now better treated as a **stable bounded Route 2 behavioral distinction** rather than a single-panel artifact.
