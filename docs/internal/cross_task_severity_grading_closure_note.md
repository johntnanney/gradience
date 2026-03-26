# Cross-Task Severity Grading — Closure Note

## Status: closed for now

The cross-task severity grading research line is paused. Three studies tested whether any available signal can reliably predict severity within cross-task pairs across backbones. None can.

## What was tested

| Signal | DistilBERT result | RoBERTa result | Cross-backbone? |
|--------|------------------|----------------|-----------------|
| Exact task-pair identity | Strong predictor (stable across seeds) | Different severity profile | **No** — backbone-dependent |
| Core-space shared-basis | r = -0.614 (promising) | r = +0.273 (sign flips) | **No** — does not replicate |
| Task-format similarity | Misleading (same-format contains catastrophic) | Same | **No** |
| Source-strength gap | Confounded with task-pair identity | Same | **No** |
| Pair-risk | 83% rated medium regardless of severity | Similar | **No** |

## What IS solved

The **boundary** between same-task (safe) and cross-task (degraded) is solved:
- Task-relationship advisory: 0 false positives across 132+ pairs on 2 backbones
- Same-task safety: 49 pairs, 0 material degradations across 3 blind-spot studies
- Cross-task degradation: 0 near-safe cross-task pairs on either backbone

## What is NOT solved

**Severity grading within cross-task pairs.** Once the advisory flags a pair as cross-task, current Gradience cannot predict whether it will degrade by 2pp or 42pp. No tested signal provides this reliably across backbones.

## Why this is acceptable

The advisory already provides the most important practical information: "this is a cross-task merge — proceed with caution." A practitioner who heeds this warning avoids all catastrophic outcomes. The inability to grade severity within the caution zone is a limitation, not a failure — the boundary itself prevents the worst decisions.

## When to reopen

Reopen severity grading research only if:
1. A new structural signal is hypothesized that has a plausible mechanism for backbone-independence
2. The project moves to larger models where the severity landscape may differ
3. A practitioner use case requires finer severity grading to be worth the research cost

Do not reopen to try more variants of the same signals that already failed to replicate.

## Evidence base

- 28-pair DistilBERT subtype study: 4 severity levels confirmed, task-pair identity strongest predictor
- 28-pair RoBERTa generalization study: boundary replicates, severity does not
- 48-pair core-space replication: shared-basis correlation sign flips across backbones (r=-0.614 → r=+0.273)
- Total: 56 cross-task pairs across 2 backbones, 0 reliable cross-backbone severity signals found
