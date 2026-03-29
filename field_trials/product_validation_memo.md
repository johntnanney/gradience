# Product Validation Memo

**Gradience Preflight Pipeline — Field Trial Results**
**Date:** 2026-03-28
**Status:** Operationally validated

---

## What was validated

The full Gradience preflight workflow — adapter audit, behavioral evidence bootstrap, pairwise merge audit, inventory summary, action plan, and HTML report — was tested across 5 inventories, 3 backbones, 4 task families, and 16 evaluated merges.

## Core finding: the narrowing logic works

Gradience reduces candidate merge space by 90–93% (10→1, 28→2) and the retained pairs are the right first choices. Across all evaluated merges, retained same-task pairs either improve over both sources (+0.028, +0.006) or degrade modestly (-0.006 to -0.088), while cross-task controls degrade substantially more (-0.042 to -0.096).

| Category | Pairs evaluated | Avg Δ vs best source | Improvers |
|----------|----------------|----------------------|-----------|
| Retained same-task | 7 | -0.024 | 2/7 (29%) |
| Near-miss | 7 | -0.006 | 1/7 (14%) |
| Cross-task control | 4 | -0.047 | 0/4 (0%) |

## Evidence gate

The three-way eligibility classification (eligible / uncertain / flagged_weak) is the most impactful single feature. Without behavioral evidence, the pipeline produces nothing useful (Pilot 1). With evidence, it correctly handles genuine failures, misleading evals, marginal passes, ambiguous ties, and strong performers. The gate is well-calibrated except at the margin, where adapters that barely beat base (delta +0.01 to +0.06) pass as eligible but contribute little to merges.

## Near-miss: confirmed

Near-miss pairs — same-task, structurally plausible, excluded only because one source is evidence-constrained — degrade comparably to retained pairs and 5× less than cross-task controls. The pattern holds across distilbert (irony, hate), bert-base-uncased (hate, emotion), and roberta (ag_news). Weak-source severity modulates the outcome: sources that barely miss the gate (delta -0.002 to -0.004) produce merges indistinguishable from retained; deeply weak sources (delta -0.150) introduce more variance.

The near-miss action-plan section is implemented and validated. No further product change is required.

## What is now operationally validated

- Candidate reduction (90%+ across inventories of 10–28 pairs)
- Retained-pair prioritization (correct ordering in all tested inventories)
- Task-boundary detection (zero false positives across 5 inventories)
- Evidence gate (three-way classification, correct across the full range)
- Near-miss detection (confirmed across 3 backbones and 3 task families)
- Action plan rendering (terminal, markdown, HTML, preflight summary JSON)

## What is not yet validated

- Inventories with >28 pairs
- High-rank adapters (r≥32)
- Generation tasks
- Non-accuracy metrics (F1, BLEU, perplexity)
- Multi-task adapters targeting different module sets
