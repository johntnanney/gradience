# Near-Miss Validation

**Conclusion:** Confirmed. The near-miss category is a practically useful middle ground between retained pairs and excluded controls. The current action-plan implementation is sufficient. No further product change required.

---

## Evidence

Phase 2b evaluated 11 merge pairs across 2 new inventories (distilbert irony cluster, bert hate+emotion) plus the original Pilot 3 near-miss. Three backbones, three task families.

| Category | Pairs | Avg Δ vs best source | Improvers | Avg Δ vs avg source |
|----------|-------|----------------------|-----------|---------------------|
| Retained | 4 | -0.018 | 1/4 | +0.064 |
| Near-miss | 7 | -0.006 | 1/7 | +0.058 |
| Cross-task control | 1 | -0.096 | 0/1 | +0.078 |

Near-miss pairs degrade comparably to retained pairs and 5× less than the cross-task control. They occupy the retained neighborhood, not the excluded neighborhood.

## Weak-source modulation

How weak the excluded source is modulates the outcome:

| Weak source delta vs base | Near-miss Δ vs best source | Interpretation |
|---------------------------|----------------------------|----------------|
| -0.002 to -0.004 | -0.002 to -0.012 | Indistinguishable from retained |
| -0.150 | -0.002 to -0.088 | Higher variance, still better than controls |

Sources that barely miss the evidence gate produce near-miss merges essentially identical to retained pairs. Deeply weak sources introduce more variance but still outperform cross-task exclusions.

## What was implemented

The action plan now includes a near-miss section between same-task safe zone and cross-task caution: "Structurally plausible, evidence-constrained. Optional if evaluation budget allows." This appears in the terminal renderer, markdown action plan, preflight summary, and HTML report. 9 unit tests cover detection logic and rendering. The feature is additive and backward-compatible.

## Scope limitation

All near-miss evidence comes from r=1 TransferGraph adapters on classification tasks with 500-sample CPU evals. The pattern should be re-examined when the adapter ecosystem includes higher-rank adapters, generation tasks, or larger evaluation budgets.
