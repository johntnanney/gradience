# Product Validation Summary

**Status:** Operationally validated across 5 inventories, 3 backbones, 4 task families, 16 evaluated merges.

---

## What the preflight pipeline does

Gradience takes a directory of LoRA adapters and reduces the candidate merge space to a small number of promising pairs, ranked by structural plausibility and behavioral evidence. The pipeline: audit each adapter's spectral profile → bootstrap behavioral evidence (500-sample CPU evals) → audit all eligible pairwise merges → produce an inventory summary, action plan, and HTML report.

## Field trial results

| Category | Pairs evaluated | Avg Δ vs best source | Improvers |
|----------|----------------|----------------------|-----------|
| Retained same-task | 7 | -0.024 | 2/7 (29%) |
| Near-miss | 7 | -0.006 | 1/7 (14%) |
| Cross-task control | 4 | -0.047 | 0/4 (0%) |

The pipeline reduces candidate space by 90–93% (10→1, 28→2). Retained pairs are the correct first choices. Cross-task controls consistently degrade more.

## Key validated capabilities

**Task-boundary detection:** Zero false positives across 5 inventories and 53+ same-task pairs. The same-task/cross-task classification is the pipeline's most reliable gate.

**Evidence gate:** The three-way eligibility classification (eligible / uncertain / flagged_weak) is the highest-impact single feature. Without behavioral evidence, the pipeline produces nothing useful. With it, the gate correctly handles the full range from strong performers to genuine failures.

**Near-miss detection:** Same-task pairs excluded only because one source lacks sufficient evidence degrade comparably to retained pairs (avg Δ = -0.006) and 5× less than cross-task controls. Confirmed across 3 backbones and 3 task families. Weak-source severity modulates the outcome: sources that barely miss the gate produce merges indistinguishable from retained; deeply weak sources introduce more variance but still outperform cross-task exclusions.

## Not yet validated

Inventories with >28 pairs. High-rank adapters (r≥32). Generation tasks. Non-accuracy metrics. Multi-task adapters targeting different module sets.

---

*Full details: `field_trials/product_validation_memo.md`, `field_trials/near_miss_validation.md`*
