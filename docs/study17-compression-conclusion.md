# Study 17 Compression Conclusion and Near-Term Product Direction

**Status:** Binding product/research decision for the near term.

**Date:** 2026-03-11

## Executive Decision

Compression should not be treated as a default or central Gradience workflow feature.

Compression may remain in the codebase as an experimental, gated, advanced option for narrow high-risk cases, but the main product direction is:

- adapter QA
- eligibility screening
- pairwise merge-risk reporting
- inventory preflight / aggregation

That is the validated product spine.

## What We Tested

### Study 17A

**Question:** Does 95% cumulative-energy compression improve merge outcomes as a default pre-merge cleanup rule?

**Result:** No meaningful structural improvement. Effective-rank reductions were too small to matter. 95% compression was effectively a no-op in primary cases.

**Conclusion:** 95% compression is too conservative to function as a useful default workflow step.

### Study 17B

**Question:** If 95% is too mild, does more aggressive compression help enough to justify the added distortion?

**Primary behavioral comparison:** full_normeq, comp90_normeq, comp80_normeq on pair_03 and pair_04.

**Result:** Aggressive compression was behaviorally low-cost. Some worst-side improvements appeared. Effect sizes were small. No large or transformative behavioral gains emerged.

**Conclusion:** Compression survived. Compression did not win.

## What the Results Mean

### 1. Compression is not fake

The Study 17B results do not suggest that compression is destructive or useless in all cases. Aggressive compression can be behaviorally survivable, sometimes slightly helpful, especially in certain high-risk merge settings. That matters.

### 2. Compression is not strong enough to be a product pillar

The gains are too modest to justify making compression part of the default documented workflow, centering the product around compress-then-merge, or treating spectral truncation as a major validated intervention. The evidence does not support that.

### 3. The stronger validated story remains elsewhere

The more reliable, better-supported Gradience value is: detecting weak source adapters, identifying domination risk and structural merge risk, surfacing clear warnings and caveats, providing machine-readable preflight artifacts, and enabling inventory-level summary and filtering. This is the current center of gravity of the product.

## Product Decision

### Core Gradience identity

Gradience is positioned primarily as a **preflight QA and merge-risk layer for LoRA adapter decisions**. That includes:

- single-adapter QA
- source eligibility judgment
- pairwise merge-risk reporting
- strict-QA blocking behavior
- inventory summary / aggregation

### Compression status

Compression is retained only as: experimental, advanced, gated, non-default.

Compression is not: a headline feature, the main recommendation path, part of the default getting-started story, or a broadly validated pre-merge requirement.

## Practical Consequences

### Keep

- compression code paths
- threshold-sensitivity findings
- experimental documentation for advanced users

### Change

- demote compression in README and top-level docs
- remove compression from the default workflow narrative
- clearly label compression as experimental / advanced
- avoid implying compression is part of the standard recommendation stack

### Do not do

- do not rip compression out entirely
- do not continue to present it as a core workflow branch
- do not prioritize more compression work over QA/reporting/inventory work

## Default Documented Workflow

1. `gradience audit-adapter`
2. `gradience merge-audit`
3. `gradience summarize-inventory`

Compression appears only in experimental features docs, advanced usage notes, and research notes.

## Roadmap Priority

**Near-term:** Continue investing in adapter QA, eligibility screening, merge-risk reporting, inventory-level aggregation, scripting/reliability/workflow clarity.

**Lower priority:** Further compression-led product work, broader compression theory expansion, compression as a primary CLI/documentation path.

Compression work may continue only if a future experiment addresses a new roadmap-relevant question with a plausible chance of changing product decisions. For now, that threshold is not met.

## Final Conclusion

Compression is behaviorally low-cost in the tested high-risk cases, but its benefits are too modest to justify making it a default or central Gradience workflow feature.

Gradience should be developed primarily as an adapter QA, merge-risk reporting, and inventory preflight system. Compression remains available only as an experimental, gated, advanced option.

## Immediate Actions Taken

1. Updated README and top-level docs to center the preflight spine
2. Moved compression language into experimental / advanced sections
3. Getting-started workflow is compression-free
4. This memo added to `docs/` as the current product decision
5. Compression branch will not be reopened without a genuinely new fork-setting question
