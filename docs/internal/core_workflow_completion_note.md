# Core Workflow Completion Note

**Date:** 2026-03-26
**Decision Point 1 verdict:** Passed. Core Gradience is workflow-complete enough to shift primary emphasis to the sidecar.

## What is now complete in core

The inventory preflight workflow has five working layers:

1. **Actionability.** The INVENTORY ACTION PLAN partitions an adapter pool into same-task safe zone, cross-task caution zone, evaluate-first subset, and excluded sources. A user can open one summary and see a clear plan without mentally re-synthesizing the underlying outputs.

2. **Repeatability.** Run bundles package each preflight into a standard layout (preflight_summary.md, preflight_summary.json, run_manifest.json, inventory_action_plan.md, compare_to_previous.md). Auto-discovery of previous runs eliminates manual chaining. Batch summary provides a cross-run comparison table.

3. **Trust clarity.** Evidence tiers (behavioral_reported, behavioral_weak, behavioral_missing) are tracked in the inventory summary, surfaced in the terminal formatter, threaded through the markdown outputs, the JSON outputs, and the run-to-run comparison. The non-verification disclaimer is consistent: "user-reported," never "verified."

4. **Usability.** The preflight workflow doc, mixed-task walkthrough, and same-task control walkthrough give a new user a working path. Summary blocks are structured, reduced-candidate-set presentation includes per-pair risk and strategy, and all three output formats (terminal, markdown, JSON) agree on field names and values.

5. **Evidence.** The utility round supports the core claim: 65-90% candidate reduction in mixed-task inventories, 81% average where the advisory is the main discriminator.

## What remains maintenance-only

Core Gradience should now receive only:

- Bug fixes
- Wording cleanup when needed
- Low-cost UX improvements that are clearly worth the effort
- Test coverage improvements as CI environments stabilize

No new conceptual expansion in core unless it directly solves an immediate workflow problem that a real user has encountered.

## What should NOT be done in core before the sidecar advances

- No additional provenance layers beyond what exists
- No larger batch orchestration system
- No severity grading integration (that is a sidecar question)
- No schema changes beyond additive fields
- No new output formats

## What research should now move to the sidecar

The biggest unanswered questions are no longer core workflow questions. They are:

- **Instability.** Is instability more portable than severity? Can pairs be grouped into stable vs. unstable regimes? Does instability explain why severity signals failed to generalize?
- **Catastrophic anchors.** What makes a (task-pair × backbone) combination catastrophic? Can this be predicted from structural signals?
- **Local/mechanistic interpretation.** Do per-layer patterns (norm concentration, layerwise conflict, structural dispersion) reveal more than coarse global summaries?
- **DeBERTa adjudication.** Does the instability ranking survive a third backbone? This is the decisive test.

## What kinds of future core ideas should be rejected

Any proposal to expand core should meet all five of these criteria:

1. It solves a real workflow problem that a user has actually encountered
2. It replicates across backbones or across clearly defined regimes
3. It improves decisions beyond current stable signals
4. It can be expressed simply and conservatively
5. It does not add more conceptual overhead than practical value

If a proposal fails any one of these, it belongs in the sidecar or nowhere.

## Bottom line

Core Gradience now does what it should do: reduce a messy adapter pool to a defensible set of candidates, with honest provenance, repeatable bundles, and clear action guidance. The next serious intellectual work belongs in the sidecar.
