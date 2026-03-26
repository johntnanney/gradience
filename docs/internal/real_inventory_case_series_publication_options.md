# Real Inventory Case Series — Publication Options

## Possible outward-facing outputs

### Option A — Follow-up blog post

**Title concept:** "Three Inventories, One Pattern: Source QA Before Pair Analysis"

**Content:** Walk through the three inventories showing how source QA consistently did the heaviest narrowing and neighborhoods scaled with pool size. Core-space flagged structural divergence on cross-task pairs, but verified adjudication showed ordinary pair-risk already separates these. Frame core-space as structurally informative but behaviorally narrow in the tested regime. Use the concrete numbers (14/15 pairs blocked) and the adjudication results honestly.

**Strength:** Directly extends the first worked-example post. Concrete, evidence-backed, not speculative.

**Risk:** Three inventories is a small sample. The post needs to be honest about that.

### Option B — Paper subsection

**Title concept:** "Empirical note: inventory-level screening in LoRA merge preflight"

**Content:** Short empirical subsection (~2 pages) documenting the source-QA-first pattern across the series. Include the running tally table, core-space tracking results, and the "where it strains" observations.

**Strength:** Appropriate scope for the evidence. Does not overclaim.

**Risk:** May not be enough for a standalone paper. Better as a subsection in a larger methods paper.

### Option C — Appendix example set

**Content:** Package the three inventories as a structured appendix to the existing documentation. Each inventory gets a one-page summary with the decision flow and key numbers.

**Strength:** Low overhead, immediately useful for users.

**Risk:** Appendices are rarely read. The evidence deserves a more prominent home.

### Option D — Doc-page series

**Content:** Add a "Case Studies" section to the Gradience documentation with one page per inventory. Each page shows the workflow, the narrowing, and the lesson.

**Strength:** Permanent, discoverable, directly useful for Gradience users.

**Risk:** Documentation maintenance burden. Case studies become stale as the tool evolves.

## Updated recommendation (after adjudication + advisory validation)

**Blog follow-up (Option A), with the validated regime map as the main contribution.**

The story is now stronger and more honest than any earlier version:

- Wave 1 story: "source QA always dominates" — true but incomplete
- Wave 2 story: "core-space becomes the primary driver in credible cross-task pools" — now known to be overstated
- **Current story:** "the workflow's value is regime-dependent, and the most predictive signal for cross-task merge safety is task identity metadata, not spectral structure"

**Revised title concept:** "When Does Preflight Help? A Regime Map for LoRA Merge Inventories"

**Key content:**
1. The regime map (5 regimes, empirically grounded):
   - Messy pools → QA dominates
   - Same-task credible pools → workflow mostly confirmatory
   - Adjacent-task credible pools → **task advisory is the primary discriminator**
   - Distant cross-task pools → advisory clarifies and reinforces structural caution
   - Large mixed pools → neighborhoods + advisory together improve matrix readability
2. The adjudication result: 23/23 perfect discrimination by task identity vs 57-65% for spectral signals
3. The advisory validation: 46 pairs, 0 false positives, clean separation
4. Core-space honestly positioned: structurally real, behaviorally narrow, regime-dependent
5. Honest "where it strains" section

**Why this is now much stronger:**
- Backed by verified downstream merge evaluation, not just structural judgment changes
- Includes a concrete, validated additive feature (task advisory) that emerged from the evidence
- Core-space claims are tightened rather than inflated
- The regime map is grounded in 69 verified pairs across 2 adjudication studies + 5 validation inventories

**Deferred:** Paper subsection (viable but would benefit from a larger-backbone replication). Doc-page series (after blog).
