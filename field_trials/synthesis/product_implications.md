# Product Implications — CPU Field Research Protocol

**Date:** 2026-03-29
**Based on:** Campaigns A + B, Phase 2/2b reference data

---

## 1. Immediate Product Changes Worth Making

### 1a. Task-Family Taxonomy (from Campaign A)

**What:** Add a static registry mapping known dataset names to task-family labels. When both adapters in a pair belong to the same family, downgrade the cross-task advisory from "caution" to "informational."

**Why:** SST-2 x IMDB merges behave identically to same-task retained pairs (avg Δ -0.022 vs -0.017). The current strict boundary misclassifies these as equivalent in risk to genuine cross-task merges (avg Δ -0.047).

**Design notes:**
- Start with binary sentiment (SST-2, IMDB, Yelp, Amazon), topic classification (AG News, DBpedia), NLI (MNLI, SNLI, RTE), emotion/affect (tweet_eval/emotion, GoEmotions)
- Static registry, not learned classifier — inspectable, overridable, versionable
- Unknown datasets remain in singleton families (conservative default)
- Advisory text: "Different datasets but same task family — similar tasks tend to merge well"
- The advisory is never suppressed entirely, only modulated

**Implementation scope:** Small. Add a `TASK_FAMILY_REGISTRY` dict to `vnext/merge/` or `vnext/inventory/`, add family lookup to the advisory logic, update advisory text templates.

**Validation before release:** Replicate on at least one additional task family (e.g., NLI) on a second backbone.

### 1b. Near-Miss Severity Ranking (from Campaign B)

**What:** Within the near-miss section of preflight output, rank pairs by the weak source's proximity to the eligibility threshold. Barely-weak pairs (source delta -0.002 to -0.010) appear first; deeply-weak pairs (delta below -0.050) appear last.

**Why:** Barely-weak near-miss pairs show avg Δ -0.007 (essentially harmless), while deeply-weak pairs show avg Δ -0.045 (genuine risk). Users benefit from knowing which near-miss pairs are most worth evaluating.

**Design notes:**
- Add `weakness_severity` field to QA artifact: `marginal` (delta > -0.010), `moderate` (-0.010 to -0.050), `substantial` (< -0.050)
- Sort near-miss section by severity (marginal first)
- Advisory text: "Source is marginally below base — structurally plausible merge" vs "Source substantially underperforms base — merge risk elevated"

**Implementation scope:** Small. Add severity classification in `qa_artifact.py`, update near-miss sorting in inventory summary and action plan.

## 2. Things Confirmed Good Enough Already

### 2a. Evidence Gate at delta < 0

The evidence gate correctly excludes adapters that underperform base. Barely-weak adapters near the boundary are safe merge partners (Campaign B), but the gate's conservatism is appropriate: excluding them costs little (they appear in near-miss), and the alternative — lowering the threshold — would admit genuinely weak adapters.

**No change needed.** The gate is correct. The near-miss ranking (1b above) provides the user-facing resolution.

### 2b. Cross-Task Advisory for Genuinely Different Tasks

Phase 2 cross-task controls (sentiment x AG News, AG News x emotion, SST-2 x AG News) consistently show avg Δ around -0.047 to -0.096. The cross-task advisory is correct for these cases.

**No change needed.** The task-family taxonomy (1a above) modulates the advisory for same-family pairs without weakening it for genuine cross-task merges.

### 2c. Task-Boundary Detection Zero False Positives

Across 5 inventories, 53+ pairs, Gradience has zero false positives on task-boundary detection. Every flagged cross-task pair was indeed cross-task.

**Confirmed.** Detection accuracy is the product's strongest empirical claim.

### 2d. Candidate Reduction Ratio (90-93%)

Consistent across all field trials. Gradience reliably reduces the candidate space by 90-93%.

**Confirmed.** This is stable and reliable.

## 3. Things That Should Wait for More Evidence

### 3a. Task-Family Generalization Beyond Sentiment

Campaign A tested only binary sentiment (SST-2 x IMDB). Whether the finding extends to NLI variants, multi-class topic classification variants, or more distant task families is unknown. The task-family taxonomy should be released with appropriate caveats and validated incrementally.

**What evidence would look like:** 2-3 NLI adapters (MNLI, SNLI, RTE) on a common backbone; merge evaluation showing same-family NLI pairs behave like retained.

### 3b. Weakness Severity Boundary Location

The gap between barely-weak (-0.007) and deeply-weak (-0.045) is clear at the extremes, but the intermediate range (source delta -0.010 to -0.050) is unrepresented. The severity labels (marginal/moderate/substantial) are currently based on reasonable cutpoints, not empirically validated thresholds.

**What evidence would look like:** 4-6 adapters with source deltas spanning -0.005 to -0.050, merged with eligible partners, showing where degradation transitions from harmless to meaningful.

### 3c. Large-Inventory Ergonomics (Campaign C)

Current validated ceiling is 28 pairs / 8 adapters. Whether HTML reports, action plans, and region summaries remain useful at 66+ pairs (12+ adapters) is an open usability question.

**What evidence would look like:** Qualitative assessment of a 12-adapter inventory. No merge evaluation needed — this is about output readability.

### 3d. Ecosystem Robustness (Campaign D)

TransferGraph adapters exercise some edge cases (transfer-chain bases, unusual configs), but a systematic test of graceful failure on broken/weird adapters would strengthen confidence.

**What evidence would look like:** 6-8 deliberately messy adapters; load success rate, error message quality, honest uncertainty reporting.

### 3e. DeBERTa Adjudication (GPU-blocked)

The decisive test for the causal mechanism behind cross-task interference. This is the sidecar research question, not a product question, but it would strengthen the theoretical foundation for all product decisions.

**What evidence would look like:** 8 DeBERTa adapters, 28 pairs, 5 pre-registered predictions. Requires ~3h GPU compute.

---

## Summary Priority

| Item | Type | Effort | Impact |
|------|------|--------|--------|
| Task-family taxonomy | Ship | Small | High — reduces false caution for common sentiment pools |
| Near-miss severity ranking | Ship | Small | Medium — helps users prioritize near-miss investigation |
| Evidence gate | Confirm, no change | — | — |
| Cross-task advisory | Confirm, no change | — | — |
| Task-family generalization | Validate | Medium | Deferred — ship with caveats, validate incrementally |
| Weakness boundary | Validate | Medium | Deferred — current cutpoints are reasonable |
| Scale ergonomics | Deferred | Low | Low priority until a user hits the ceiling |
| Ecosystem robustness | Deferred | Low | Nice-to-have operational hardening |
| DeBERTa adjudication | GPU-blocked | Medium | Research, not product |
