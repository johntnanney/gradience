# Over-Accumulation Refinement Decision Memo

## 1) What Was Tested
- We decomposed OA into raw subfactors, interactions, hotspot metrics, risk-concentration metrics, and optional module summaries.
- We tested rank relationships and coarse split behavior against strict-naive `merge_delta_vs_best_source`.
- We compared all refined OA candidates against current pair-level OA baselines.

## 2) What Improved (If Anything)
- Top OA candidate: `concentration_oa_mass_top3_fraction` with Spearman=0.5835 and LOO sign consistency=1.000.
- Best baseline |Spearman| was 0.2195.

## 3) What Remained Weak
- Pair-level activation remains sparse under current thresholds.
- Small cohort size limits confidence and increases outlier sensitivity.
- Directional stability of candidate features is still limited.
- The strongest OA concentration signal points in a non-risk direction in this cohort.
- Control covariates (especially source score gap) remain stronger than OA-derived features.

## 4) Decision
- **keep_exploratory**
- Evidence strength: **suggestive**

## 5) If Refine, What Exactly Changes
- Not selected in this pass.
- If selected later: promote one risk-direction-consistent hotspot/interaction feature as OA-v2 candidate.
- Keep current taxonomy unchanged and re-run strict-naive validation before policy updates.

## 5b) Chosen Path Next Steps
- Keep OA as low-confidence exploratory signal.
- Retain current diagnostic but avoid stronger prominence claims.
- Build a larger high-overlap strict-naive cohort with stronger top-tail coverage.
- Re-run this exact decomposition/interactions/hotspot analysis before any OA-v2 implementation.

## 6) If Pause, What Would Revive The Line
- Broader strict-naive high-overlap cohort with stronger top-tail score coverage.
- Cleaner matched cases with lower source-quality confounding.
- Improved theoretical framing for benign alignment vs harmful concentration interactions.
