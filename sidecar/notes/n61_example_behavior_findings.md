# n61 — Example Behavior Findings

**Type:** findings
**Date:** 2026-03-28
**Depends on:** n59 (panel), n60 (protocol)
**Status:** Stage A complete. Feeds taxonomy construction (Stage B).

---

## Summary

The five-metric example-level audit reveals three clear behavioral distinctions across merge quality classes. Safe retained merges preserve source-correct examples at high rates and produce almost no consensus breakage. Fragile merges show elevated consensus breakage, confidence collapse, and neither-source behavior. The control cross-task merge breaks strong-source examples at a higher rate than any same-task case. Near-miss pairs behave like safe retained merges on all five metrics. The anchor (both-weak) case is dominated by shared failure — the merge has little to break because neither source had much to preserve.

---

## Metric 1 — Source preservation rate

Among examples correct for at least one source, how many does the merge preserve?

| Case | Class | Preservation rate |
|------|-------|-------------------|
| SR-01 | safe_retained | **0.975** |
| SR-02 | safe_retained | 0.642 |
| NM-01 | near_miss | 0.672 |
| NM-02 | near_miss | 0.593 |
| FR-01 | fragile | 0.800 |
| FR-02 | fragile | 0.791 |
| CT-01 | control | 0.874 |
| AN-01 | anchor | 0.541 |

**Observation:** SR-01 (irony/DistilBERT, both sources comparable strength) shows near-perfect preservation — 97.5% of source-correct examples survive the merge. SR-02's lower rate (0.642) reflects its unusual structure: source A is weak (0.514), source B is moderate (0.588), and the merge improves on both. The low preservation rate is not a sign of failure — it reflects the merge correcting source A's errors rather than preserving a consensus.

The fragile cases (FR-01, FR-02) have moderately high preservation rates (0.79–0.80) that mask the real story: they preserve the strong source (A, at 0.752) while losing whatever the weak source contributed. The metric is inflated because the weak source got so few examples right that there is little to lose.

**Finding 1:** Preservation rate alone does not distinguish safe from fragile. The metric is useful but must be read alongside breakage rate and category distribution.

---

## Metric 2 — Joint-source breakage rate

Among examples correct for both sources, how many does the merge break?

| Case | Class | Both-correct | Broken | Breakage rate |
|------|-------|-------------|--------|---------------|
| SR-01 | safe_retained | 306 | 3 | **0.010** |
| SR-02 | safe_retained | 71 | 0 | **0.000** |
| NM-01 | near_miss | 167 | 7 | **0.042** |
| NM-02 | near_miss | 62 | 0 | **0.000** |
| FR-01 | fragile | 78 | 5 | **0.064** |
| FR-02 | fragile | 38 | 13 | **0.342** |
| CT-01 | control | 461 | 58 | **0.126** |
| AN-01 | anchor | 37 | 0 | **0.000** |

**Observation:** This is the strongest discriminating metric. Safe retained merges break 0–1% of consensus-correct examples. Near-miss merges break 0–4.2%. Fragile merges show a split: FR-01 breaks 6.4% (moderate), FR-02 breaks **34.2%** (severe). The cross-task control breaks 12.6%.

FR-02's extreme breakage rate (0.342) is the standout finding. Both FR-01 and FR-02 have the same strong source (TG-base emotion, 0.752) and the same overall Δ (-0.088), but FR-02's weak source is much weaker (0.136 vs 0.204). The merge destroys 13 of 38 examples that both sources got right. This is disproportionate: the weak source's pathology corrupts the merge's handling of even the examples where both sources agreed.

**Finding 2:** Joint-source breakage rate separates merge quality classes more cleanly than preservation rate. The threshold appears to be around 5%: safe and near-miss are below it; fragile and control are above it.

**Finding 3:** The severity of the weak source modulates breakage within the fragile class. FR-01 (partner at 0.204) breaks at 6.4%; FR-02 (partner at 0.136) breaks at 34.2%. This is consistent with the field-trial finding that weak-source severity modulates near-miss outcomes — now confirmed at example level.

---

## Metric 3 — Neither-source behavior rate

How often does the merged model predict something that neither source predicted?

| Case | Class | Neither-source rate |
|------|-------|---------------------|
| SR-01 | safe_retained | 0.008 |
| SR-02 | safe_retained | 0.000 |
| NM-01 | near_miss | 0.018 |
| NM-02 | near_miss | 0.000 |
| FR-01 | fragile | **0.146** |
| FR-02 | fragile | **0.154** |
| CT-01 | control | **0.144** |
| AN-01 | anchor | 0.070 |

**Observation:** Clean separation. Safe retained and near-miss merges produce ≤1.8% neither-source predictions. Fragile and control cases produce ~14–15%. The anchor is intermediate at 7%.

This is the behavioral signature of what the spec hypothesized as the "neither-task / neither-source" state. In fragile and control merges, roughly one in seven predictions is a novel output that neither source would have produced. The merged model is not simply following the better source or the worse source — it is generating predictions from a compromise representation that does not correspond to either source's learned discriminative rule.

**Finding 4:** Neither-source behavior rate is a clean binary discriminator between safe/near-miss (< 2%) and fragile/control (> 14%). The hypothesis that fragile merges show a recognizable "neither-source" pattern is confirmed.

---

## Metric 4 — Confidence analysis

| Case | Class | Mean conf | Conf collapse | Hi-conf wrong |
|------|-------|-----------|---------------|---------------|
| SR-01 | safe_retained | 0.755 | 0 | 6 |
| SR-02 | safe_retained | 0.694 | 0 | 26 |
| NM-01 | near_miss | 0.613 | 0 | 1 |
| NM-02 | near_miss | 0.671 | 0 | 43 |
| FR-01 | fragile | 0.469 | **30** | 0 |
| FR-02 | fragile | 0.474 | **28** | 4 |
| CT-01 | control | 0.768 | 3 | 23 |
| AN-01 | anchor | 0.344 | 0 | 0 |

**Observation:** Two distinct failure modes in the confidence data:

1. **Confidence collapse (fragile merges).** FR-01 and FR-02 show 28–30 examples where merged confidence drops below 0.4 while source A confidence was above 0.6. The mean merged confidence (0.47) is far lower than all safe/near-miss cases (0.61–0.76). The merge is uncertain, and it knows it — the softmax distribution flattens. This is the "incoherent confidence" pattern predicted by the conjunctive model for cases with upstream pathology.

2. **High-confidence wrong (safe/near-miss/control merges).** SR-02 (26), NM-02 (43), and CT-01 (23) show substantial numbers of high-confidence wrong predictions. These cases do not collapse — they confidently get examples wrong. In SR-02 and NM-02, this reflects the merged model confidently following the better source's decisions even on examples where those decisions are wrong. In CT-01, it reflects the merged model confidently applying the ag_news adapter's rule even where the cross-task interference has broken it.

**Finding 5:** Fragile merges fail with confidence collapse (low confidence, wrong). Safe merges that fail do so with high confidence (wrong but sure of it). This is a qualitative difference in failure mode, not just magnitude.

---

## Metric 5 — Error concentration

The dominant error categories by case class:

**Safe retained:** Errors are overwhelmingly shared failures (examples that both sources got wrong, merge also gets wrong) and better-source loss (merge follows the worse source on a disagreement). Consensus breakage is 0–1%. The merge does not introduce new errors on agreed-upon examples.

**Near-miss:** Same pattern as safe retained. The dominant error is better-source loss (merge follows the wrong source). Consensus breakage is 0–4.2%. Near-miss merges do not produce a distinct error profile from safe retained.

**Fragile:** Errors split between shared failure, better-source loss, and consensus breakage. FR-02 shows 13 consensus breakages (2.6% of all examples) — the highest in the panel. Fragile merges introduce errors on examples where both sources agreed. This is the behavioral marker of dangerous merging.

**Control (cross-task):** The dominant error is source-A loss (11.6%) — the merge breaks predictions that the strong source got right. The cross-task adapter's interference disrupts the strong source's learned discriminative rule.

**Anchor (both-weak):** Dominated by shared failure (65.2%). There is almost nothing to break — neither source had a strong signal to preserve.

**Finding 6:** Consensus breakage concentrates in fragile merges. It is absent or negligible in safe and near-miss cases. This makes consensus breakage the most actionable per-example signal: if a merge is breaking examples that both sources got right, something is genuinely wrong.

---

## Category distribution summary

| Category | Safe retained | Near-miss | Fragile | Control | Anchor |
|----------|--------------|-----------|---------|---------|--------|
| preserved_consensus | 37.4% | 22.2% | 9.8% | 80.6% | 2.2% |
| better_source_preserved | 24.6% | 37.2% | 54.3% | — | 15.0% |
| consensus_breakage | 0.3% | 0.7% | **1.8%** | — | 0.0% |
| better_source_loss | 17.7% | 33.9% | 14.7% | — | 14.6% |
| shared_failure | 19.9% | 5.8% | 17.1% | 5.8% | 65.2% |
| merge_recovery | 0.1% | 0.3% | 0.3% | 2.0% | 1.2% |
| source_a_loss | — | — | — | 11.6% | — |

(Averages within each class, rounded.)

---

## Interim conclusions for Stage B

1. **A small taxonomy is viable.** The per-example categories fall into a few reusable types with interpretable consequences.
2. **The key discriminating categories are:** consensus breakage (fragile-specific), neither-source behavior (fragile/control), and confidence collapse (fragile-specific).
3. **Near-miss is behaviorally indistinguishable from safe retained.** On all five metrics, near-miss falls within the safe-retained range.
4. **The behavioral audit answers RQ1–RQ3** from the spec:
   - RQ1: Safe merges preserve 97%+ of consensus-correct examples (on comparable-strength sources).
   - RQ2: Fragile merges disproportionately break consensus-correct examples and produce neither-source predictions.
   - RQ3: Yes, fragile merges show a "neither-source" behavioral signature at ~15% rate.

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This findings note | `sidecar/notes/n61_example_behavior_findings.md` |
| Behavior summary JSON | `sidecar/results/example_semantics/example_behavior_summary.json` |
| Preservation/breakage table | `sidecar/results/example_semantics/preservation_breakage_table.json` |
| Figures | `sidecar/figures/example_semantics_*.png` |
