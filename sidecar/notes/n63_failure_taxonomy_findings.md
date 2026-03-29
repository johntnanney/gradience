# n63 — Failure Taxonomy Findings

**Type:** findings
**Date:** 2026-03-28
**Depends on:** n61 (behavior findings), n62 (taxonomy protocol)
**Status:** Complete. Taxonomy is stable and usable.

---

## The taxonomy

Five categories, plus one excluded baseline. This is the final taxonomy for the Output Example Semantics program.

### A — Preserved consensus

Both sources correct, merge also correct. The safe baseline. Dominant in safe retained merges (60.6% for SR-01) and cross-task control (80.6% for CT-01, because the strong ag_news source gets most examples right and the merge preserves most of them). Low in fragile merges (5–15%) and anchor (2.2%).

### B — Consensus breakage

Both sources correct, merge wrong. The most important failure category. Negligible in safe (0.3%) and near-miss (0.7%). Elevated in fragile (1.8%), with FR-02 reaching 2.6%. Absent from the final taxonomy counts because the raw consensus-breakage examples were reclassified: some were upgraded to D (neither-source) because the merged prediction matched neither source. The category remains conceptually important even though its count is absorbed into D.

**Clarification:** The B category was absorbed into D during taxonomy construction for cases where the merged prediction is neither-source. Pure consensus breakage (both correct, merge wrong, but merge follows one source's incorrect pattern) is rare. The more common failure is consensus breakage *with* neither-source behavior.

### C — Better-source loss

One source correct, the merge fails to preserve the better source's answer. The main error type in near-miss merges (28.6–39.2%) where the merge occasionally follows the weaker source. Also present in fragile merges (5.8–6.0%) and anchor (12.4%). This is the "merge averages two rules and lands on the wrong side" pattern.

### D — Neither-source behavior

The merged model predicts something neither source predicted. Concentrates in fragile (12.4–13.4%) and control (13.6%) cases. Negligible in safe (0.8%) and near-miss (0–1.8%). This is the behavioral signature of what the mechanism ladder calls "upstream pathology transmitted through an open readout gate." The merge has produced a novel decision from a compromise representation.

### E — Benign disagreement absorption

Sources disagree (or both wrong), but the merge lands on the correct answer. This is the positive category — the merge successfully absorbs disagreement or recovers from shared errors. Dominant in SR-02 (47.4%) and the fragile cases (49–59%). The high rate in fragile cases is not paradoxical: it means the merge is correctly handling the majority of disagreement cases, even though it fails badly on the minority (consensus breakage + neither-source).

### X — Shared failure (excluded)

Neither source correct, merge also wrong. Pre-existing failure, not caused by the merge. Dominant in anchor (65.2%), moderate in safe (4–36%), negligible in control (5.8%). Excluded from the actionable taxonomy because the merge cannot be blamed for it.

---

## Taxonomy composition by class

| Class | A (preserved) | C (better-source loss) | D (neither-source) | E (benign) | X (shared, excluded) |
|-------|--------------|----------------------|-------------------|-----------|---------------------|
| Safe retained | 37.4% | 17.7% | 0.4% | 24.6% | 19.9% |
| Near-miss | 22.2% | 33.9% | 0.9% | 37.2% | 5.8% |
| Fragile | 9.8% | 5.9% | **12.9%** | 54.3% | 17.1% |
| Control | 80.6% | 0% | **13.6%** | 0% | 5.8% |
| Anchor | 2.2% | 12.4% | 5.2% | 15.0% | 65.2% |

(Class averages.)

---

## Key findings

### Finding 1: The taxonomy is stable at 5 categories

The five categories capture all observed behavioral patterns without forcing. No data-free categories were retained (B was absorbed into D because pure consensus breakage without neither-source behavior was too rare to be a distinct category in this panel). No categories needed splitting.

### Finding 2: D (neither-source) is the signature category for fragile/control merges

Neither-source behavior accounts for 12–14% of examples in fragile and control cases, versus <2% in safe and near-miss. This is the cleanest behavioral discriminator. Its concentration in these classes is consistent with the mechanism ladder: when V-module pathology is transmitted through an open readout gate, the merged representation averages into a direction that neither source learned.

### Finding 3: E (benign absorption) is unexpectedly high in fragile merges

The fragile cases show 49–59% benign absorption — meaning the merge correctly resolves most disagreements. This prevents the fragile merges from being catastrophic (overall accuracy is 0.664, not 0.25). The fragile merges are not uniformly broken — they fail selectively on consensus-correct examples and neither-source cases while correctly handling the majority of source disagreements.

### Finding 4: The taxonomy confirms near-miss ≈ safe retained

Near-miss taxonomy composition mirrors safe retained on D (neither-source): both are below 2%. The main difference is a higher rate of C (better-source loss) in near-miss (34% vs 18%), which is the expected consequence of merging a strong source with a weaker one. But this better-source loss is not structurally dangerous — it is the normal cost of imperfect averaging.

### Finding 5: The anchor case is interpretively useful

AN-01 (both sources near chance) shows that when there is no strong signal to preserve, the taxonomy is dominated by shared failure (65.2%). The merge cannot break what was never there. This validates the taxonomy's logic: merge-caused categories (B, C, D) only become meaningful when at least one source has meaningful predictive power.

---

## Relationship to spec categories

The spec proposed six candidate categories (A–F). The implemented taxonomy tracks closely:

| Spec category | Implemented as | Notes |
|--------------|----------------|-------|
| A (preserved consensus) | A | Identical |
| B (consensus breakage) | Absorbed into D | Pure consensus breakage is rare; merge-wrong-and-neither-source is more common |
| C (better-source loss) | C | Identical |
| D (ambiguity collapse) | Folded into D | Ambiguity collapse is the confidence signature of D, not a separate category |
| E (neither-source) | D | Combined with spec's D |
| F (benign disagreement absorption) | E | Identical in concept |

The consolidation of spec-B and spec-D/E into a single D is the main structural change. In practice, consensus breakage and neither-source behavior co-occur: when the merge breaks an agreed-upon example, it usually does so by producing a prediction that neither source made. Separating them would be artificial.

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This findings note | `sidecar/notes/n63_failure_taxonomy_findings.md` |
| Taxonomy JSON | `sidecar/results/example_semantics/failure_taxonomy.json` |
| Example flip catalog | `sidecar/results/example_semantics/example_flip_catalog.json` |
