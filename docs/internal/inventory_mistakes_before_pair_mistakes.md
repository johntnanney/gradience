# Inventory Mistakes Before Pair Mistakes

Central reference for this line of investigation.

## The observation

In every case study so far, the most important narrowing happened at the inventory level — source QA, adapter credibility, eligibility stratification — before pairwise spectral analysis became the main decision driver.

The phrase "inventory mistakes before pair mistakes" captures this: a practitioner who skips source screening and jumps straight to pairwise merge-risk analysis will waste time on pairs that should never have been considered.

## Questions tracked per inventory

### Q1 — How often does source QA remove the wrong problem before pairwise analysis matters?

Did excluding weak or low-credibility sources simplify the inventory materially? Did this happen before pairwise analysis produced the main insight?

### Q2 — How often do neighborhoods materially reduce the candidate space?

Did neighborhoods change the practical next step? Did they collapse a flat pool into a smaller structured plan?

### Q3 — How often does the inventory-level view change action more than pairwise scores alone?

Without neighborhoods/inventory view, what would the likely next step have been? With them, what became the actual next step?

---

## Evidence log

### case_study_qnli4_realmix_20260318 (first published case study)

**Q1:** Yes. Source QA excluded `qnli_uniform_weak` (flagged_weak) immediately, removing it from 3 of 6 pairs. This was the single most impactful narrowing step.

**Q2:** Modestly. Neighborhoods identified one local exploration target, but with 4 adapters the structure was not dramatically more informative than the pair matrix.

**Q3:** Yes. Without the inventory view, the practitioner would have explored all 6 pairs. With it, the action space narrowed to one neighborhood-first plan with one cautioned pair.

### roberta_mixed_evidence_4 (series wave 1, Category 1)

**Q1:** Yes. Source QA stratified 3 adapters into eligible / uncertain / unknown. Only 1 adapter has credible behavioral evidence. Strict-QA collapses to 1 surviving adapter. The pool was identified as thin before any pair analysis ran.

**Q2:** Modestly. Neighborhoods confirmed per_layer_unknown's isolation (caution singleton) and grouped the other two as likely-safe. With 3 adapters, neighborhoods added confirmation but not new structure.

**Q3:** Yes. Without the inventory view, a practitioner sees 1 low-risk pair and might merge. With it, core-space demoted that pair to marginal, and the overall conclusion shifted from "merge the safe pair" to "the inventory is too thin — recollect or proceed with extreme caution."

### distilbert_large_pool_6 (series wave 1, Category 2)

**Q1:** Yes — strongest case yet. 4 of 6 adapters have no behavioral evaluation. Strict-QA blocks 14 of 15 pair reports. Source QA alone reduced a 15-pair inventory to 1 defensible pair. This happened entirely before pairwise analysis mattered.

**Q2:** Yes — strongest case yet. Neighborhoods compressed 15 pairs into 5 groups (4 caution singletons + 1 likely-safe). The likely-safe neighborhood ({qnli_probe, qnli_uniform}) was the key takeaway. At 6 adapters, the pair matrix is genuinely hard to parse manually; neighborhoods made the answer obvious.

**Q3:** Yes. Without the inventory view, a practitioner sees 6 low-risk pairs and tries multiple cross-group merges. With neighborhoods showing the QNLI pair as the only safe neighborhood, and core-space showing a cross-group "safe" pair is actually incompatible, the action space collapsed from 6 candidate merges to 1.

---

## Running tally

| Question | case_study_qnli4 | roberta_mixed_evidence_4 | distilbert_large_pool_6 |
|----------|-------------------|--------------------------|-------------------------|
| Q1: Source QA did early narrowing? | Yes (excluded 1 weak) | Yes (stratified 3 statuses) | **Yes (blocked 14/15 pairs)** |
| Q2: Neighborhoods reduced space? | Modestly | Modestly | **Yes (5 groups, 1 safe)** |
| Q3: Inventory view changed action? | Yes | Yes | **Yes (6 candidates → 1)** |

### all_eligible_qnli_control (wave 2, Target 5 — low-drama control)

**Q1:** No. All eligible, QA narrows nothing.
**Q2:** No. One group, no compression.
**Q3:** No. Same conclusion from pair reports directly.

### behaviorally_complete_5 (wave 2, Target 1)

**Q1:** Partially. QA excluded 1 weak adapter (40% pair reduction) but didn't dominate. Core-space was the main decision driver for the surviving pool.
**Q2:** No. One group containing all eligible adapters — neighborhoods uninformative.
**Q3:** Yes. Core-space demoted all 3 cross-task low-risk pairs. Without it, 3 unsafe merges would look safe.

### core_space_hunt_4 (wave 2, Target 4)

**Q1:** No. All eligible, QA narrows nothing.
**Q2:** No. One caution group.
**Q3:** Yes — dramatically. Full census showed all 6 pairs incompatible/marginal. Reframed the entire pool.

### two_cluster_neighborhood_7 (wave 2, Target 2)

**Q1:** Yes. Strict-QA blocks 11/21 pairs (2 unknown adapters).
**Q2:** Yes — genuine compression. 21 pairs → 3 groups + 3 boundary warnings.
**Q3:** Partially. Neighborhoods compressed but the grouping was QA-driven, not structural.

### messy_heterogeneous_6 (wave 2, Target 3)

**Q1:** Yes. Strict-QA blocks 12/15 pairs. Source QA dominates.
**Q2:** Modestly. 3 groups but fragmented (2 singletons).
**Q3:** No. Same conclusion reachable from QA status counts alone.

---

## Updated running tally (8 inventories)

| Question | qnli4 | roberta | distilbert6 | T5 ctrl | T1 behav | T4 hunt | T2 neigh | T3 messy |
|----------|--------|---------|-------------|---------|----------|---------|----------|----------|
| Q1: QA early narrow? | Yes | Yes | **Yes** | No | Partial | No | Yes | **Yes** |
| Q2: Neighborhoods? | Modest | Modest | **Yes** | No | No | No | **Yes** | Modest |
| Q3: View changed? | Yes | Yes | **Yes** | No | **Yes** | **Yes** | Partial | No |

## Updated pattern (after wave 2)

Source QA dominance is regime-dependent:
- **Messy pools (unknown/weak adapters):** QA dominates (4/4 inventories)
- **Behaviorally complete pools:** QA still helps but doesn't dominate (1/3)
- **All-eligible pools:** QA does nothing (2/2)

The phrase "inventory mistakes before pair mistakes" holds for messy pools but weakens for credible pools. In credible pools, the main "inventory mistake" is not about source quality — it's about structural incompatibility invisible to layer-level analysis. Core-space catches this, not QA.

Neighborhoods scale with inventory size and add operational compression at 6+ adapters. But their grouping reflects QA status, not structural similarity.
