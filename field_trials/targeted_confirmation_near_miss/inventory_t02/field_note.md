# Field Note — Targeted Confirmation T02: Near-Miss Severity Ordering

Date: 2026-03-29

## 1. Inventory

**Backbone**: distilbert-base-uncased (derivatives via TransferGraph)
**Adapters**: 5 (all r=1 LoRA, q_lin+v_lin)

| Adapter | Dataset | Delta | Status |
|---------|---------|-------|--------|
| JB173_irony | tweet_eval/irony | +0.012 | eligible |
| neibla_irony | tweet_eval/irony | +0.202 | eligible |
| phailyoor_irony | tweet_eval/irony | +0.218 | eligible |
| jaesun_hate | tweet_eval/hate | +0.006 | eligible |
| Aureliano_hate | tweet_eval/hate | -0.112 | flagged_weak |

**Why this inventory**: Mixed-task inventory (irony + hate) on distilbert-base-uncased derivatives. Designed to produce retained, near-miss, and excluded pairs with different severity levels. Prior campaign data (inventory_04) showed phailyoor_irony with delta = -0.004 (marginal near-miss), but re-bootstrapping with shuffled sampling (seed=42) shifted it to +0.218 (eligible). This reduced the near-miss set to a single pair at one severity level.

**Important caveat**: All adapters are r=1 LoRA. The perturbation is extremely small. This limits the discriminative power of merge evaluations — merged outputs are dominated by the base model and classifier head, not by the LoRA delta.

## 2. Gradience Stance

**Retained (same-task safe zone)**: 2 pairs
- JB173_irony × neibla_irony (medium risk, norm_equalized)
- JB173_irony × phailyoor_irony (low risk, linear) [note: phailyoor shifted to eligible on re-bootstrap]

**Near-miss candidates**: 1 pair
- jaesun_hate × Aureliano_hate (low risk, linear — Aureliano is evidence-constrained; **deeply weak**)

**Near-miss severity ordering**: The near-miss section correctly displays severity label "deeply weak" for the single near-miss pair. The ordering header states "Ordered by weak-source severity (best prospects first)." With only one near-miss pair, ordering is trivially correct.

**Excluded**: Aureliano_hate (weak source — low confidence)

**Excluded pair**: JB173_irony × Aureliano_hate (cross-task + weak source → excluded entirely, not near-miss)

## 3. Tiny Evaluation Results

| Pair | Category | Eval On | Merged Acc | Best Source | Delta vs Best |
|------|----------|---------|-----------|-------------|---------------|
| JB173 × neibla | retained | irony | 0.606 | 0.616 | -0.010 |
| JB173 × phailyoor | retained (was planned marginal) | irony | 0.606 | 0.616 | -0.010 |
| jaesun × Aureliano | near_miss substantial | hate | 0.606 | 0.540 | +0.066 |
| JB173 × Aureliano | excluded control | irony | 0.604 | 0.614 | -0.010 |

**Note on flat outcomes**: All four merges produce nearly identical accuracy (0.604–0.606). This is a direct consequence of r=1 LoRA adapters: the merged delta AB is so small that the output is dominated by the base model and classifier head initialization. The merges are structurally sound (zero reconstruction error in 3/4 cases) but produce negligible behavioral variation. This means outcome alignment cannot be assessed for this inventory.

## 4. Product Judgment

### Did the near-miss severity ordering feel useful in practice?

**Mechanism: yes. Discrimination: not testable with this inventory.**

The near-miss section in the action plan works as designed:
- It clearly separates near-miss from retained and excluded.
- The severity label ("deeply weak") is visible and unambiguous.
- The ordering header ("Ordered by weak-source severity, best prospects first") communicates the principle.
- If evaluation budget allowed only one optional pair, a reader would immediately identify jaesun × Aureliano as the only candidate.

However, the confirmation could only produce ONE near-miss pair (substantial severity). The planned marginal near-miss (phailyoor_irony) shifted to eligible on re-bootstrap, removing the severity contrast. This means I could not test whether the ordering between marginal and substantial near-miss pairs is useful in practice.

### Did the labels make sense?

Yes. The label "deeply weak" for Aureliano (delta = -0.112) is appropriate. The rendering "TransferGraph__Aureliano_hate is evidence-constrained; deeply weak" clearly communicates both the reason (evidence-constrained) and the severity (deeply weak).

### Decision usefulness

With only one near-miss pair, the ordering is trivially informative — the reader sees exactly one optional candidate. The structural presentation is good: the near-miss section is visually distinct from both the retained safe zone and the excluded list. A user would immediately understand this is optional.

### Outcome alignment

**Not assessable.** All merged accuracies are within 0.002 of each other (0.604–0.606). This is expected for r=1 adapters where the LoRA perturbation is negligible relative to the base model. The near-miss substantial pair (0.606) actually marginally outperforms the retained pair (0.606, same) and the excluded control (0.604), but these differences are within noise.

### Observations on re-bootstrap shift

phailyoor_irony shifted from delta = -0.004 (inventory_04, prior campaign) to +0.218 (this run). The reason: irony detection on distilbert-base-uncased is at chance level (~50%), so the base score is highly variable across different sample subsets (0.398 here vs 0.622 in the prior run). This variance completely flips the "barely weak" classification. This is itself a useful finding: the marginal band (delta > -0.010) is sensitive to sampling noise for tasks where the base model is near chance. The severity ordering feature correctly handles this — adapters that shift above zero are simply classified as eligible and exit the near-miss pool.

### Summary verdict

The near-miss severity mechanism works correctly in all structural respects: labels, ordering, section layout, and decision guidance. The confirmation is **mixed** because outcome differences were too flat (r=1 limitation) and the marginal severity level could not be naturally produced (sampling variance limitation). Neither limitation reflects a flaw in the feature itself.
