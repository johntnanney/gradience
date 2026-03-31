# Field Note — Targeted Confirmation T01: Same-Family Routing

Date: 2026-03-29

## 1. Inventory

**Backbone**: distilbert-base-uncased
**Adapters**: 4

| Adapter | Dataset | Delta | Status |
|---------|---------|-------|--------|
| myselfmankar/distilbert-base-sst2-lora | sst2 | +0.350 | eligible |
| rambodazimi/distilbert-base-uncased-finetuned-LoRA-SST2 | sst2 | +0.368 | eligible |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | +0.334 | eligible |
| TransferGraph/JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news | ag_news | +0.366 | eligible |

**Why this inventory**: Reuses Campaign A adapters (SST-2 and IMDB, known sentiment family) plus one ag_news adapter as a true cross-task control. All adapters are strong (delta > +0.33). Provides clean three-way contrast: exact same-task, same-family cross-dataset, true cross-task.

## 2. Gradience Stance

**Retained (same-task safe zone)**: 2 pairs
- myselfmankar × rambodazimi (exact same-task, medium risk, audit_aware)
- myselfmankar × dipanjanS (same-family, medium risk, audit_aware)

**Cross-task caution zone**: 1 pair
- myselfmankar × JB173 ag_news (cross-task, AG_NEWS × SST-2 region)

**Excluded / near-miss**: none

**Routing details**:
- Exact same-task pair received `task_relationship=same_task`, routed to same-task safe zone.
- Same-family pair received `task_relationship=same_family`, routed to same-task safe zone (not cross-task caution). Report headline: "Same-family pair — plausible candidate." Advisory: "TASK-FAMILY NOTE" (informational) with "Treat this pair like a same-task candidate."
- Cross-task pair received `task_relationship=cross_task`, routed to cross-task caution zone. Report headline: "Cross-task pair — caution region." Advisory: "TASK-BOUNDARY WARNING" (caution).

## 3. Tiny Evaluation Results

All merges evaluated on SST-2 (500 samples, shuffled seed=42).

| Pair | Category | Merged Acc | Best Source | Delta vs Best |
|------|----------|-----------|-------------|---------------|
| myselfmankar × rambodazimi | retained (same-task) | 0.876 | 0.884 | -0.008 |
| myselfmankar × dipanjanS | same_family | 0.878 | 0.884 | -0.006 |
| myselfmankar × JB173 ag_news | cross_task_control | 0.842 | 0.884 | -0.042 |

## 4. Product Judgment

### Did the new routing behave sensibly?

Yes. The three-way routing produced exactly the expected separation:

1. The exact same-task pair went to the safe zone (correct — baseline behavior).
2. The same-family pair also went to the safe zone, but with a "TASK-FAMILY NOTE" informational advisory rather than a caution warning (correct — same family should not be treated like cross-task).
3. The true cross-task pair went to the caution zone with a "TASK-BOUNDARY WARNING" (correct — ag_news is genuinely different from SST-2).

The routing is visibly distinct across all three categories.

### Did the ordering feel useful?

The action plan correctly listed the same-family pair as an "evaluate first" candidate alongside the retained pair. A reader would immediately see that these are the two best options. The cross-task pair was clearly separated into the caution zone.

### Did the report explain it clearly?

Mostly yes, with one fix applied during confirmation:

**Issue found and fixed**: The original headline for same-family pairs read "Cross-task pair — caution region" even though the advisory correctly identified the pair as same-family. This was misleading — the headline contradicted the advisory text. Fixed: headline now reads "Same-family pair — plausible candidate" when `task_relationship == "same_family"`.

After the fix, a reader can understand:
- **Same-task** is still preferred (headline: "Same-task pair — safe region").
- **Same-family** is plausible but different (headline: "Same-family pair — plausible candidate"; advisory is informational not cautionary).
- **Cross-task** remains clearly different (headline: "Cross-task pair — caution region"; advisory is a warning).

### Outcome sanity

The same-family pair (0.878) performed essentially identically to the retained pair (0.876) — actually marginally better. The cross-task pair (0.842) was meaningfully worse (delta −0.042 vs best source, compared to −0.006/−0.008 for the other two). This strongly confirms the routing logic: same-family pairs behave like same-task, not like cross-task.
