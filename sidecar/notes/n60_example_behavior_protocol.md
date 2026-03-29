# n60 — Example Behavior Audit Protocol

**Type:** protocol
**Date:** 2026-03-28
**Depends on:** n59 (panel definition)
**Status:** Executed. Findings in n61.

---

## Objective

Build a compact behavioral summary for each panel case: what each source gets right, what the merge preserves, what it newly breaks, and whether failures are diffuse or structured.

---

## Method

### Data collection

For each of the 8 panel cases, re-evaluate source A, source B, and the merged adapter on the same 500-example slice used in the Phase 2b field trial. Collect:

- Per-example predictions (argmax of logits)
- Per-example softmax probabilities (full distribution)
- Gold labels

Cross-task cases (CT-01) where source B cannot be loaded with the evaluation task's num_labels are handled by marking source B as unavailable and computing metrics against source A only.

Script: `sidecar/scripts/collect_example_predictions.py`

### Per-example classification

Each example is classified into one of these behavioral categories based on the truth table of (source A correct, source B correct, merged correct):

| Source A | Source B | Merged | Category |
|----------|----------|--------|----------|
| ✓ | ✓ | ✓ | preserved_consensus |
| ✓ | ✓ | ✗ | consensus_breakage |
| ✓ | ✗ | ✓ | better_source_preserved |
| ✗ | ✓ | ✓ | better_source_preserved |
| ✓ | ✗ | ✗ | better_source_loss |
| ✗ | ✓ | ✗ | better_source_loss |
| ✗ | ✗ | ✓ | merge_recovery |
| ✗ | ✗ | ✗ | shared_failure |

An "other" category captures edge cases in multi-class settings (e.g., sources disagree, both wrong, merge also wrong but in a different way).

### Metrics

Five metrics computed per case:

1. **Source preservation rate:** Among examples correct for either source, how many remain correct after merge?
2. **Joint-source breakage rate:** Among examples correct for both sources, how many become wrong after merge?
3. **Neither-source behavior rate:** How often does the merged model's prediction match neither source's prediction?
4. **Confidence analysis:** Mean merged confidence, confidence collapse count (merged < 0.4 where source A > 0.6), high-confidence wrong count (merged wrong with confidence > 0.8).
5. **Error concentration:** Distribution of error types across behavioral categories.

Script: `sidecar/scripts/analyze_example_behavior.py`

---

## Success criteria

- All 8 cases produce valid per-example predictions: **met**
- At least one clear behavioral distinction between safe and fragile cases: **assessed in n61**
- Confidence/logit data available for all cases: **met**

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This protocol | `sidecar/notes/n60_example_behavior_protocol.md` |
| Prediction collection script | `sidecar/scripts/collect_example_predictions.py` |
| Analysis script | `sidecar/scripts/analyze_example_behavior.py` |
| Per-case predictions | `sidecar/results/example_semantics/predictions/*.json` |
| Behavior summary | `sidecar/results/example_semantics/example_behavior_summary.json` |
| Preservation/breakage table | `sidecar/results/example_semantics/preservation_breakage_table.json` |
| Per-case analyzed results | `sidecar/results/example_semantics/analyzed_*.json` |
