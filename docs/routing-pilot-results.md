# Routing Pilot Results: Policy Vocabulary and Aggregation Strategy as the Two Extraction Seams

**Date:** 2026-03-29
**Status:** Canonical result — documents the substrate generality validation

**Context update (2026-03-30):** This pilot is now one axis in a three-axis generalization story: scenario generality (this document), low-rank PEFT artifact-class generality (Ring 1), and summary-based full-checkpoint-delta generality (Ring 2). See `docs/architecture-assessment.md` and `docs/design/ring2_stage_d_assessment_memo.md`.

---

## What happened

The routing pilot (`experiments/routing_pilot/`) consumed Gradience's existing spectral analysis pipeline to assess adapter routing confusability instead of merge compatibility. Four distilbert-base-uncased LoRA adapters (r=16, three NLI tasks, two seeds for RTE) were compared across all six pairs. The pilot produced a fleet-level confusability report that discriminated same-task pairs (high), same-family pairs (moderate), and cross-task pairs (low).

Total new code: ~370 lines across three files. Execution time: 0.2s on CPU. Zero modifications to any existing Gradience module.

---

## What was reused

Five functions from the merge package's public API, called identically to how merge uses them:

| Function | Module | Purpose |
|----------|--------|---------|
| `load_adapter` | `vnext/merge/io.py` | Load PEFT directory → `AdapterInfo` |
| `match_layers` | `vnext/merge/io.py` | Find shared module prefixes between two adapters |
| `extract_factors` | `vnext/merge/io.py` | Extract `(A, B, rank)` tuples per layer |
| `compute_subspace_metrics` | `vnext/merge/spectral_compat.py` | Per-layer spectral comparison → `SubspaceMetrics` |
| `assess_layer` | `vnext/merge/verdicts.py` | Per-layer verdict (used only for comparison, not for routing path) |

The routing layer consumed `SubspaceMetrics` directly. It did not need `assess_layer`, `diagnose_pair`, or any merge-specific verdict logic for its own assessment path. The only reason `assess_layer` was called was to produce a side-by-side merge-vs-routing comparison.

---

## What new code was added

Three files in `experiments/routing_pilot/`:

**`routing_compat.py`** (~180 lines) — routing-specific interpretation of `SubspaceMetrics`. Defines `RoutingLayerAssessment` and `RoutingPairAssessment`. The key inversion: high overlap + aligned = "high confusability" (merge calls this "redundant"), low overlap = "low confusability" (merge calls this "safe"). Magnitude imbalance is irrelevant for routing — a router doesn't need scale-balanced adapters, only separable subspaces.

**`routing_report.py`** (~120 lines) — fleet-level report builder. Aggregates pair assessments into a `RoutingFleetReport` with confusable/caution/clean pair lists.

**`run_routing_pilot.py`** (~170 lines) — end-to-end script. Loads adapters, computes all-pairs metrics, produces both routing assessments and merge verdicts for comparison, outputs JSON and text reports.

---

## Why aggregation is the real differentiator

The original architecture assessment predicted the extraction seam at the diagnosis/policy boundary: same spectral data, different policy interpretation. The pilot confirmed this but revealed a second, more consequential seam: aggregation strategy.

Merge's `assess_overall` uses worst-case aggregation — if any layer is classified "redundant," the pair is "redundant." This is correct for merge: one problematic layer can poison the merged model. The result: all six NLI adapter pairs were classified "redundant," with no discrimination between same-task and cross-task pairs.

The routing layer uses distributional aggregation — mean confusability score plus fraction of high-confusability layers. This is correct for routing: a few confusable layers don't necessarily dominate the overall routing picture. The result: a three-tier discrimination that matches intuition — same-task pairs are highly confusable, same-family pairs are moderately confusable, cross-task pairs are easily routed.

The divergence table:

| Pair | Merge verdict | Routing verdict | Diverge? |
|------|--------------|----------------|----------|
| rte_s42 × rte_s7 | redundant | high | No — both flag it |
| mnli_s7 × qnli_s7 | redundant | moderate | Yes |
| mnli_s7 × rte_s42 | redundant | moderate | Yes |
| mnli_s7 × rte_s7 | redundant | moderate | Yes |
| qnli_s7 × rte_s42 | redundant | low | Yes |
| qnli_s7 × rte_s7 | redundant | low | Yes |

Five of six pairs show divergent operational guidance from identical spectral data. The divergence is not about different labels for the same conclusion — it's about different aggregation strategies revealing different truths about the same geometric measurements.

---

## What this implies for substrate generality

The extractable architecture is not a two-layer split (diagnosis vs. policy). It is four layers:

1. **Measurement** — per-adapter and per-pair spectral computation. Fully general. No scenario-specific logic.
2. **Diagnosis** — classification of measurements into named conditions. Mostly general. Optional for non-merge consumers (the routing pilot bypassed it entirely).
3. **Aggregation** — how per-layer assessments combine into pair-level or fleet-level pictures. Scenario-specific. The first real parameterization point.
4. **Policy** — scenario-specific decisions and vocabulary. The second parameterization point.

This is a richer model than the original assessment proposed, and it emerged empirically rather than being designed in advance. The codebase already has this structure — it just didn't have a name for it until the routing pilot exercised the aggregation seam.

---

## What this does not imply

This result does not mean Gradience should immediately become a multi-scenario platform. One successful non-merge pilot is evidence of substrate generality, not a product mandate. The smart sequence is:

1. Document the result (this memo).
2. Update the architecture assessment with the four-layer model (done — §7 revised, §8 added).
3. Decide deliberately whether extraction is worth the effort, based on whether a real (not experimental) non-merge use case materializes.

No refactoring, no new abstractions, no product reframing. The substrate is broader than the product. That is now a demonstrated fact. What to do about it is a separate decision.

---

*Full pilot data: `experiments/routing_pilot/routing_report.json`, `routing_report.txt`, `routing_pilot_field_note.md`.*
