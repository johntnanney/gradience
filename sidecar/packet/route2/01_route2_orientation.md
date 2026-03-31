# Route 2 Orientation — Broadened Substrate Scope

**Date:** March 2026
**Purpose:** Quick orientation for a new reader. What was tested, what generalized, what didn't.

---

## The starting point

Gradience's stable product analyzes LoRA adapters for merge compatibility. It uses spectral measurement (SVD-based), pairwise comparison, and inventory-level aggregation. Field trials confirmed: zero false positives, 90–93% candidate reduction, near-miss as a validated middle category.

Route 2 asked: does this substrate extend beyond its original scope?

---

## Three generalization axes

Each was tested by one experiment. Each produced a clear answer.

### Axis 1: Scenario generality (merge to routing)

**Experiment:** Routing pilot (`experiments/routing_pilot/`). ~370 lines of new code, 3 files. Reused 5 functions from the merge package. Zero modifications to existing code.

**Result:** The spectral substrate discriminates same-task / same-family / cross-task for routing confusability, not just merge compatibility. The structural measurement layer is shared. The divergence point is policy — what you do with the measurements depends on whether you are merging or routing.

**Seam identified:** Policy vocabulary and aggregation strategy are the two extraction points. Everything below aggregation is shared.

### Axis 2: Artifact-class generality (LoRA to LoHa)

**Experiment:** Ring 1. ~160-line extraction shim (`loha_shim.py`) that materializes LoHa's Hadamard-product factors into the (A, B) format that all downstream functions expect.

**Result:** The full pipeline — audit, merge-audit, inventory, action plan — ran on LoHa adapters with zero core code changes. Every metric, verdict, and report was produced normally. Same-task pairs were correctly identified as safe.

**What generalized:** Everything downstream of factor extraction. Spectral measurement, subspace comparison, verdicts, reports, inventory pipeline.

**What stayed LoRA-specific:** Factor extraction itself. The shim translates representation; it does not eliminate the representation boundary.

### Axis 3: Representation-path generality (LoRA to checkpoint deltas)

**Experiment:** Ring 2 (`experiments/ring2_checkpoint_delta/`). Four stages tested three candidate representations for full fine-tune checkpoint deltas (not pre-factored).

**Result:** Low-rank SVD approximation (turning deltas into synthetic LoRA factors) was rejected at CPU-feasible ranks — too much information loss. The selected representation (Representation C) uses layerwise summary statistics: norms, spectral properties, effective rank. This produces a different representation path but the same workflow shape.

**What generalized:** Evidence bootstrap, QA artifact schema, inventory triage, action plan, report vocabulary. The workflow survives.

**What did not generalize:** Factor-level subspace geometry. Merge execution. Structural metrics that require (A, B) factors. The representation path is fundamentally different — summary-based, not factor-based.

**Implication:** Checkpoint delta triage is viable. Checkpoint delta merge execution remains out of scope.

---

## The three-axis summary

| Axis | Experiment | New code | Core changes | Representation path |
|------|-----------|----------|-------------|---------------------|
| Scenario (merge → routing) | Routing pilot | ~370 lines | Zero | Same (factor-based) |
| Artifact class (LoRA → LoHa) | Ring 1 | ~160 lines | Zero | Same (shimmed) |
| Artifact class (LoRA → checkpoint delta) | Ring 2 | ~8 scripts | Zero | Different (summary-based) |

## Stability checks (what became sturdier)

Route 2 now includes two local robustness passes on top of the core programs:

- Cross-artifact stability check (`docs/strategy/cross_artifact_stability_summary.md`)
- Aggregation stability check (`docs/strategy/aggregation_stability_summary.md`)
- Aggregation mixed-evidence triage perturbation (`docs/strategy/aggregation_mixed_evidence_summary.md`)

Together, the aggregation follow-up passes strengthen four key claims:

1. aggregation is a real decision seam,
2. QA-dominant logic is a distinct operational family,
3. worst-case collapse of routing gradation is a stable pattern,
4. the triage soft middle remains structured under mixed-evidence weighting, with explicit guardrails on fine-grained thresholds.

Current Route 2 aggregation posture:

- stable at the seam level,
- guarded at the threshold level,
- behaviorally grounded at the profile level.

---

## What this means for the product

The stable Gradience product is a LoRA merge tool. Route 2 shows that the substrate underneath it is more general than its current packaging:

- **The measurement layer** generalizes to any factorized low-rank adapter (LoRA, LoHa, and likely others) and, in summary form, to full-checkpoint deltas.
- **The triage workflow** (evidence bootstrap → QA → pairwise comparison → inventory action plan) generalizes to routing and triage scenarios, not just merge.
- **The report vocabulary** (pair risk, dominant issue, recommended strategy, action plan zones) is meaningful across artifact classes and scenarios.

What does *not* generalize:
- **Structural metrics** (V-module dim ratio, subspace overlap) require factor-level data. They do not apply to summary-based representations.
- **Merge execution** is LoRA-specific. Checkpoint delta merging is a different problem that Route 2 does not address.
- **Numeric thresholds** are calibrated on the current evidence base (two backbones, classification tasks, small encoders). They are not portable to new regimes without validation.

---

## The alpha workflow

The checkpoint triage alpha is the first concrete product of Route 2. It runs the full triage workflow on checkpoint deltas under a scope contract:

- Shared base model only
- Small encoder checkpoints only
- Classification tasks only
- Evidence bootstrap required

Canonical instance: `field_trials/checkpoint_inventory_t02/`. HTML report at `preflight/alpha_bundle/report.html`.

---

## What comes next

The substrate generalization story is validated for the current scope. Extending further requires:

1. **GPU compute** — DeBERTa adjudication tests whether the mechanism ladder (not just the workflow) generalizes to a third backbone. ~3 hours on one GPU.
2. **Larger models** — Everything so far is on small encoders. Scaling requires new validation.
3. **Non-classification tasks** — All evidence is on classification. Generation, extraction, and retrieval are untested territory.

For the four Route 2 research programs that sit on top of this substrate (decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge), see the synthesis memo: `../../notes/n93_route2_synthesis.md`.
