# How to Read Gradience Outputs

This document walks through the six major output sections produced by Gradience's inventory and monitoring workflow. The intended audience is a researcher or collaborator interpreting Gradience output for the first time.

The outputs are produced by three commands:

```bash
gradience summarize-inventory --qa-dir qa/ --report-dir reports/
gradience suggest-neighborhoods --qa-dir qa/ --report-dir reports/
gradience monitor run.jsonl
```

The sections below appear in the order you would encounter them reading top to bottom through a full workflow run.

---

## 1. Overview

**Produced by:** `gradience summarize-inventory`
**Section header:** `INVENTORY OVERVIEW`

```
  INVENTORY OVERVIEW
  ============================================================
  Mixed-quality inventory — 2 weak/unknown source(s) identified

  Merge reports:      2
  QA artifacts:       3
```

The headline is auto-generated from the counts. Three possible forms:

- `"Mixed-quality inventory — N weak/unknown source(s) identified"` — one or more adapters failed or lacked behavioral QA. This is the most important case; source QA will be the dominant narrowing step.
- `"N adapters, M pairs — high structural risk dominates"` — most pairs are high-risk and no weak sources are blocking. Task-boundary advisories on individual pair reports will be the next thing to check.
- `"N adapters, M pairs"` — baseline. Inventory passes QA and has no structural alarm; proceed to deeper sections.

**`QA artifacts`** is the count of adapter-level QA artifacts (`gradience.adapter_qa/v1`) passed to the summarizer. **`Merge reports`** is the count of pairwise reports (`gradience.merge_qa_report/v1`). These should be consistent with your input directories — if they are not, check that your `--qa-dir` and `--report-dir` paths are correct and that all files have the expected schema field.

---

## 2. Trust Snapshot

**Produced by:** `gradience summarize-inventory`
**Section header:** `SOURCE QA SNAPSHOT`

```
  SOURCE QA SNAPSHOT
  ----------------------------------------
  eligible:  1
  flagged_weak:  1
  unknown_no_behavioral_eval:  1

  1 source(s) with behavioral evidence (user-provided)
  1 source(s) flagged weak
  1 source(s) with missing behavioral evidence
  Note: behavioral scores are user-provided; Gradience does not
  independently verify claimed evaluation results.
  Strict-QA block candidates: 2
```

This section reports the eligibility status assigned to each adapter during `gradience audit`. There are four possible statuses:

| Status | Meaning |
|--------|---------|
| `eligible` | Adapter has behavioral evaluation and outperforms (or matches) the base model on the reported metric. |
| `uncertain` | Adapter has behavioral evaluation but the result is ambiguous (e.g. margin is within noise, or a metric interaction warrants caution). |
| `flagged_weak` | Adapter has behavioral evaluation and **underperforms** the base model. It is an active liability: merging it will likely degrade the merged adapter. |
| `unknown_no_behavioral_eval` | No behavioral evaluation was provided. Structural measurements are present but the adapter has never been tested against its task. |

The provenance note is important: Gradience does not run evaluations itself. All `eval_dataset`, `adapter_score`, and `base_score` fields in QA artifacts are user-supplied when generating the artifact with `gradience audit --eval-*` flags. If a collaborator provided QA artifacts that you did not personally generate, treat `eligible` as "claims to be eligible" until you verify the evaluation methodology.

**`Strict-QA block candidates`** counts merge reports where at least one adapter is `flagged_weak`, `unknown_no_behavioral_eval`, or has no QA provided at all. This is the count of pairs that would be blocked if you ran `gradience merge-audit --strict-qa`. It is not a recommendation to use `--strict-qa` in all cases — it is a pre-flight signal. A count of 0 means all pairs in the inventory have two QA-verified sources on both sides.

**Common misread:** `unknown_no_behavioral_eval` is not a structural failure. An adapter with this status may be perfectly well-trained — Gradience simply has no task performance evidence for it. Whether this matters depends on your use case. If you are evaluating a merge candidate for deployment, it matters a lot. If you are doing exploratory spectral research, it may not.

---

## 3. Policy Summary

**Produced by:** `gradience summarize-inventory`
**Section headers:** `STRUCTURAL DETAIL` and `INTERPRETATION`

```
  STRUCTURAL DETAIL
  ----------------------------------------
  Flags: high_rank_waste: 1, low_utilization: 1
  Pair risk: high: 1, low: 1
  Strategies: audit_aware: 1, linear: 1
  Issues: none: 1, subspace_conflict: 1
```

This section is pure counting — it aggregates fields from the individual QA artifacts and merge reports. No new scoring happens here. Each entry is a distribution over possible values.

### Adapter structural flags

Flags are set during `gradience audit` and reflect the adapter's internal rank utilization:

| Flag | Meaning |
|------|---------|
| `low_utilization` | Mean utilization < ~0.3. The adapter is not using most of its configured rank. |
| `high_rank_waste` | `rank_waste_ratio` > ~0.5. More than half the nominal rank is structurally inert. |
| `concentrated_spectrum` | Energy is concentrated in very few singular values. Often co-occurs with `low_utilization`. |
| `underutilized_capacity` | Broader low-use condition; may overlap with `low_utilization` at different thresholds. |

Flags do not by themselves determine eligibility. A `flagged_weak` adapter with `high_rank_waste` is doubly problematic; an `eligible` adapter with `high_rank_waste` may simply be over-provisioned.

### Pair risk

`pair_risk` is assigned by `gradience merge-audit` and reflects the per-layer verdict distribution for the pair. It is **structural risk only** — it does not incorporate source QA quality.

- `low` — most layers are safe or mildly redundant; no dominant adverse issue
- `medium` — mixed verdicts; some conflict or imbalance present but not dominant
- `high` — conflicting or severely imbalanced layers dominate

**Common misread:** Medium pair risk does not mean "safe." In the mixed-task example in `examples/inventories/`, 11 of 15 pairs are medium-risk — but 13 of those 15 are cross-task pairs that would degrade at least one task if merged. Pair risk alone cannot distinguish safe same-task pairs from risky cross-task ones. The task-boundary advisory on individual pair reports is needed for that partition.

### Recommended strategies

Strategies are set by `gradience merge-audit` based on the pair diagnosis:

| Strategy | When assigned |
|----------|---------------|
| `linear` | Low-risk pairs with orthogonal or weakly overlapping subspaces. Simple weighted average. |
| `norm_equalized` | Medium risk with norm imbalance as dominant issue. Equalizes spectral norms before blending. |
| `audit_aware` | High-risk pairs or pairs with `subspace_conflict`. Per-layer strategy selection required. |
| `ties` / `dare_ties` | Redundancy-dominated pairs. Trims redundant parameters before merging. |

### Dominant issues

Each pair report identifies a single dominant issue:

| Issue | What it means |
|-------|---------------|
| `none` | No adverse structure detected; adapters are compatible. |
| `partial_redundancy` | Overlap is present but not total; some unique contribution from each. |
| `high_redundancy` | Strong overlap; merging will not improve over either source alone. |
| `norm_imbalance` | One adapter's singular values are much larger; will dominate the merged output. |
| `subspace_conflict` | Adapters point in incompatible directions in weight space; merging will suppress one or both. |
| `unknown` | Could not determine dominant issue (e.g. degenerate adapter). |

```
  INTERPRETATION
  ----------------------------------------
  2 adapter(s) have weak or missing behavioral evidence.
  Source QA is likely the main narrowing step for this inventory.
```

The interpretation block synthesizes the counts into a one or two sentence guidance. It follows simple rules: if weak/unknown sources are present, QA dominates; if high risk dominates (>50% of pairs), structural risk is the story; otherwise check task-boundary advisories. Use this as a reading aid, not a substitute for examining individual pair reports.

---

## 4. Action Plan

**Produced by:** `gradience summarize-inventory` (when `build_action_plan` is invoked, see `gradience.api.summarize_inventory`)
**Section header:** `INVENTORY ACTION PLAN`

```
  INVENTORY ACTION PLAN
  ============================================================

  REDUCED CANDIDATE SET
  ----------------------------------------
  Starting pairs:      4
  Retained candidates: 2
  Reduction:           50%

  Evaluate first:
    - mixed_a × mixed_b (same-task)
    - mixed_b × mixed_c (same-task)

  Exclude / deprioritize
  ----------------------------------------
  - mixed_d: weak source — low confidence

  Same-task safe zone
  ----------------------------------------
  - mixed_a × mixed_b
  - mixed_b × mixed_c

  Cross-task caution zone
  ----------------------------------------
  - MNLI × QNLI region
  - do not prioritize these pairs for casual merge exploration

  Summary
  ----------------------------------------
  QA and task boundary dominate this inventory. Candidate space reduced from 4 pairs to 2 (50% reduction).
```

The action plan is the highest-level output of the inventory workflow. It is a presentation layer over existing signals — no new scoring.

### Reduced candidate set

The reduction percentage tells you how much work the preflight pass saved. In the mixed-task example in `examples/inventories/mixed_task_preflight_example.md`, a 6-adapter 4-task inventory goes from 15 pairs to 2 (87% reduction) after task-boundary advisory filtering alone — even though pair risk alone showed 11 pairs as medium-risk (structurally plausible).

A reduction of 0% is expected for same-task inventories; it means the workflow is confirmatory. From `examples/inventories/same_task_control_example.md`:

> This is a same-task safe pool. The workflow confirms what context already suggests: all 3 pairs are reasonable merge candidates.

### Evaluate first

Capped at 4 entries. These are the same-task pairs with no weak sources — the candidates with the strongest prior for a successful merge. Ordering within this list is not meaningful; all entries are co-equal priority.

### Exclude / deprioritize

Adapters removed from the candidate set due to QA status. Two possible labels:
- `"weak source — low confidence"` — adapter is `flagged_weak`
- `"missing behavioral evidence — low confidence"` — adapter is `unknown_no_behavioral_eval`

Adapters in this list are not deleted from your inventory; they are deprioritized. If a particular pair involving a deprioritized adapter is important to you for research reasons, you can still run `gradience merge-audit` on it directly.

### Same-task safe zone / Cross-task caution zone

Same-task pairs are those where both adapters were evaluated on the same dataset (determined by comparing `eval_dataset` fields in their QA artifacts). The task-relationship advisory in merge reports is the signal that drives this partition.

Cross-task pairs appear in the caution zone. The region labels (e.g. `"MNLI × QNLI region"`) are derived from the `eval_dataset` field and normalized. If `eval_dataset` is absent on either side, no region label is generated, but the pair still falls into the cross-task bucket if an advisory is present on the merge report.

**What "cross-task caution" means:** Validated on 49 same-task pairs across tested stressors, same-task merges produced 0 material degradations. Cross-task merges in the same study showed consistent degradation on at least one task. "Do not prioritize" means you should not invest evaluation resources there unless you have a specific hypothesis. It does not mean the merge is physically impossible.

### Summary line

Five possible summary forms, in order of narrative priority:

1. `"No pair reports available for interpretation."` — nothing to act on
2. `"QA dominates this inventory; no credible same-task candidates remain."` — weak sources + no same-task pairs
3. `"All pairs are cross-task; no same-task safe region exists in this inventory."` — all eligible adapters but all cross-task
4. `"QA and task boundary dominate this inventory. Candidate space reduced from N pairs to M."` — both filters applied
5. `"Inventory is mostly explained by task boundary. Candidate space reduced from N pairs to M (P% reduction)."` — task boundary only
6. `"This same-task inventory is mostly confirmatory."` — all pairs retained, nothing filtered

---

## 5. Region Summary / Candidate-Space Map

**Produced by:** `gradience suggest-neighborhoods`
**Section header:** `MERGE NEIGHBORHOODS`

```
MERGE NEIGHBORHOODS

Group 1: qnli_r16_s42, qnli_r16_s123
- characterization: likely-safe neighborhood
- common strategy: linear
- dominant issue: none

Group 2: sst2_r16_s42, sst2_r16_s123
- characterization: likely-safe neighborhood
- common strategy: linear
- dominant issue: none

Excluded:
- chat_lora (flagged_weak)

Boundary warnings:
- cluster_01 <-> cluster_02: conditional cross-group merge risk (audit-aware checks recommended)
```

The neighborhood report clusters adapters into groups where within-group merges are structurally coherent. It is the spatial complement to the action plan — where the action plan gives you a ranked list of candidates, the neighborhood report gives you a map of the merge landscape.

### Groups

Each group has:

**`group_id`** — internal identifier used in boundary warnings. Typically `cluster_01`, `cluster_02`, etc.

**`members`** — adapter names or paths within the group. All members are plausible merge candidates with each other. In a well-structured same-task inventory, each task forms its own group (as in the mixed-task example in `examples/inventory_preflight_mixed_task/`).

**`characterization`** — one of three values:

| Value | Meaning |
|-------|---------|
| `"likely-safe neighborhood"` | Members share compatible subspaces and have low structural risk with each other. |
| `"caution neighborhood"` | Members have conditional compatibility; structural risk is not severe but requires audit-aware checks. |
| `"audit-aware neighborhood"` | Members have significant structural complexity; per-layer merge planning is required. |

**`common_strategy`** — the merge strategy that applies to most within-group pairs. Derived from the `recommended_strategy` field of the underlying merge reports for within-group pairs.

**`dominant_issue`** — the dominant structural issue across within-group pairs. `none` means no adverse structure detected.

### Excluded

Adapters excluded from all groups due to their QA status. An excluded adapter will not appear as a member of any group. The reason field is the same as in the action plan.

### Boundary warnings

Boundary warnings flag cross-group merges that carry elevated risk. `cluster_A <-> cluster_B: reason` means that merging members of group A with members of group B requires additional care. The reason is always one of:

- `"conditional cross-group merge risk (audit-aware checks recommended)"` — the most common form; cross-group pairs have structural incompatibilities that may not be apparent from either group's individual characterization.

**Interpreting a fully-fragmented neighborhood report:** When every adapter forms its own singleton group (as in `examples/inventory_preflight_mixed_task/inventory/neighborhoods.json`), with boundary warnings between all pairs, the neighborhood report is confirming the action plan's finding that all pairs are cross-task. No within-group candidates exist. This is not a failure — it is the correct output for a cross-task inventory.

**What neighborhoods add over the action plan:** The action plan gives you a flat list. The neighborhood report gives you group membership. If your inventory has 20+ adapters, you can use neighborhoods to identify which sub-pools to evaluate together rather than evaluating all O(N²) pairs.

---

## 6. Drift Summary

**Produced by:** `gradience monitor`
**Section headers:** `GRADIENCE MONITOR`, `Diagnostics`, `LoRA audit`, `Alerts`, `Recommendations`

The drift summary is the training-time counterpart to the post-training QA workflow. Where `gradience audit` measures a completed adapter, `gradience monitor` reads a telemetry JSONL file (written by `GradienceCallback` during training) and reports on training dynamics as they developed.

A typical monitor output:

```
========================================================================
GRADIENCE MONITOR
========================================================================
File: run.jsonl
Model:   microsoft/DialoGPT-small
Dataset: gsm8k
Profile: gsm8k

Latest eval signals:
  Train PPL: 6.23
  Test  PPL: 9.41
  Gap:       1.51x
  Train Acc: 24.0%
  Test  Acc: 21.0%

Diagnostics:
  Stable rank (mean): 12.8
  Utilization (mean): 78.0%

LoRA audit:
  LoRA params: 442.4K
  Layers:      32
  Energy rank k@90% (p50/p90): 9.2/11.5
  Suggested rank (median): r=8 likely sufficient for most layers (p50 k@90%=9.2)
  Suggested rank (p90):    r=12 covers worst-case layers at 90% energy (p90 k@90%=11.5)

Alerts (1):
  01. [WARNING] excessive_drift: Model diverging from pretrained weights: 62% drift rate

Recommendations (1):
  01. [WARNING] reduce_rank: Adapter effective rank suggests r=8 would suffice
```

### Latest eval signals

These are the most recent values from the telemetry stream:

- **Train PPL / Test PPL** — perplexity on training and evaluation sets. The ratio between them is the **Gap**.
- **Gap** — `test_ppl / train_ppl`. A gap above ~1.5 (configurable via `--gap-threshold`) triggers a memorization warning. A gap of 1.51x is borderline; above 2.0x is strongly concerning for generalization.
- **Train Acc / Test Acc** — task accuracy if available. These are user-supplied from the evaluation loop; Gradience does not compute them itself.

### Diagnostics

The diagnostics section surfaces LoRA-specific structural signals from the most recent checkpoint:

- **Stable rank (mean)** — mean stable rank across adapter layers. Conceptually, the effective number of independent dimensions the adapter is using. For a nominal rank-16 adapter with stable rank 12.8, the adapter is using roughly 80% of its configured capacity.
- **Utilization (mean)** — fraction of the configured rank that is contributing meaningfully to the adapter's function. `78%` means roughly 78% of singular values are above the utilization threshold. Below ~30% is flagged as `low_utilization` in the post-training QA artifact.

### LoRA audit

The LoRA audit block reports the energy rank at 90% cumulative energy, split by percentile:

- **`Energy rank k@90% (p50/p90)`** — across all adapter layers, the median layer needs `k=9.2` singular values to capture 90% of its spectral energy; the worst-case layer (90th percentile) needs `k=11.5`. This is the empirical basis for the rank suggestion.
- **Suggested rank (median / p90)** — conservative rank recommendations. `r=8` is suggested as sufficient for most layers; `r=12` covers even the most rank-hungry layers. These are the same signals that inform `rank_waste_ratio` in the post-training artifact.

If `by_type` data is present, a per-module-type breakdown appears:

```
  Diagnostics:
    attn: params=221.2K  util=81.0%  sr=13.2
    mlp:  params=221.2K  util=75.0%  sr=12.4
```

This shows whether attention vs. MLP layers are using their rank differently. Divergent utilization between layer types can inform targeted rank selection (e.g. higher rank for attention, lower for MLP).

### Alerts

Alerts are triggered by condition classes in `gradience.finetune.alerts`. Each alert has a severity, a code, and a message. Severity levels: `INFO`, `WARNING`, `CRITICAL`.

Key alert codes:

| Code | Severity | Signal |
|------|----------|--------|
| `excessive_drift` | WARNING | `||W_t - W_0||` (delta magnitude) growing >50% per measurement window. Model diverging from pretrained weights faster than expected. |
| `capacity_collapse` | CRITICAL | Effective rank of model weights dropping >10% from baseline. Model is losing representational capacity — potentially forgetting capabilities outside the fine-tuning distribution. Dangerous because training loss may still look good. |
| `feature_distortion` | CRITICAL | Spectral norm σ_max spiking >30% above baseline. Pretrained features being distorted by excessively large updates. |
| `lora_effective_rank_low` | WARNING | Adapter using <25% of its configured nominal rank. VRAM is being wasted. |
| `rank_growing_fast` | WARNING | ΔW rank growing >50% in recent window. Possible overfitting. |
| `training_saturated` | INFO | ΔW rank growth <5% in recent window. Training has plateaued; consider stopping. |

Each alert includes a `recommendation` with specific actionable steps (reduce learning rate, add regularization, roll back checkpoint, etc.). In non-verbose mode, recommendation text appears for CRITICAL alerts only. Use `--verbose` to see all recommendations.

**What `excessive_drift` means in practice:** The condition compares the mean delta magnitude in the most recent window to the preceding window. A 62% drift rate means the adapter weights have moved 62% faster in the recent period than in the period just before. This is a rate-of-change alarm, not an absolute magnitude alarm — early training has legitimate rapid movement; a spike in an otherwise stable training run is the signal. Check whether the drift coincides with a learning rate schedule change or a dataset shift.

### Recommendations

Recommendations are higher-level, actionable suggestions that may not correspond to a single alert. Common examples:

- `reduce_rank` — suggests a smaller nominal rank based on observed `energy_rank_90`
- `consider_lora` — if full fine-tuning is in use and ΔW rank is consistently low, suggests switching to LoRA
- `checkpoint_recommended` — issued before a CRITICAL alert materializes, based on leading indicators

---

## Reading the Full Workflow Together

The six sections form a decision funnel:

1. **Overview** — how many artifacts are in play; first alarm if sources are weak
2. **Trust snapshot** — which adapters have credible behavioral evidence; sets the QA filter
3. **Policy summary** — structural characteristics of the pair matrix; sets the risk filter
4. **Action plan** — combines QA and structural signals into a ranked candidate list
5. **Region summary** — maps the candidate space spatially; identifies which sub-pools to evaluate
6. **Drift summary** — for adapters still in training, surfaces spectral and behavioral signals before committing to a full QA pass

In a clean same-task inventory (all adapters eligible, no structural pathologies), sections 2–4 are confirmatory. The value of running the workflow comes when there is heterogeneity — mixed tasks, mixed quality, or structural outliers — because the funnel will reduce a combinatorially large pair matrix to a small, defensible candidate set.

The canonical worked example for the mixed-task case is in `examples/inventories/mixed_task_preflight_example.md`. The canonical same-task control is in `examples/inventories/same_task_control_example.md`.
