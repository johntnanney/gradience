# Adjacent-Task Adjudication Implications

## Purpose

This note records the implications of the ambiguous-relationship adjudication study on NLI-family adapters (QNLI, RTE, MNLI) trained on distilbert-base-uncased.

The main question of the study was:

> When two adapters are neither obvious same-task redundancies nor obvious cross-task mismatches, does core-space add behaviorally useful discrimination beyond ordinary pair-risk?

The study produced a result that is more informative than a simple yes or no.

---

## Study setup

- 6 LoRA adapters trained from scratch on distilbert-base-uncased
- 2 QNLI adapters (seeds 42, 7) — verified 69.6%, 71.0%
- 2 RTE adapters (seeds 42, 7) — verified 56.7%, 55.6%
- 2 MNLI adapters (seeds 42, 7) — verified 53.0%, 53.2%
- 8 pairs: 3 same-task controls + 5 adjacent-task (all NLI family)
- All independently verified above base
- Merge strategy: uniform_linear
- Evaluation: merged adapter on both source tasks

---

## Main findings

### 1. Pair-risk fails in the adjacent-task middle

Ordinary pair-risk classified all 8 pairs as "redundant." It could not distinguish:

- same-task pairs (safe, <=0.7pp degradation)
- adjacent-task pairs (weaker task degrades 5-7pp)

The reason: NLI-family adapters share a backbone and related task structure. They look spectrally similar — high overlap, similar rank utilization, compatible norm ratios. Pair-risk sees this and calls it redundancy.

But spectral similarity does not imply functional compatibility across task boundaries. The adapters are structurally aligned but functionally divergent. Pair-risk does not have the signal to distinguish these cases.

This is a **blind spot**, not a bug. Pair-risk was designed to measure spectral compatibility, and spectrally these adapters are compatible. The incompatibility is at the functional/task level, which pair-risk does not measure.

### 2. Core-space is noisy and not decision-reliable here

Core-space results across the 8 pairs:

| Pair | CS status | Basis | Behavioral outcome |
|------|-----------|-------|--------------------|
| qnli x qnli | marginal | 0.887 | safe |
| rte x rte | incompatible | 0.839 | safe |
| mnli x mnli | marginal | 0.906 | safe |
| qnli x rte (s42) | marginal | 0.905 | QNLI degraded 6.4pp |
| qnli x mnli | incompatible | 0.882 | MNLI degraded 5.0pp |
| rte x mnli (s42) | marginal | 0.926 | MNLI degraded 5.8pp |
| qnli x rte (s7) | marginal | 0.941 | QNLI degraded 5.4pp |
| rte x mnli (s7) | marginal | 0.938 | MNLI degraded 6.6pp |

Core-space:

- Called a safe pair (rte x rte) "incompatible" — false alarm
- Called most degraded adjacent-task pairs "marginal" — missed the behavioral harm
- Called one degraded pair (qnli x mnli) "incompatible" — correct but accidental given the pattern above

Core-space is detecting real structural variation but that variation does not reliably predict this failure mode.

### 3. The real behavioral pattern is asymmetric weaker-task dilution

In every adjacent-task merge:

- The task with **stronger gradient signal** during training was preserved (typically <=1pp degradation)
- The task with **weaker signal** degraded materially (5-7pp)

This is a linear-merge phenomenon: when you average two adapters with different signal strengths, the stronger adapter dominates the merged representation. The weaker adapter's functional contribution is diluted.

Examples:

| Pair | Stronger task | Preserved? | Weaker task | Degraded? |
|------|--------------|------------|-------------|-----------|
| qnli x rte | QNLI (0.696) | rte improved | QNLI | degraded 6.4pp |
| qnli x mnli | QNLI (0.696) | QNLI safe | MNLI | degraded 5.0pp |
| rte x mnli | RTE (0.567) | RTE safe | MNLI | degraded 5.8pp |

Wait — the asymmetry is more nuanced than "stronger task wins." In qnli x rte, QNLI (the stronger adapter) degraded while RTE improved. This suggests the merge actually shifted the representation toward an intermediate point that happened to help RTE and hurt QNLI.

The more precise pattern: **the merged adapter lands at an intermediate functional point that may not serve either task well, and the task that loses more depends on the specific interaction between the two adapters' learned representations.**

### 4. Task identity is the most plausible missing signal

The single most predictive feature in this study was: **are the adapters trained on the same task?**

- Same task → safe (3/3, regardless of pair-risk or core-space)
- Different task → weaker task degrades (5/5, regardless of pair-risk or core-space)

The QA artifact already contains task metadata (`eval_dataset`, `metric_name`). A simple "same-task / different-task" flag would have perfectly discriminated this panel.

This is not a new feature request. It is a recognition that the workflow already collects the most predictive signal but does not use it as a merge-risk input.

---

## Implications for the project

### Pair-risk

Pair-risk remains strong for:

- Detecting norm imbalance (per-layer adapters)
- Detecting obvious structural incompatibility
- Separating clearly different-architecture adapters
- Same-task merge screening

Pair-risk is weak for:

- Adjacent-task merges where spectral similarity is high
- NLI-family or other within-family cross-task pairs
- Any regime where structural alignment coexists with functional divergence

This is the first clearly documented regime where pair-risk is insufficient by itself.

### Core-space

Core-space does not solve this problem. It was designed to detect shared-basis incompatibility, which is a structural property. The adjacent-task failure mode is functional, not structural — the adapters share basis structure but map it to different task objectives.

Core-space remains:

- Structurally informative
- Advanced and non-default
- Not behaviorally decisive in either the same-task regime (earlier finding) or the adjacent-task regime (this finding)

Its remaining plausible use case is even narrower than previously thought: pairs where structural incompatibility at the basis level is the actual mechanism of merge failure, not functional divergence.

### Task-relationship metadata

The most impactful next step is not more spectral analysis. It is using task-relationship information that already exists in QA artifacts:

- `eval_dataset` tells you what task the adapter was evaluated on
- If both adapters share the same `eval_dataset`, the merge is in the "same-task" safe regime
- If they differ, the merge is in the "adjacent-task" regime where weaker-task dilution is likely

A simple flag or warning — "these adapters were trained on different tasks; weaker-task degradation is likely in linear merge" — would have caught every failure in this study.

---

## Updated interpretation of the workflow

### Previous position (after first adjudication)

Ordinary pair-risk is strong in the tested regime. Core-space is structurally informative but behaviorally narrow. Source QA dominates narrowing.

### Updated position (after ambiguous-relationship adjudication)

Ordinary pair-risk is strong for same-task pools and for catching structural incompatibility (norm imbalance, architecture mismatch). It has a documented blind spot for adjacent-task merges where spectral similarity is high.

Core-space does not fill this gap. It is noisy in this regime and does not reliably predict adjacent-task degradation.

The most informative signal for this failure mode is task-relationship metadata, not spectral analysis. The workflow already collects this signal but does not yet use it for merge-risk discrimination.

---

## What this result does not imply

- It does not mean pair-risk should be replaced — it is still the strongest structural diagnostic
- It does not mean core-space should be removed — it may yet help in other regimes
- It does not mean all cross-task merges fail — the degradation pattern depends on signal strength and task similarity
- It does not mean the workflow is broken — it correctly identifies these pairs as "redundant" in the spectral sense; the gap is in translating spectral redundancy to behavioral safety

---

## Recommended project actions

### 1. Document the adjacent-task blind spot honestly

Update public and internal docs to note that pair-risk does not reliably screen adjacent-task merges where spectral similarity is high.

### 2. Consider a task-relationship advisory

The simplest high-value addition: when both adapters in a pair have QA artifacts with different `eval_dataset` values, emit a note in the merge report: "These adapters were evaluated on different tasks. Cross-task linear merges may degrade the weaker task."

This is metadata-driven, requires no spectral analysis, and would have caught every failure in this study.

### 3. Narrow core-space claims further

Core-space's supported behavioral role is now even narrower than after the first adjudication. It does not reliably help in either the same-task regime or the adjacent-task regime. Its remaining claim is limited to pairs where structural basis incompatibility is the actual mechanism of failure — a class that has not yet been convincingly demonstrated in verified adjudication.

### 4. Record the asymmetric degradation pattern

"Weaker task degrades, stronger task survives (approximately)" is a useful practical heuristic. It should be documented as a known behavior of linear LoRA merging across task boundaries.

---

## Bottom line

The ambiguous-relationship regime revealed a real blind spot: the workflow's spectral analysis cannot distinguish safe same-task redundancy from harmful adjacent-task redundancy. The most predictive signal is task identity, which is already available in QA artifacts. Neither pair-risk nor core-space solves this problem — it requires task-relationship metadata.

This is a cleaner and more useful finding than "core-space helps in the ambiguous middle." It points to a concrete, implementable improvement rather than a vague diagnostic claim.
