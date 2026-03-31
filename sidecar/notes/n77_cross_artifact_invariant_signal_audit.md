# n77 -- Cross-Artifact Invariant Signal Audit

**Type:** findings note
**Date:** 2026-03-31
**Program:** Cross-Artifact Compatibility Research (Route 2)
**Stage:** B
**Depends on:** n76 (panel definition), Ring 1 results, Ring 2 Stages A-D, field trial data, decision-dependent compatibility (n70-n74)
**Status:** complete

---

## Question

Which compatibility signals recur across artifact classes when the artifact classes differ in representation form but share the same base-model family and task relations?

---

## Method

Audit five signal families against all nine panel cases from n76, grouped by artifact class (LoRA, LoHa, checkpoint delta). For each signal family, record whether it is present, absent, or not testable in each class. A signal qualifies as "recurring" if it appears in all classes where it can be tested.

---

## Signal families audited

### 1. QA / evidence regime

**Question:** Does evidence gating remain the dominant operational force?

| Class | Observed? | Evidence |
|-------|-----------|----------|
| LoRA | Yes | Behavioral data for 2/3 panel cases. QA gating dominates all field trial inventories. Strict-QA blocks unknown/weak. |
| LoHa | Yes | All 3 adapters `unknown_no_behavioral_eval`. Strict-QA blocks all 3 pairs despite low structural risk. |
| Ckpt delta | Yes | All 4 sources `behavioral_missing`. QA flagged qnli_s42 weak. Triage: "Source QA is the binding constraint." 6 pairs -> 1. |

**Verdict: recurs across all three classes. Strong invariant.**

The QA regime is representation-agnostic. It operates on evidence metadata, not on structural measurements. The same blocking logic fires regardless of whether the underlying data is native LoRA factors, shimmed LoHa materialization, or checkpoint summary representation.

### 2. Same-task vs cross-task separation

**Question:** Does same-task tend to look more compatible than cross-task?

| Class | Observed? | Evidence |
|-------|-----------|----------|
| LoRA | Yes | Same-task: retained, merge acc 0.876. Cross-task: control, compatibility 0.111. Sidecar S01: same-task always safe. |
| LoHa | Not testable | All 3 pairs are same-task. No cross-task comparator exists. |
| Ckpt delta | Yes | Same-task compatibility 0.892. Cross-task mean 0.704. Same-minus-cross = 0.188. Only same-task pair survived triage. |

**Verdict: recurs across tested classes. Moderate invariant.**

The separation is directionally consistent in LoRA and checkpoint delta. The magnitude differs (LoRA shows a wider gap in triage outcomes; checkpoint delta shows a moderate but consistent numerical separation). Cannot be tested in LoHa with current panel.

### 3. Same-family intermediate behavior

**Question:** Does same-family occupy a distinct intermediate position?

| Class | Observed? | Evidence |
|-------|-----------|----------|
| LoRA | Yes | MNLI x QNLI (NLI): compatibility 0.431, routing confusability moderate (0.379). Between same-task (0.475, high confusability) and cross-task (0.111, low confusability). |
| LoHa | Not testable | No same-family pairs exist. |
| Ckpt delta | Yes | SST-2 x Yelp (sentiment_binary): compatibility 0.652. Between same-task (0.892) and cross-task (0.626). Routed to same-family informational caution. |

**Verdict: recurs across tested classes. Moderate invariant.**

The three-way ordering `same_task > same_family > cross_task` holds in both LoRA and checkpoint delta. The exact position of same-family within the range varies (closer to cross-task in checkpoint deltas, more centered in LoRA), but the intermediate status is consistent.

### 4. Conservative narrowing

**Question:** Does the workflow still reduce to a smaller, useful subset?

| Class | Observed? | Evidence |
|-------|-----------|----------|
| LoRA | Yes | 70-90% reduction in field trials. Inventory 02: 10->1. Inventory 03: 36->3. |
| LoHa | Yes | 3 pairs all structurally low-risk, but all blocked by strict-QA. 100% blocked without behavioral evidence. |
| Ckpt delta | Yes | 6 pairs -> 1 retained. 83% reduction. QA + task boundary + structural risk. |

**Verdict: recurs across all three classes. Strong invariant.**

The narrowing mechanism is the same everywhere: QA evidence gating fires first, then task-relation boundaries, then structural risk. The workflow shape (broad input -> narrow output) survives representation change. The narrowing ratio varies by panel size and evidence coverage, but the logic is consistent.

### 5. Near-miss / middle states

**Question:** Do intermediate categories (near-miss, marginal) recur?

| Class | Observed? | Evidence |
|-------|-----------|----------|
| LoRA | Yes | 7 near-miss pairs in field trials. Avg delta vs best = -0.006. Behaviorally safe. Distinct product category confirmed. |
| LoHa | No | All pairs low-risk, same-task. Panel too homogeneous. |
| Ckpt delta | Partial | No explicit near-miss pairs. Same-task pair is closest probe but QA dominance prevented the distinction from emerging. |

**Verdict: present in one class only. Weak / inconclusive.**

Near-miss is well-validated in LoRA but has not appeared in the other two classes. This is likely a panel coverage gap: both non-LoRA panels are too small and homogeneous to produce near-miss conditions. The absence is inconclusive, not negative.

---

## Summary

| Category | Signals | Count |
|----------|---------|-------|
| Strong invariants (all classes) | QA/evidence regime, conservative narrowing | 2 |
| Moderate invariants (tested classes) | Same-task vs cross-task, same-family intermediate | 2 |
| Weak / inconclusive | Near-miss middle states | 1 |

**Key finding:** The two strongest invariants -- QA evidence gating and conservative narrowing -- are workflow-level signals, not structural measurement signals. They operate on evidence metadata and triage policy, not on factor geometry or spectral metrics. This is consistent with H3 ("artifact broadening preserves workflow shape more than feature parity").

The two moderate invariants -- task-relation separation and same-family intermediate behavior -- are structural in nature but survive representation change because the distinction they capture (task similarity) is upstream of representation form.

---

## Implications for Stage C

The invariant signals identified here are workflow-level and task-relational. The next stage should identify what is *not* portable -- the representation-local signals that should not be generalized across artifact classes.

---

## Output artifacts

- `sidecar/results/cross_artifact_portability/invariant_signal_matrix.json`
- `sidecar/results/cross_artifact_portability/invariant_signal_matrix.md`
- `sidecar/notes/n77_cross_artifact_invariant_signal_audit.md` (this note)
