# Cross-Artifact Invariant Signal Matrix

Generated: 2026-03-31

## Signal Family Recurrence

| Signal Family | LoRA | LoHa | Ckpt Delta | Recurrence | Strength |
|---------------|------|------|-----------|------------|----------|
| QA / evidence regime | Present | Present | Present | All classes | **Strong** |
| Same-task vs cross-task separation | Present | Not testable | Present | Tested classes | Moderate |
| Same-family intermediate behavior | Present | Not testable | Present | Tested classes | Moderate |
| Conservative narrowing | Present | Present | Present | All classes | **Strong** |
| Near-miss / middle states | Present | Not observed | Partial | One class | Weak |

## Strong Invariants (recur across all three classes)

**1. QA / evidence regime.** Evidence bootstrap and QA gating are the single most portable compatibility signal. In all three artifact classes, missing or weak behavioral evidence triggers the same triage narrowing regardless of representation form. The mechanism is identical: `unknown_no_behavioral_eval` or `flagged_weak` sources are blocked by strict-QA, and the evidence tier dominates over structural metrics when present.

**2. Conservative narrowing.** The workflow reduces candidate space to a smaller, useful subset in all three classes. LoRA field trials show 70-90% reduction. Checkpoint deltas show 83% reduction (6 to 1). LoHa shows 100% blocking by QA (structural risk is low but evidence is missing). The narrowing logic -- QA gating + task boundary + structural risk -- survives representation change.

## Moderate Invariants (recur in classes where testable)

**3. Same-task vs cross-task separation.** Confirmed in LoRA (behavioral data: retained vs control) and checkpoint delta (compatibility 0.892 vs 0.704). Cannot be tested in LoHa (same-task only panel). The ordering is consistent where observable.

**4. Same-family intermediate behavior.** LoRA: MNLI x QNLI (NLI family) sits between same-task and cross-task on both merge compatibility and routing confusability. Checkpoint delta: SST-2 x Yelp (sentiment_binary) sits between same-task and cross-task on compatibility score. The three-way ordering `same_task > same_family > cross_task` holds in both classes.

## Weak / Inconclusive

**5. Near-miss middle states.** Well-validated in LoRA (7 field trial pairs, behaviorally safe). Not observed in LoHa or checkpoint delta -- but both non-LoRA panels may be too small to produce near-miss cases. Inconclusive rather than negative.
