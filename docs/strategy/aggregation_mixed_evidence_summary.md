# Aggregation Mixed-Evidence Summary (Route2 Targeted Perturbation)

**Date:** 2026-03-31
**Scope:** CPU-only, triage-weighted soft-middle stress test

---

## What this pass tested

This pass stress-tested the aggregation-sensitive framework on a panel intentionally biased toward:

- mixed-evidence review cases
- same-family optional cases
- borderline triage states

The goal was to test coherence in the practical triage middle, not to reopen the whole aggregation program.

---

## Main outcome

The soft middle remained coherent, with guardrails.

- QA-dominant aggregation remained a distinct and stable family.
- Same-family optional cases stayed in review/clear lanes, not blocked/collapse lanes.
- Mixed-evidence review states remained interpretable rather than collapsing into noise.

---

## What remained guarded

- exact review-state thresholds
- exact taxonomy boundary cut lines
- strong claims about fine-grained ordering inside review states

Structural nuance appears to re-enter as secondary prioritization inside `qa_review`, but this should remain guarded language.

---

## Route 2 language implication

Safe to strengthen:

- QA-dominant logic is coherent even in mixed-evidence-heavy triage.
- Same-family optional cases are generally better described as review/optional than collapse-like.
- The triage middle is structured-with-guardrails.

---

## Bottom line

This targeted pass supports a more confident but still bounded statement: the aggregation framework remains usable in the triage soft middle, provided we keep threshold and fine-grained taxonomy claims explicitly guarded.
