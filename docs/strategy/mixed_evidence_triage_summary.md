# Mixed-Evidence Triage Stress-Test Summary

Date: 2026-03-31  
Status: complete (bounded stress-test pass)

## Purpose

Stress-test whether Route 2 triage remains coherent when the panel is intentionally weighted toward mixed-evidence review and same-family optional cases.

## Study outputs

- `sidecar/notes/n113_mixed_evidence_triage_baseline.md`
- `sidecar/notes/n114_mixed_evidence_triage_panel.md`
- `sidecar/notes/n115_mixed_evidence_triage_rerun.md`
- `sidecar/notes/n116_mixed_evidence_triage_interpretation.md`
- `sidecar/notes/n117_mixed_evidence_triage_stress_test_memo.md`
- `sidecar/results/route2_stress_tests/mixed_evidence_triage/`

## Key outcomes

1. QA-dominant aggregation remained coherent and distinct (`qa_clear` / `qa_review` / `qa_blocked`).
2. Same-family optional cases remained review-like / safe-like rather than collapse-like.
3. The soft middle remained structured enough for review-first narrowing.
4. Structural nuance remained useful inside `qa_review` as secondary prioritization, not primary gate replacement.

## Guardrails

Keep guarded:

- exact review thresholds,
- strict internal ordering claims,
- hard cutpoint language across different panels.

## Route 2 language update

Safe strengthening:

- “The triage middle is structured-with-guardrails under mixed-evidence stress.”
- “Same-family optional cases are generally review/optional, not collapse-like.”

Still bounded:

- fine-grained review ranking remains sidecar-level and non-canonical.

## Bottom line

This stress test supports stronger confidence in the practical triage middle without expanding scope: coherence holds at family level, while threshold precision remains explicitly guarded.
