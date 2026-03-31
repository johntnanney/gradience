# Shared vs Specific Table

Generated: 2026-03-31T01:23:15.534449+00:00

| Layer | Merge | Routing | Triage |
|---|---|---|---|
| Measurement | shared unchanged | shared unchanged | shared with translation |
| Diagnosis | shared with translation | shared with translation | shared with translation |
| Aggregation | scenario-specific (worst-case) | scenario-specific (distributional) | scenario-specific (QA gate-first) |
| Policy | scenario-specific (merge candidacy) | scenario-specific (routing actions) | scenario-specific (evaluation prioritization) |

## Scenario-pair notes

- **merge vs routing:** first divergence at `aggregation`; worst-case (merge) vs distributional (routing).
- **merge vs triage:** first divergence at `aggregation`; worst-case pair risk vs gate-first source QA.
- **routing vs triage:** first divergence at `aggregation`; distributional separability vs QA gate-first narrowing.
