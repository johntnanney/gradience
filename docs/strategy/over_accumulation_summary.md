# Over-Accumulation Follow-up Summary

## Outcome
- OA-v2 parallel line completed a strict-naive 30-pair rerun/cross-check cycle.
- Cohort design gate passed (`30` pairs in target `30–40` window).
- Threshold/policy promotion gate failed (`rule_1=false`, `rule_2=false`, `rule_3=false`, `rule_4=true`).
- Overall Spearman improved vs OA-v1, but intended high-overlap/low-conflict slice remained unstable.

## Interpretation
- Keep OA-v2 explicitly exploratory.
- Keep OA-v1 authoritative in policy/report defaults.
- Do not apply threshold/policy changes from current OA-v2 evidence.

## Next Step
- Run one final regime-pure strict-naive cohort only if we can increase intended-slice coverage materially.
- Current feasibility check: strict intended slice (`overlap>=0.25`, `conflict<=0.10`) has only `9` available pairs in current inventory.
- Use pre-committed stop rule:
  - if gate still fails after regime-pure rerun, pause OA-v2 as exploratory and stop policy escalation work.
