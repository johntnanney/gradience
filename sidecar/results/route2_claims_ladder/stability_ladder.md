# Route 2 Claims Stability Ladder

Program: Route 2 Claims Stability Ladder  
Stage: D  
Generated: 2026-03-31

## Distribution

- stable: 11
- moderately_stable: 6
- thin: 2
- local_only: 1
- blocked_or_open: 0

## Stable

- A1: QA/evidence gating invariant (`safe_to_expose`)
- A2: conservative narrowing invariant (`safe_to_expose`)
- A3: checkpoint triage as real broadened workflow (`safe_with_guardrails`)
- C1: strongest structural metrics are representation-local (`safe_with_guardrails`)
- C2: workflow-layer portability stronger than metric-layer portability (`safe_to_expose`)
- C3: checkpoint broadening is summary-based, not factor-equivalent (`safe_with_guardrails`)
- D1: aggregation is first-class seam (`safe_to_expose`)
- D2: worst-case/distributional/QA-dominant families are operationally distinct (`safe_with_guardrails`)
- D3: QA-dominant is not weaker structure-only mode (`safe_with_guardrails`)
- F1: broadened substrate is real but narrow (`safe_with_guardrails`)
- F2: broadest portable value is evidence-aware triage workflow (`safe_to_expose`)

## Moderately Stable

- B1: same-task vs cross-task separation is directionally useful (`safe_with_guardrails`)
- B2: same-family intermediate status is real but fragile (`safe_with_guardrails`)
- B3: same-family optional often behaves safe-like (`safe_with_guardrails`)
- D4: aggregation taxonomy is viable with guarded thresholds (`safe_with_guardrails`)
- E1: Route 2 profiles show distinct behavioral footprints (`safe_with_guardrails`)
- E4: same-family optionality tracks review/safe-like behavior (`safe_with_guardrails`)

## Thin

- B4: cross-artifact optional/near-miss portability outside LoRA (`research_only`)
- E3: routing-confusability lacks stable merge-setting behavioral signature (`research_only`)

## Local Only

- E2: collapse vs contamination channel split is strong but currently local to merge-facing behavioral evidence (`research_only`)

## Interpretation

Route 2 now has a stable core centered on workflow invariants and aggregation seam logic. Middle-state claims are usable with guardrails, while optionality portability and some behavioral transfer claims remain explicitly non-canonical.
