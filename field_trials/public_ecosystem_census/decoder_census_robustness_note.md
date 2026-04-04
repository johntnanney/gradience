# Decoder Census Robustness Note

- Date (UTC): 2026-04-04T00:28:00.196640+00:00
- Full task-balanced cohort n: 36

## Full Cohort Baseline

- Mean architecture eta-sq: 0.143
- Mean task eta-sq: 0.3359
- Architecture metrics > 0.10: 5
- Task metrics > 0.10: 5

## Slice 1: High-Confidence Task Labels

- n: 33
- Mean architecture eta-sq: 0.164
- Mean task eta-sq: 0.332
- Architecture metrics > 0.10: 5

## Slice 2: Rank-Matched (Modal Rank)

- Modal rank: 8
- n: 11
- Mean architecture eta-sq: 0.082
- Mean task eta-sq: 0.1108
- Architecture metrics > 0.10: 3

## Slice 3: Dominant Task Downweighted

- Dominant task: chat_instruct
- Downweighted from 13 to 10
- n: 33
- Mean architecture eta-sq: 0.1446
- Mean task eta-sq: 0.3311

## Robustness Check Outcome

- High-confidence labels: True
- Rank-matched architecture detectable: True
- Downweighted dominant task: True

Architecture effects remain visible across the high-confidence and dominant-task-downweighted slices. Rank-matched slice remains interpretable but is lower-power due to small n.

## Guardrails

- Observational found-artifact setting only.
- Task labels remain partially noisy.
- Rank-matched subset is small and not a causal result.
- No policy or product changes from this note.
