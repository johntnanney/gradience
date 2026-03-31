# External Use-Case Scan (Route 2 Workstream 5)

Date: 2026-03-31  
Status: initial scan; internal evidence-backed prioritization

## Scope

This scan maps plausible external workflow classes to current Route 2 capabilities and gates.

Gates used:

1. representation clarity
2. workflow usefulness
3. decision-relative interpretability
4. practical confirmation

## Candidate use cases

| Use case | Problem shape | Gate 1 | Gate 2 | Gate 3 | Gate 4 | Current fit |
|---|---|---|---|---|---|---|
| Shared-base checkpoint inventory triage (classification teams) | Many checkpoints, weak metadata, limited eval budget | high | high | high | medium | strongest current pull |
| Adapter routing/confusability hygiene (multi-adapter serving) | Need to identify confusable pairs before routing failures | high | medium | high | medium | promising, still pilot-scale |
| Adapter/checkpoint portfolio hygiene (dedup + review ordering) | Periodic cleanup of variant libraries | medium | high | medium | low | plausible, needs workflow owner |
| Cross-architecture checkpoint comparison | Mixed bases/backbones in one inventory | low | low | low | low | defer |

## Pulled use case recommendation

Pulled next-use-case candidate: **shared-base checkpoint inventory triage**.

Why:

- it already passes representation and interpretability gates in bounded scope,
- it has repeatable workflow outputs (T01/T02 structure),
- it aligns with conservative narrowing and QA-first behavior,
- it is broader than merge preflight without requiring platform refactor.

Current alpha package (canonical demo path):

- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html`
- `docs/examples/checkpoint-triage-alpha-workflow.md`
- `docs/strategy/checkpoint_triage_alpha_scope.md`

## Near-term validation step

Run one additional checkpoint inventory trial with a second same-family branch (for example SST-2 with IMDB or Amazon polarity) under the same CPU protocol and normalized artifact layout.

Decision rule:

- if same-family behavior remains conservative but useful, keep checkpoint triage as the first stabilized broadened workflow,
- if behavior is inconsistent or opaque, keep checkpoint triage experimental and pause broader pull.

## Not pulled now

- cross-architecture inventory workflows,
- decoder/generation workflows,
- anything that requires checkpoint merge execution.

These fail at least one Route 2 gate under current evidence.
