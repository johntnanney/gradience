# CPU Rank-Proxy Validation Design

## Study Question
Can Gradience's cheap spectral rank suggestions approximate useful layerwise structure and produce competitive fixed-budget compression behavior versus lightweight offline proxy targets on small encoder adapters?

## Scope
- CPU-only
- Small encoder adapters in current Gradience regime
- Classification datasets already used in field-trial/evidence flows
- Shared base-family focus (prefer DistilBERT)

## Methods Compared
- Gradience spectral policies:
  - `energy_90`
  - `knee`
  - `erank`
  - `oht`
  - `stable_rank_ceil`
- Proxy signals:
  - per-layer gradient norm
  - per-layer ablation sensitivity
- Compression baselines:
  - uniform matched-budget
  - random matched-budget

## Fixed-Budget Protocol
- Define a target parameter budget as a ratio of original LoRA parameter count.
- Allocate per-layer ranks under that budget with integer rank constraints and per-layer max-r caps.
- Evaluate compressed behavior under matched budgets.

## Outputs
- `allocation_comparison_table.{json,md}`
- `compression_evaluation_table.{json,md}`
- `disagreement_memo.md`
- `study_summary.json`
- strategy summary note in `docs/strategy/`

## Guardrails
- No adaptive training-loop reproduction.
- No decoder expansion in this pass.
- No policy escalation claims from this run alone.
- Treat results as bounded evidence for advisor-style usefulness.

