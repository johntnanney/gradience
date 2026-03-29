# Field Note — Pilot 1: Same-Task Control

## Gradience stance (recorded before evaluation)

### Preflight summary

- 4 adapters, 6 pairs, **0 retained**, 100% reduction
- All 4 adapters classified `unknown_no_behavioral_eval`
- All 4 excluded: "missing behavioral evidence — low confidence"
- No same-task safe zones, no cross-task caution zones, no evaluate-first subset
- Summary line: "QA dominates this inventory; no credible same-task candidates remain."

### Pair-level detail

| Adapter A | Adapter B | Risk | Issue | Strategy | Advisory |
|-----------|-----------|------|-------|----------|----------|
| muneeb (IMDB, r=4) | jmeneu (IMDB, r=1) | low | none | linear | — |
| muneeb (IMDB, r=4) | RAJESH (txt, r=4) | low | none | linear | cross-task |
| muneeb (IMDB, r=4) | TG (emotion, r=1) | high | norm_imbalance | audit_aware | cross-task |
| jmeneu (IMDB, r=1) | RAJESH (txt, r=4) | medium | norm_imbalance | norm_equalized | cross-task |
| jmeneu (IMDB, r=1) | TG (emotion, r=1) | medium | norm_imbalance | norm_equalized | cross-task |
| RAJESH (txt, r=4) | TG (emotion, r=1) | medium | norm_imbalance | norm_equalized | cross-task |

### Adapter QA detail

| Adapter | Status | Rank | Layers | Util mean | Flags |
|---------|--------|------|--------|-----------|-------|
| muneeb (IMDB, r=4) | unknown_no_behavioral_eval | 4 | 42 | 0.308 | none |
| jmeneu (IMDB, r=1) | unknown_no_behavioral_eval | 1 | 6 | 0.907 | none |
| RAJESH (txt, r=4) | unknown_no_behavioral_eval | 4 | 6 | 0.232 | none |
| TG (emotion, r=1) | unknown_no_behavioral_eval | 1 | 12 | 0.917 | none |

## What happened

Gradience excluded the entire inventory. Every adapter was classified `unknown_no_behavioral_eval` because no behavioral scores (adapter_score, base_score) were provided. The preflight correctly noted that without behavioral evidence, no credible candidates remain.

This is technically correct behavior per the design. The trust/provenance policy requires behavioral evidence to make eligibility judgments, and none was provided.

## What Gradience got right

1. **The pair-level structural analysis is sensible.** The muneeb × jmeneu pair (both IMDB sentiment, same backbone) was rated `low` risk with `none` as dominant issue — the cleanest pair in the inventory, as expected. The r=4 × r=1 pairs correctly flagged `norm_imbalance` due to the rank/alpha disparity.

2. **Cross-task advisory detection worked.** 5 of 6 pairs received cross-task advisories. The only pair without one was muneeb × jmeneu (both IMDB) — correct, since those are genuinely same-task.

3. **The high-risk flag on muneeb × TG(emotion) is reasonable.** r=4/alpha=32 (all 8 module types including classifier) vs. r=1/alpha=1 (q+v only) is a substantial structural mismatch — 42 layers vs 12 layers, different module coverage.

## What Gradience got wrong or missed

1. **5/6 pairs got cross-task advisory, but muneeb × RAJESH should arguably be same-task.** RAJESH is also a distilbert text-classification adapter — likely IMDB or similar sentiment. Gradience flagged it as cross-task because the eval_dataset metadata was `unknown` vs. `imdb`. This is a metadata gap, not a real task boundary. The structural analysis (low risk, no dominant issue) correctly saw the similarity.

2. **The "exclude everything" outcome is unhelpful as workflow guidance.** If I'm a practitioner who assembled this inventory without behavioral scores, "exclude all of them" tells me nothing I didn't already know. The report would be more useful if it said something like: "No behavioral evidence available. Structural analysis suggests the muneeb × jmeneu pair is the strongest candidate if you proceed despite the evidence gap."

3. **Layer count discrepancy is confusing.** muneeb shows 42 layers (because it targets all 8 module types × 6 transformer layers, minus some), while jmeneu shows 6 (q_lin only × 6 layers). The n_layers field in QA artifacts doesn't distinguish "LoRA layers" from "transformer layers," which could mislead a practitioner.

## Product usefulness ratings

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Search reduction | **low** | Reduced to 0 retained — too aggressive, not useful |
| Interpretive clarity | **medium** | Pair-level analysis is clear; overall summary is a dead end |
| Trust usefulness | **low** | Trust policy dominated everything; no structural signal reached the practitioner |
| Report usefulness | **medium** | HTML report rendered cleanly, structure is good, but content is "nothing to see" |
| Large-inventory usefulness | n/a | Not a large inventory |

## Key observations for the trial

1. **The "no behavioral evidence → exclude everything" pathway needs a softer alternative.** In a field trial with public adapters that don't ship with eval scores, this pathway will fire on every inventory. The plan should either (a) run evals to generate behavioral evidence before preflight, or (b) test whether Gradience has a mode that falls back to structural-only assessment.

2. **Structural analysis underneath the QA layer is working.** The pair-level verdicts are reasonable. The problem is that the inventory-level policy overrides them entirely when behavioral evidence is missing.

3. **Norm imbalance from rank/alpha mismatch is the dominant structural signal.** 5 of 6 pairs show it. This is driven by the heterogeneous LoRA configs (r=1/alpha=1 vs r=4/alpha=32). In a real inventory, this would be a useful signal. In this trial, it's somewhat obvious.

4. **The TransferGraph adapter (r=1, alpha=1) vs community adapters (r=4, alpha=32) creates a natural norm cliff.** This is realistic — real inventories will have mixed LoRA configurations — but it means every mixed pair will flag norm_imbalance.

## Implication for remaining pilots

Before running Pilots 2 and 3, we should either:
- **Option A:** Run quick evals on the adapters to generate behavioral evidence, then re-run preflight. This tests the full pipeline.
- **Option B:** Feed synthetic behavioral evidence (adapter_score, base_score) into the audit to get past the evidence gate. This tests the structural analysis layer but not the trust layer.
- **Option C:** Accept that all pilots will hit the "exclude everything" pathway and focus field notes on the structural layer underneath.

Option A is the most informative but requires more compute. Option B is faster and still tests the product surface. Option C is the least effort but limits what the trial can learn.
