# Module-Type Asymmetry Analysis

## Aggregate Sign Test

- Pairs tested: 8
- Valid (non-tied): 8
- Attention < MLP: 2
- MLP < Attention: 6
- Proportion attn < MLP: 0.25
- Replicates encoder pattern: False

## By Architecture Family

| Family | Attn<MLP | MLP<Attn | Ties | Prop | Mean diff | Direction |
|--------|---------|---------|------|------|-----------|-----------|
| llama | 2 | 6 | 0 | 0.25 | 0.0832 | MLP < attn |

## Subtype-Level Breakdown

| Subtype | N layers | SR mean | SR std | Util mean |
|---------|----------|---------|--------|-----------|
| attn_other | 2688 | 1.9924 | 0.9459 | 0.1690 |
| mlp_other | 816 | 1.4996 | 0.578 | 0.1129 |

## Architectural Attribution

- **llama**: mean asymmetry magnitude = 0.197 (std = 0.0915, n = 8)

## Guardrails

- Subtype labels (q/k/v/o, gate/up/down) inferred from layer name patterns; may not match all architectures.
- Asymmetry direction is an empirical finding, not a causal claim about architecture design.
- Small sample sizes per family limit statistical power for architectural attribution.
