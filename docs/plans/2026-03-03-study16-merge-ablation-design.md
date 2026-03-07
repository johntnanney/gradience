# Study 16: End-to-End Merge Ablation

## Goal

Demonstrate that Gradience's spectral merge recommendations produce
measurably better merges than naive averaging, across adapter pairs
spanning the compatibility spectrum (SAFE, REDUNDANT, CONFLICTING,
IMBALANCED).

## Base Model

All pairs use **meta-llama/Llama-2-7b-hf** adapters from the Study 14
broader benchmarks dataset (n=10 audited adapters, 6 with 224 LoRA
layers, 3 with 64 layers).

## Adapter Pairs (6)

| # | Adapter A | Adapter B | Expected verdict |
|---|-----------|-----------|------------------|
| 1 | LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32 (math, r=16) | LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens (math, r=16) | SAFE/REDUNDANT |
| 2 | LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32 (math, r=16) | LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32 (code, r=16) | Moderate overlap |
| 3 | LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32 (code, r=16) | AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter (chat, r=8) | CONFLICTING |
| 4 | LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens (math, r=64) | AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter (chat, r=8) | IMBALANCED |
| 5 | LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32 (math, r=16) | McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised (classification, r=16) | CONFLICTING |
| 6 | shivanikerai/Llama-2-7b-chat-hf-adapter-cat-subcat-mapping-v2.0 (chat, r=16) | AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter (chat, r=8) | Mixed (same task, rank mismatch) |

## Merge Conditions

For each pair, up to 3 merge conditions:

1. **Naive** (`uniform_linear`): coefficients [0.5, 0.5], no trimming.
2. **Audit-aware** (`audit_aware`): per-layer strategy from `recommend_merge()`.
3. **Compress-first audit-aware**: pre-compress to energy-rank-90 targets
   before merging. Only runs when recommendation includes `compress_first=True`.

## Evaluation Metrics (Phase 1 -- CPU)

For each merged adapter:
- **Q_min**: worst-case retention across both source adapters
- **Dominance D**: asymmetry of retention between sources
- **retention_A, retention_B**: per-source retention
- **is_bad_merge**: Q_min < 0.95 OR D > 0.2
- **Merged adapter spectral audit**: utilisation, stable rank, energy rank
- **Per-layer reconstruction error**: from SVD refactoring (merge_result.json)

## Evaluation Metrics (Phase 2 -- GPU, deferred)

Optional `--eval-gpu` flag:
- Perplexity on WikiText-2 (model-agnostic)
- Task-specific lm-eval benchmarks (GSM8K, HumanEval, etc.)

## Output Structure

```
results/study16_merge_ablation/
  study16_merge_ablation.json
  pair_01_metamath_x_openwebmath16/
    audit/merge_audit.json
    naive/          (PEFT adapter + merge_result.json)
    recommended/    (PEFT adapter + merge_result.json)
    compressed/     (if applicable)
  pair_02_.../
  ...
```

## Script

`scripts/study16_merge_ablation.py`

Pipeline: download -> audit pairs -> plan merges -> execute merges ->
evaluate merged adapters -> produce summary JSON and comparison table.

## Success Criteria

Audit-aware merges should show:
- Higher Q_min than naive (better worst-case retention)
- Lower D than naive (less adapter dominance)
- Fewer is_bad_merge flags
- More balanced spectral energy in the merged adapter

The effect should be most pronounced on CONFLICTING and IMBALANCED pairs,
where naive averaging is expected to perform worst.
