# Adapter Landscape Assessment — Pilot Readiness

## What we found

15 public LoRA adapters verified across two backbone families, covering 7 distinct task types. All load successfully via `PeftConfig.from_pretrained()`. The HuggingFace Hub has enough real PEFT/LoRA adapters on small encoders to populate all three pilot inventories.

## Critical observation: LoRA configuration heterogeneity

The verified adapters show substantial variation in LoRA configuration — this is not a controlled panel.

### Rank and alpha variation

| Adapter | Backbone | r | alpha | target_modules |
|---------|----------|---|-------|----------------|
| muneeb-ai (IMDB) | distilbert | 4 | 32 | q,k,v,out,lin1,lin2,pre_cls,cls |
| jmeneu (IMDB) | distilbert | 1 | 32 | q only |
| RAJESHCHAUHAN101 | distilbert | 4 | 32 | q only |
| myselfmankar (SST-2) | distilbert | 16 | 32 | q,v |
| NightPrince (SST-2) | distilbert | 8 | 32 | q,v |
| TransferGraph (emotion) | distilbert | 1 | 1 | q,v |
| TransferGraph (ag_news/JB173) | distilbert | 1 | 1 | q,v |
| TransferGraph (hate/jaesun) | distilbert | 1 | 1 | q,v |
| TransferGraph (hate/Aureliano) | distilbert | 1 | 1 | q,v |
| TransferGraph (ag_news/phailyoor) | distilbert | 1 | 1 | q,v |
| TransferGraph (hate/roberta) | roberta | 1 | 1 | query,value |
| TransferGraph (ag_news/roberta) | roberta | 1 | 1 | query,value |
| TransferGraph (irony/roberta) | roberta | 1 | 1 | query,value |
| TransferGraph (formality→ag_news) | roberta | 1 | 1 | query,value |
| yuuhan (MNLI) | roberta | 8 | 16 | query,value |

### What this means for the field trial

**The good:** This heterogeneity is realistic. Real adapter inventories will have mixed LoRA configurations. Testing Gradience on heterogeneous adapters is exactly the right thing to do — it's where the product needs to work.

**The concern:** The TransferGraph adapters all use r=1, alpha=1, targeting only query+value. These are extremely low-rank adapters — essentially rank-1 perturbations. Gradience's spectral analysis may not find much geometric structure to work with at r=1. This is a genuine test: does Gradience produce useful outputs when the adapters are very small?

**Base model heterogeneity within the TransferGraph series:** The `base_model_name_or_path` field in these adapters points to intermediate fine-tuned models (e.g., `jaesun/distilbert-base-uncased-finetuned-cola`, `JB173/distilbert-base-uncased-finetuned-emotion`), not to the raw distilbert-base-uncased. These are LoRA adapters applied *on top of* already-fine-tuned models. This is a transfer-learning chain: base → task-A fine-tune → LoRA for task-B. Gradience has never been tested on this pattern. It may or may not matter for spectral analysis — the LoRA delta is still low-rank — but it's worth noting.

### Configuration clusters

- **TransferGraph series:** r=1, alpha=1, q+v only. Consistent methodology, varied tasks and source models.
- **Community adapters (muneeb-ai, jmeneu, myselfmankar, etc.):** r=1–16, alpha=32, varied module targets. No consistent methodology.
- **yuuhan MNLI:** r=8, alpha=16, q+v. The only roberta-base adapter not from TransferGraph.

## Pilot inventory composition

### Pilot 1 — Same-task control (4 adapters, distilbert)
- 2× IMDB sentiment (different authors, different LoRA configs: r=4 vs r=1)
- 1× text classification (distilbert, r=4)
- 1× emotion classification (TransferGraph, r=1)
- **Key test:** Does Gradience correctly identify the IMDB pair as same-task safe despite different LoRA ranks?

### Pilot 2 — Mixed task (5 adapters, roberta-base)
- 2× AG News (TransferGraph, different lineage)
- 1× hate speech (TransferGraph)
- 1× irony detection (TransferGraph)
- 1× MNLI (yuuhan, r=8 — the outlier)
- **Key test:** Task-boundary detection across tweet-domain tasks. Does the MNLI adapter get flagged?

### Pilot 3 — Large mixed task (9 adapters, distilbert)
- 4× sentiment (2 IMDB, 2 SST-2 — different authors and ranks)
- 1× emotion (TransferGraph)
- 2× AG News (TransferGraph, different source models)
- 2× hate speech (TransferGraph, different source models)
- **Key test:** 36 candidate pairs. Does large-inventory mode produce useful region summaries?

## Trial readiness

**Status: ready to proceed.**

All adapters are public, loadable, and free. The heterogeneity is a feature, not a bug — it tests Gradience in realistic conditions. The r=1 adapters are an interesting edge case that may reveal whether Gradience needs a minimum-rank threshold.

## Next step

Download adapter weights for all 15 adapters and run Gradience preflight on Pilot 1.
