# RunPod GPU Experiments for Gradience Validation

## Overview

Complete setup for validating Gradience's new audit-driven compression features on RunPod GPU infrastructure. Includes rapid iteration configs and production-quality benchmark configs for public demonstration.

## 🎯 New Features to Validate

- **Second Rung Logic**: Audit-driven compression candidates (Tier A/B)
- **Self-Contained Aggregates**: Complete per-seed, per-candidate breakdowns  
- **Engineering Hygiene**: Progress tracking, heartbeat monitoring, artifact cleanup
- **Decision Transparency**: Full audit pipeline with UDR computation

## 📋 Configuration Files

### DistilBERT + SST-2 (Classification)
- **`runpod_distilbert_sst2_dev.yaml`**: Single-seed rapid iteration (4 min, $0.06)
- **`runpod_distilbert_sst2.yaml`**: Multi-seed validation (15 min, $0.23)

### Mistral-7B + GSM8K (Mathematical Reasoning)  
- **`runpod_mistral_gsm8k_dev.yaml`**: Single-seed development (30 min, $0.45)
- **`runpod_mistral_gsm8k_public.yaml`**: Production demo (150 min, $2.25)

## 🚀 Quick Deployment

```bash
# 1. Upload gradience codebase to /workspace/gradience
# 2. Run setup script
chmod +x /workspace/gradience/scripts/setup_runpod.sh
/workspace/gradience/scripts/setup_runpod.sh

# 3. Start with rapid iteration
python -m gradience.bench.run_bench \
  --config runpod_distilbert_sst2_dev.yaml \
  --output /workspace/experiments/distilbert_dev
```

## 📊 Expected Second Rung Behavior

### When Second Rung Logic Triggers
**Conditions**: 
- Low utilization (< 0.55) OR high rank efficiency (suggested_rank ≤ 0.75 * probe_rank)
- Very low utilization (< 0.30) AND low stable rank (≤ probe_rank/4)

**Expected Candidates**:
- **Tier A (Moderate)**: uniform_r* variants for moderate compression
- **Tier B (Aggressive)**: uniform_r* variants for aggressive compression

### When It Doesn't Trigger
- High utilization AND high stable rank → stick to policy-based candidates only
- Still demonstrates audit-driven decision making in decision trace

## 📈 Key Artifacts Generated

### Per-Seed Artifacts
- **`compression_configs.json`**: Decision trace with second rung candidates
- **`bench.json`**: Complete compression analysis with audit metrics
- **`progress.txt`** & **`heartbeat.log`**: Engineering hygiene monitoring

### Multi-Seed Aggregates  
- **`bench_aggregate.json`**: Statistical analysis across seeds
- **`bench_aggregate.md`**: Human-readable benchmark report
- **Detailed results section**: Self-contained per-seed, per-candidate breakdown

## 🔍 Analysis Tools

```bash
# Automated analysis for public demonstration
python scripts/analyze_public_artifacts.py /workspace/experiments/mistral_public

# Raw data inspection
cat /workspace/experiments/mistral_public/bench_aggregate.json
cat /workspace/experiments/mistral_public/seed_42/compression_configs.json
```

## 💡 Recommended Workflow

1. **Start Fast**: `runpod_distilbert_sst2_dev.yaml` for rapid feature validation
2. **Expand**: `runpod_distilbert_sst2.yaml` for multi-seed statistical robustness  
3. **Scale Up**: `runpod_mistral_gsm8k_dev.yaml` for complex reasoning validation
4. **Production**: `runpod_mistral_gsm8k_public.yaml` for public demonstration

## 🎯 Validation Checklist

### Technical Features
- [ ] Second rung logic triggers with appropriate audit thresholds
- [ ] Decision trace captures all candidate generation logic
- [ ] Self-contained aggregates include complete per-seed details
- [ ] Progress tracking prevents stuck runs
- [ ] Heartbeat monitoring maintains SSH sessions

### Mathematical Reasoning (Mistral + GSM8K)
- [ ] Probe quality gate ensures meaningful compression evaluation  
- [ ] UDR computation provides decision transparency
- [ ] Multi-seed analysis shows statistical robustness
- [ ] Compression effectiveness preserves reasoning capability

### Public Demonstration
- [ ] Artifacts are self-contained and interpretable
- [ ] Decision traces show transparent audit-driven logic
- [ ] Statistical analysis demonstrates reproducibility  
- [ ] Engineering hygiene shows production readiness

## 💰 Cost Summary

| Configuration | Runtime | Cost | Purpose |
|---|---|---|---|
| DistilBERT Dev | 4 min | $0.06 | Rapid iteration |
| DistilBERT Full | 15 min | $0.23 | Multi-seed validation |
| Mistral Dev | 30 min | $0.45 | Reasoning validation |
| Mistral Public | 150 min | $2.25 | Production demo |

**Total for complete validation**: ~$3.00