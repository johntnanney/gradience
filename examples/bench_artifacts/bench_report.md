# Gradience Bench v0.1

- **Model:** microsoft/DialoGPT-small
- **Task:** gsm8k/main
- **Date:** 2024-01-15 10:30:45 UTC
- **Git:** v0.4.4 (a1b2c3d4)

## 🎯 Probe Quality Gate

| Metric | Value | Threshold | Status |
|--------|--------|-----------|--------|
| **eval_exact_match** | **0.216** | ≥ 0.15 | ✅ **PASSED** |

The probe achieves adequate quality for compression validation.

## 📊 Probe Results

- **Rank:** 16
- **Parameters:** 442,368
- **Accuracy:** 21.56%
- **Utilization:** 72.34%
- **Energy Rank (P50):** 6.8
- **Energy Rank (P90):** 11.2

## 🗜️ Compression Results

### uniform_median (r=8)
- **Parameters:** 221,184 (50.0% reduction)
- **Accuracy:** 20.89% (Δ = -0.67%)
- **Verdict:** ✅ **PASS**
- **Efficiency:** 96.9% accuracy retention

### uniform_p90 (r=12)  
- **Parameters:** 331,776 (25.0% reduction)
- **Accuracy:** 21.34% (Δ = -0.22%)
- **Verdict:** ✅ **PASS**
- **Efficiency:** 98.9% accuracy retention

## 🏆 Summary

- **Recommendations validated:** ✅ Yes
- **Best compression:** uniform_median (50% parameter reduction)
- **Quality assessment:** Conservative compression maintains high accuracy

### Notes
- Probe quality gate passed (0.216 ≥ 0.15)
- Both compression variants achieved PASS verdict  
- Median rank suggestion (r=8) provides 50% parameter reduction
- Conservative compression maintains 96.9% of probe accuracy

## ⚙️ Configuration

<details>
<summary>Expand configuration details</summary>

```yaml
model:
  name: microsoft/DialoGPT-small
  type: causal_lm

task:
  dataset: gsm8k
  subset: main
  profile: gsm8k_causal_lm
  eval_max_samples: 100

lora:
  probe_r: 16
  alpha: 16
  target_modules: [c_attn, c_proj]
  dropout: 0.1

train:
  train_samples: 500
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 3
  seed: 42
```

</details>

## 🖥️ Environment

- **Python:** 3.10.12
- **PyTorch:** 2.1.0+cu118  
- **GPU:** NVIDIA GeForce RTX 4090 (24GB)
- **CUDA:** 11.8

---

*Generated on 2024-01-15 10:30*