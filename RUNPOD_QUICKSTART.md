# RunPod Quickstart for Mechanism Testing

**Goal**: Test whether audit-guided per-layer rank patterns provide real benefit beyond compression through controlled shuffled experiments.

## 🚀 One-Line Setup

```bash
# SSH into RunPod, then run:
curl -sSL https://raw.githubusercontent.com/your-username/gradience/main/experiments/runpod_launcher.sh | bash
```

## 📋 Manual Setup (if needed)

### 1. Create RunPod Instance
- **GPU**: RTX 4090 (24GB) or A100 (40GB+) 
- **Template**: PyTorch 2.1+ or Ubuntu with CUDA
- **Storage**: 100GB+

### 2. Connect and Setup
```bash
# SSH to your instance
ssh root@<runpod-ip> -p <port>

# Clone repository
git clone https://github.com/your-username/gradience.git
cd gradience

# Run automated setup
./experiments/runpod_launcher.sh
```

### 3. HuggingFace Login
When prompted:
1. Get token from: https://huggingface.co/settings/tokens
2. Ensure access to: `mistralai/Mistral-7B-v0.1`
3. Enter token in CLI

### 4. Launch Experiment
Choose option when setup completes:
- **Quick Test**: 30 min validation run
- **Full Experiment**: 6-8 hour 3-seed run  
- **Background**: Run in tmux/nohup

## 📊 Expected Results

After 6-8 hours, you'll get statistical analysis showing:

### Success Case (Real Mechanism):
```
🎯 OVERALL CONCLUSION
✅ HYPOTHESIS CONFIRMED: Audit-guided ranks provide real benefit beyond heterogeneity

📊 PERFORMANCE SUMMARY
probe               : 0.650 ± 0.010
per_layer           : 0.675 ± 0.008  
per_layer_shuffled  : 0.660 ± 0.012

🧬 MECHANISM BENEFIT TEST
per_layer vs per_layer_shuffled: +0.015 ± 0.008
✓ PASS: Audit-guided placement provides real benefit
```

### Null Case (Heterogeneity Only):
```
🎯 OVERALL CONCLUSION  
⚠️ PARTIAL SUCCESS: Compression works, but no clear mechanism benefit

📊 PERFORMANCE SUMMARY
probe               : 0.650 ± 0.010
per_layer           : 0.672 ± 0.008
per_layer_shuffled  : 0.670 ± 0.012

🧬 MECHANISM BENEFIT TEST
per_layer vs per_layer_shuffled: +0.002 ± 0.010
❌ FAIL: No significant mechanism benefit detected
```

## 🔧 Troubleshooting

**CUDA OOM**: Reduce `per_device_train_batch_size` in config
**Download fails**: Check HF token and model access
**Disk space**: Monitor `/workspace` usage during setup

## 💰 Cost Estimate

**RTX 4090**: ~$6-7 total (setup + 8hr experiment)  
**A100**: ~$18-20 total (faster but pricier)

## 📁 Results Location

```
/workspace/mechanism_test_results/
├── seed_42/bench.json          # Individual results
├── seed_43/bench.json
├── seed_44/bench.json  
├── aggregated_results.json     # Combined statistics
└── mechanism_analysis.json     # Hypothesis testing
```

Download with: `scp root@<runpod-ip>:/workspace/mechanism_test_results.tar.gz ./`

---

**Scientific Impact**: This experiment tests a fundamental question in LoRA compression - whether audit-guided rank placement behaves like adaptive regularization or if any heterogeneity is sufficient. The shuffled control isolates the mechanism cleanly.