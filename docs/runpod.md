# RunPod Operations Guide

**Complete survival guide for running Gradience on RunPod infrastructure.**

## 🚀 Quick Start (TL;DR)

```bash
# Essential first step - prevents 90% of RunPod issues
source /workspace/gradience/scripts/runpod/env.sh

# Then run gradience normally
python -m gradience.bench.run_bench --config configs/your_config.yaml --output results
```

## 📦 RunPod Storage Architecture

Understanding RunPod's dual-disk setup is critical:

| Mount Point | Size | Purpose | Speed | Persistence |
|-------------|------|---------|--------|-------------|
| `/root/` | 10-50GB | System disk | Fast SSD | ❌ Ephemeral |
| `/workspace/` | 100GB-1TB+ | Data disk | Variable | ✅ Persistent |

**Key insight**: Models and datasets MUST go to `/workspace/` to avoid quota issues.

## 🎯 Environment Variables (HuggingFace Cache)

### Current Standards (2024+)
```bash
export HF_HOME=/workspace/hf_cache/hf_home           # Primary cache directory
export HF_HUB_CACHE=/workspace/hf_cache/hub         # Model weights cache  
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets # Dataset cache
export TORCH_HOME=/workspace/hf_cache/torch         # PyTorch cache
```

### Legacy Variables (Still Supported)
```bash
export TRANSFORMERS_CACHE=/workspace/hf_cache/hub   # Deprecated but works
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub # Old name for HF_HUB_CACHE
```

### ❌ Deprecated (Avoid)
```bash
export TRANSFORMERS_CACHE=/root/.cache/transformers # Will fill system disk
```

## 🛠️ Setup Procedures

### Option 1: Automated Setup (Recommended)
```bash
# Clone and setup in one go
cd /workspace
git clone https://github.com/johntnanney/gradience.git
cd gradience
source scripts/runpod/env.sh  # Sets up all cache dirs + env vars
pip install -e ".[hf,dev]"
```

### Option 2: Manual Setup
```bash
# Set cache locations BEFORE any Python imports
export HF_HOME=/workspace/hf_cache/hf_home
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TORCH_HOME=/workspace/hf_cache/torch

# Create directories
mkdir -p /workspace/hf_cache/{hf_home,hub,datasets,torch}

# Verify setup
env | grep -E "HF_|TORCH_"
```

### Option 3: Emergency Recovery
```bash
# If /root/ already filled up
mkdir -p /workspace/hf_cache/{hf_home,hub,datasets}

# Move existing cache
mv /root/.cache/huggingface/* /workspace/hf_cache/hf_home/ 2>/dev/null || true
mv /root/.cache/torch/* /workspace/hf_cache/torch/ 2>/dev/null || true

# Clean up system disk
rm -rf /root/.cache/huggingface/*
rm -rf /root/.cache/torch/*

# Set new environment
source /workspace/gradience/scripts/runpod/env.sh
```

## 🔧 Persistent Configuration

### Automatic Setup on Pod Start
Add to `/root/.bashrc`:

```bash
# RunPod persistent cache setup
if [ -d "/workspace/gradience" ]; then
    source /workspace/gradience/scripts/runpod/env.sh
fi
```

### Manual .bashrc Entry
```bash
# HuggingFace cache configuration for RunPod
export HF_HOME=/workspace/hf_cache/hf_home
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TORCH_HOME=/workspace/hf_cache/torch
```

## 🚨 Common Issues & Solutions

### Issue 1: "No space left on device" during model download

**Symptoms:**
- Error during `transformers` model loading
- `/root/` filesystem at 100%
- Downloads failing mid-process

**Solution:**
```bash
# Check what's using space
df -h /root
du -sh /root/.cache/* 2>/dev/null | sort -hr

# Emergency cleanup
rm -rf /root/.cache/huggingface/*
rm -rf /root/.cache/torch/*
source /workspace/gradience/scripts/runpod/env.sh
```

### Issue 2: Models downloading to wrong location

**Symptoms:**
- Cache environment set but models still go to `/root/`
- Environment variables ignored

**Root Cause:** Environment must be set BEFORE importing HuggingFace libraries.

**Solution:**
```bash
# WRONG - too late
python -c "
import transformers
import os
os.environ['HF_HUB_CACHE'] = '/workspace/hf_cache/hub'  # Ignored!
"

# CORRECT - env first
export HF_HUB_CACHE=/workspace/hf_cache/hub
python -c "import transformers"  # Uses correct cache
```

### Issue 3: Jupyter Notebook cache issues

**Problem:** Jupyter kernels don't inherit shell environment variables.

**Solution:** Set in notebook cell BEFORE imports:
```python
import os
os.environ['HF_HOME'] = '/workspace/hf_cache/hf_home'
os.environ['HF_HUB_CACHE'] = '/workspace/hf_cache/hub' 
os.environ['HF_DATASETS_CACHE'] = '/workspace/hf_cache/datasets'

# Now safe to import
import transformers
from datasets import load_dataset
```

### Issue 4: Safetensors corruption after disk full

**Symptoms:**
- `incomplete metadata` errors
- `EOFError` during model loading
- Corrupted `.safetensors` files

**Solution:**
```bash
# Clear corrupted cache completely
rm -rf /workspace/hf_cache/hub/models--*
source /workspace/gradience/scripts/runpod/env.sh
# Re-download will start fresh
```

### Issue 5: Multiple Python processes fighting over cache

**Symptoms:**
- Random download failures
- Partial model files
- Inconsistent cache state

**Solution:** Use file locking (automatic in newer transformers):
```bash
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_ENABLE_HF_TRANSFER=1  # Faster downloads
```

## 📊 Monitoring & Maintenance

### Disk Usage Monitoring
```bash
# Check system disk (should stay <80%)
df -h /root | tail -1

# Check data disk  
df -h /workspace | tail -1

# Find largest cache items
du -sh /workspace/hf_cache/* | sort -hr
```

### Cache Cleanup
```bash
# Clean old model versions (keep only latest)
find /workspace/hf_cache/hub -name "*.json" -mtime +30 -delete

# Remove temporary download files
find /workspace/hf_cache -name "*.tmp*" -delete
find /workspace/hf_cache -name ".locks" -type d -exec rm -rf {} + 2>/dev/null || true
```

### Health Check Script
```bash
#!/bin/bash
# Save as scripts/runpod/health_check.sh

echo "=== RunPod Health Check ==="

# Check disk space
echo "Disk usage:"
df -h /root /workspace | tail -2

# Check cache environment
echo -e "\nCache configuration:"
env | grep -E "HF_|TORCH_" | sort

# Check cache sizes
echo -e "\nCache sizes:"
du -sh /workspace/hf_cache/* 2>/dev/null | sort -hr | head -5

# Verify writable
echo -e "\nCache write test:"
test_file="/workspace/hf_cache/write_test_$$"
if touch "$test_file" 2>/dev/null; then
    rm "$test_file"
    echo "✅ Cache directory writable"
else
    echo "❌ Cache directory not writable"
fi
```

## ⚡ Performance Optimizations

### Download Acceleration
```bash
# Use HF transfer for faster downloads (>100MB files)
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer

# Parallel downloads
export HF_HUB_DOWNLOAD_PARALLEL=1
```

### Memory Management
```bash
# Reduce memory pressure during large model downloads
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HUB_CACHE_MAX_SIZE=50GB  # Limit cache size
```

### Network Optimization
```bash
# Increase download timeouts for large models
export HF_HUB_DOWNLOAD_TIMEOUT=3600  # 1 hour timeout
export REQUESTS_TIMEOUT=300           # 5 minute request timeout
```

## 🔐 Security Considerations

### File Permissions
```bash
# Ensure cache is readable by all processes
chmod -R 755 /workspace/hf_cache/
```

### Token Management
```bash
# Use HF tokens for private models
huggingface-cli login --token your_token
# Token stored in /workspace/hf_cache/hf_home/token (persistent)
```

## 📈 Best Practices

1. **Always set environment BEFORE Python imports**
2. **Use the automated setup script** (`scripts/runpod/env.sh`)  
3. **Monitor `/root/` disk usage** (should stay <80%)
4. **Clean up failed downloads** regularly
5. **Use artifact hygiene** (`keep_adapter_weights: false`) to save space
6. **Add setup to `.bashrc`** for persistence across pod restarts

## 🔬 Advanced Debugging

### Cache Introspection
```python
# Check current cache locations from Python
import os
from pathlib import Path

print("HF_HOME:", os.environ.get('HF_HOME', 'NOT SET'))
print("HF_HUB_CACHE:", os.environ.get('HF_HUB_CACHE', 'NOT SET'))

# Check actual transformers cache
import transformers
print("Transformers cache dir:", transformers.utils.TRANSFORMERS_CACHE)
```

### Network Diagnostics
```bash
# Test HuggingFace Hub connectivity
curl -I https://huggingface.co/api/models/distilbert-base-uncased

# Check download speed
time huggingface-cli download microsoft/DialoGPT-small --cache-dir /tmp/speed_test
```

### Process Monitoring
```bash
# Monitor active downloads
lsof +D /workspace/hf_cache | grep -v "DEL"

# Check Python processes using cache
ps aux | grep python | head -5
```

---

## 📚 References

- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [RunPod Storage Documentation](https://docs.runpod.io/pods/storage)
- [Gradience Artifact Hygiene Guide](../bench/README.md#artifact-hygiene)

---

💡 **Pro Tip**: Always run `source scripts/runpod/env.sh` as the first command in any new RunPod session. This single command prevents 90% of storage-related issues.