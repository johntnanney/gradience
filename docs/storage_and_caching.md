# Storage and Caching Configuration

**TL;DR: Don't burn a day on "Disk quota exceeded" chaos.**

This document explains how to configure disk/cache behavior for HuggingFace, PyTorch, and Gradience to avoid storage issues in cloud environments, RunPod, and local development.

## 🚨 Common Storage Pain Points

1. **HuggingFace models cache to `~/.cache/`** by default (can fill up system disk)
2. **Datasets download to random temp locations** 
3. **Multiple models = multiple downloads** even if same base model
4. **RunPod `/workspace/` has more space than `/root/`** but defaults go to `/root/.cache/`
5. **"Disk quota exceeded"** kills training runs hours into the process

## ⚡ Quick Fix (Copy-Paste This)

**Add to your shell profile (`.bashrc`, `.zshrc`) or run before each session:**

```bash
# HuggingFace caching - redirect to workspace with more space
export HF_HUB_DISABLE_XET=1                    # Disable XET protocol (can cause issues)
export HF_HUB_ENABLE_HF_TRANSFER=0             # Disable hf_transfer by default (enable manually if needed)
export HF_HOME=/workspace/hf_cache/hf_home     # Main HF cache directory
export HF_HUB_CACHE=/workspace/hf_cache/hub    # Model downloads
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets  # Dataset downloads

# PyTorch caching
export TORCH_HOME=/workspace/torch_cache       # PyTorch models and weights

# Gradience-specific
export GRADIENCE_CACHE_DIR=/workspace/gradience_cache  # Gradience caches (base norms, etc.)

# Create directories (run once)
mkdir -p /workspace/hf_cache/{hf_home,hub,datasets}
mkdir -p /workspace/torch_cache
mkdir -p /workspace/gradience_cache
```

## 📋 Environment Variables Explained

| Variable | Purpose | Default | Recommended |
|----------|---------|---------|-------------|
| `HF_HOME` | Main HuggingFace directory | `~/.cache/huggingface` | `/workspace/hf_cache/hf_home` |
| `HF_HUB_CACHE` | Model downloads cache | `$HF_HOME/hub` | `/workspace/hf_cache/hub` |
| `HF_DATASETS_CACHE` | Datasets cache | `~/.cache/huggingface/datasets` | `/workspace/hf_cache/datasets` |
| `TORCH_HOME` | PyTorch cache | `~/.cache/torch` | `/workspace/torch_cache` |
| `HF_HUB_DISABLE_XET` | Disable XET protocol | `0` | `1` (prevents issues) |
| `HF_HUB_ENABLE_HF_TRANSFER` | Use hf_transfer | `0` | `0` (enable manually when needed) |

## 🔧 Platform-Specific Configurations

### RunPod / Cloud Instances

```bash
# RunPod typically has:
# /root/        - Limited space (system disk)
# /workspace/   - Large space (data disk)

export HF_HOME=/workspace/hf_cache/hf_home
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TORCH_HOME=/workspace/torch_cache
export GRADIENCE_CACHE_DIR=/workspace/gradience_cache

# Check available space
df -h /workspace
df -h /root
```

### Local Development

```bash
# For local development, you might prefer keeping in user directory
# but in a dedicated subdirectory for easy cleanup

export HF_HOME=$HOME/.cache/gradience/hf_home
export HF_HUB_CACHE=$HOME/.cache/gradience/hub
export HF_DATASETS_CACHE=$HOME/.cache/gradience/datasets
export TORCH_HOME=$HOME/.cache/gradience/torch
export GRADIENCE_CACHE_DIR=$HOME/.cache/gradience

mkdir -p ~/.cache/gradience/{hf_home,hub,datasets,torch}
```

### Docker Containers

```dockerfile
# In your Dockerfile or docker-compose.yml
ENV HF_HOME=/app/cache/hf_home
ENV HF_HUB_CACHE=/app/cache/hub  
ENV HF_DATASETS_CACHE=/app/cache/datasets
ENV TORCH_HOME=/app/cache/torch
ENV GRADIENCE_CACHE_DIR=/app/cache/gradience

# Create cache directories
RUN mkdir -p /app/cache/{hf_home,hub,datasets,torch,gradience}

# Mount cache directory as volume for persistence
VOLUME ["/app/cache"]
```

## 🗂️ What Gets Cached Where

### HuggingFace Hub Cache (`HF_HUB_CACHE`)
- **Model weights**: `transformers`, `peft` adapters
- **Tokenizers**: vocabulary files, special tokens
- **Size**: 1-50GB depending on models used
- **Content**: `models--org--modelname/` directories

### Datasets Cache (`HF_DATASETS_CACHE`) 
- **Datasets**: GLUE, GSM8K, custom datasets
- **Processed data**: tokenized, preprocessed versions
- **Size**: 100MB-10GB per dataset
- **Content**: `downloads/` and processed dataset files

### PyTorch Cache (`TORCH_HOME`)
- **Pretrained models**: ResNet, CLIP, etc. from `torch.hub`
- **Checkpoint files**: model state dicts
- **Size**: Usually < 1GB unless using vision models

### Gradience Cache (`GRADIENCE_CACHE_DIR`)
- **Base model norms**: for UDR computation
- **Audit results**: cached analysis data
- **Size**: Usually < 100MB per project

## 💡 Pro Tips

### 1. Check Disk Usage
```bash
# See what's taking space
du -sh /workspace/hf_cache/*
du -sh ~/.cache/huggingface/*

# Monitor during downloads
watch -n 1 'df -h /workspace'
```

### 2. Clean Up Old Caches
```bash
# Clear HuggingFace cache (nuclear option)
rm -rf /workspace/hf_cache/hub/*
rm -rf /workspace/hf_cache/datasets/*

# Or use HuggingFace's built-in cleanup
python -c "from huggingface_hub import scan_cache_dir; scan_cache_dir().delete_revisions(lambda r: r.size_on_disk > 1e9)"
```

### 3. Share Caches Across Projects
```bash
# Symlink common caches across project directories
ln -s /workspace/hf_cache/hub ./project1/hf_cache
ln -s /workspace/hf_cache/hub ./project2/hf_cache
```

### 4. Enable Fast Downloads When Ready
```bash
# Only enable hf_transfer when you know you want it
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer  # Must install separately
```

## 🛠️ Gradience-Specific Caching

Gradience respects standard HuggingFace caching but adds:

### Base Model Norms Cache
```python
# Gradience caches base model norms for UDR computation
# Location: $GRADIENCE_CACHE_DIR/base_norms/
from gradience.vnext.audit import audit_lora_peft_dir

result = audit_lora_peft_dir(
    "path/to/adapter",
    base_model_id="mistralai/Mistral-7B-v0.1",
    base_norms_cache="/workspace/gradience_cache/base_norms"  # Explicit path
)
```

### Bench Artifacts
```bash
# Bench results go to specified output directory
gradience bench --config config.yaml --output-dir /workspace/bench_runs/

# Keep artifacts organized
/workspace/
├── bench_runs/           # Bench results
├── hf_cache/            # HuggingFace models & datasets  
├── torch_cache/         # PyTorch downloads
└── gradience_cache/     # Gradience-specific caches
```

## 🚨 Troubleshooting

### "No space left on device"
```bash
# Check what's using space
df -h
du -sh /root/.cache/* | sort -hr
du -sh /workspace/* | sort -hr

# Emergency cleanup
rm -rf ~/.cache/huggingface/hub/*
rm -rf ~/.cache/torch/*
```

### "Permission denied" writing to cache
```bash
# Fix ownership (common in Docker)
sudo chown -R $(whoami) /workspace/hf_cache/
chmod -R u+w /workspace/hf_cache/
```

### Cache variables not taking effect
```bash
# Check if variables are set
env | grep -E "(HF_|TORCH_|GRADIENCE_)"

# Make sure they're set before importing transformers
export HF_HOME=/workspace/hf_cache/hf_home
python -c "import transformers; print(transformers.file_utils.default_cache_path)"
```

### Multiple downloads of same model
```bash
# This indicates cache isn't working - check:
ls -la /workspace/hf_cache/hub/
# Should see: models--mistralai--Mistral-7B-v0.1/

# Force cache location
python -c "
import os
os.environ['HF_HUB_CACHE'] = '/workspace/hf_cache/hub'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
print(f'Cache location: {os.environ.get(\"HF_HUB_CACHE\")}')
"
```

## 📚 References

- [HuggingFace Hub Caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [Transformers Cache](https://huggingface.co/docs/transformers/installation#cache-setup) 
- [Datasets Caching](https://huggingface.co/docs/datasets/cache)
- [PyTorch Hub Cache](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved)

---

**Remember**: Set these environment variables BEFORE importing any HuggingFace libraries. They can't be changed after the libraries are loaded in the same Python process.