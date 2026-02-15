# Installation & Environments

**Complete installation guide to prevent common issues and get you running quickly.**

## TL;DR - Quick Install

```bash
# Core package (audit, monitor, compress)
pip install gradience

# Full ML stack (recommended for most users)
pip install "gradience[bench]"

# With separate PyTorch (advanced users)
pip install torch torchvision torchaudio  # from pytorch.org
pip install "gradience[bench]"
```

## Supported Environments

### Python Versions
- **✅ Python 3.10, 3.11, 3.12** (tested in CI)
- **❌ Python 3.8** - Not supported (requires newer typing features)
- **❌ Python 3.13+** - Not yet tested (may work but not guaranteed)

### Operating Systems
- **✅ Linux** (Ubuntu 20.04+, RHEL 8+, Amazon Linux 2)
- **✅ macOS** (10.15+, both Intel and Apple Silicon)
- **✅ Windows** (Windows 10+, WSL2 recommended for best experience)

### Hardware
- **✅ CPU-only** - Full functionality (slower training)
- **✅ CUDA GPUs** - Nvidia GPUs with CUDA 11.8+ or 12.x
- **✅ Apple Silicon** - M1/M2/M3 Macs with MPS acceleration
- **⚠️ AMD GPUs** - Limited PyTorch support, CPU fallback recommended

## Installation Methods

### Method 1: Standard Install (Recommended)

```bash
# Install with full benchmarking capabilities
pip install "gradience[bench]"

# Verify installation
gradience --help           # Should complete in <3 seconds
gradience-bench --help     # Should complete in <15 seconds (first run)
```

**What you get:**
- Core Gradience tools (audit, monitor, compress)
- HuggingFace integration (transformers, peft, datasets)
- CPU-compatible PyTorch (will work on any machine)

### Method 2: CPU-Only Install

```bash
# Install CPU-only PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install Gradience
pip install "gradience[bench]"
```

**When to use:**
- Server environments without GPU
- CI/CD pipelines
- Development machines
- Cost optimization (no GPU wheel download)

### Method 3: GPU-Optimized Install

#### CUDA (Nvidia)

```bash
# For CUDA 12.1 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install Gradience
pip install "gradience[bench]"

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Apple Silicon (M1/M2/M3)

```bash
# Standard install works - MPS acceleration enabled automatically
pip install "gradience[bench]"

# Verify MPS detection
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Method 4: Development Install

```bash
# Clone repository
git clone https://github.com/johntnanney/gradience.git
cd gradience

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,bench]"

# Verify installation
pytest tests/ -x  # Run tests
gradience --help
```

**What you get:**
- Editable installation (changes reflected immediately)
- Development tools (pytest, ruff, mypy, pre-commit)
- All benchmarking capabilities

## Environment-Specific Guidance

### RunPod Setup

**Essential first step** - Configure cache locations to prevent disk space issues:

```bash
# Copy-paste this before any Gradience commands
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export TORCH_HOME="/workspace/.cache/torch"

# Create directories
mkdir -p /workspace/.cache/{huggingface/{hub,datasets},torch}

# Install Gradience
pip install "gradience[bench]"
```

**Or use the convenience script:**

```bash
# One-liner setup
source scripts/runpod/env.sh

# Then use Gradience normally
gradience-bench --config configs/your_config.yaml --output results/
```

### Google Colab

```python
# In a Colab cell
!pip install "gradience[bench]"

# Verify installation
!gradience --help

# Run CPU demo
!gradience-bench --config gradience/bench/configs/distilbert_sst2_ci.yaml \
                 --device cpu \
                 --output ./demo_run \
                 --smoke
```

### Docker/Containers

```dockerfile
# Dockerfile example
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install Gradience
RUN pip install "gradience[bench]"

# Set cache directories
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV TORCH_HOME=/app/.cache/torch

# Create cache directories
RUN mkdir -p /app/.cache/{huggingface/{hub,datasets},torch}

WORKDIR /app
```

### CI/CD Environments

```yaml
# GitHub Actions example
- name: Install Gradience
  run: |
    # CPU-only for faster CI
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "gradience[bench]"
    
    # Verify installation
    gradience --help
    gradience-bench --help
```

## Common Issues & Solutions

### 🚨 ModuleNotFoundError: datasets

**Error:**
```
ImportError: No module named 'datasets'
```

**Cause:** Trying to use benchmarking features without ML dependencies.

**Fix:**
```bash
# Install the bench extras
pip install "gradience[bench]"

# Or if you already have core gradience
pip install datasets transformers peft accelerate
```

### 🚨 ModuleNotFoundError: transformers

**Error:**
```
ImportError: No module named 'transformers'
```

**Cause:** Using HuggingFace integration without transformers installed.

**Fix:**
```bash
# Install bench extras (includes transformers)
pip install "gradience[bench]"
```

### 🚨 CLI Help Takes Too Long

**Problem:** `gradience-bench --help` takes >15 seconds

**Cause:** First run loads transformers library (imports models, tokenizers)

**Expected behavior:**
- `gradience --help`: < 3 seconds (no ML imports)
- `gradience-bench --help` first run: < 15 seconds (loads ML stack)
- `gradience-bench --help` subsequent: < 5 seconds (cached imports)

**If still slow:**
```bash
# Check for conflicting installations
pip list | grep -E "torch|transformers"

# Clean reinstall
pip uninstall gradience torch transformers
pip install "gradience[bench]"
```

### 🚨 CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# 1. Use CPU device
gradience-bench --device cpu --config your_config.yaml --output results/

# 2. Reduce batch size in config
# Edit your config.yaml:
train:
  per_device_train_batch_size: 2  # Reduce from 8
  per_device_eval_batch_size: 4   # Reduce from 32

# 3. Use smoke mode for testing
gradience-bench --smoke --config your_config.yaml --output results/
```

### 🚨 Disk Space Issues

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Cause:** Models downloading to wrong cache location (usually `/root/` on cloud instances)

**Fix:**
```bash
# Set cache environment variables BEFORE first run
export HF_HOME="/workspace/.cache/huggingface"  # or your preferred location
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export TORCH_HOME="/workspace/.cache/torch"

# Create directories
mkdir -p /workspace/.cache/{huggingface/{hub,datasets},torch}

# Clear existing cache if needed
rm -rf ~/.cache/huggingface/
```

### 🚨 Safetensors Loading Errors

**Error:**
```
OSError: Unable to load safetensors file
```

**Cause:** Corrupted download or incomplete transfer

**Fix:**
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/hub/models--*model-name*

# Or clear entire HF cache
rm -rf ~/.cache/huggingface/

# Re-run command (will re-download)
gradience-bench --config your_config.yaml --output results/
```

### 🚨 Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/root/.cache/huggingface'
```

**Fix:**
```bash
# Option 1: Set custom cache location
export HF_HOME="/tmp/hf_cache"
mkdir -p /tmp/hf_cache

# Option 2: Fix permissions (if you have sudo)
sudo chown -R $(whoami) ~/.cache/huggingface/
```

## Cache Configuration

### Environment Variables

Set these **before** first run to control where models/datasets are cached:

```bash
# HuggingFace cache (models, tokenizers)
export HF_HOME="/your/preferred/path/.cache/huggingface"
export HF_HUB_CACHE="/your/preferred/path/.cache/huggingface/hub"

# Datasets cache
export HF_DATASETS_CACHE="/your/preferred/path/.cache/huggingface/datasets"

# PyTorch models cache
export TORCH_HOME="/your/preferred/path/.cache/torch"

# Temporary files (some operations)
export TMPDIR="/your/preferred/path/tmp"
```

### Recommended Locations

| Environment | Cache Location | Reason |
|-------------|---------------|---------|
| **RunPod** | `/workspace/.cache/` | Persistent across restarts, large disk |
| **Colab** | `/content/.cache/` | Session-local, but accessible |
| **Local dev** | `~/.cache/` (default) | Standard user cache |
| **Docker** | `/app/.cache/` | Container-local, predictable |
| **CI/CD** | `/tmp/.cache/` | Ephemeral, fast SSD |

### Cache Sizes

Typical cache sizes to expect:

- **distilbert-base-uncased**: ~250MB
- **roberta-base**: ~500MB  
- **mistralai/Mistral-7B-v0.1**: ~13GB
- **SST-2 dataset**: ~7MB
- **GSM8K dataset**: ~3MB

Plan accordingly for your storage constraints.

## Performance Expectations

### CLI Responsiveness
- `gradience --help`: < 3 seconds
- `gradience audit --help`: < 3 seconds  
- `gradience-bench --help` (first run): < 15 seconds
- `gradience-bench --help` (cached): < 5 seconds

### Benchmarking Times (CPU)
- **Smoke test** (distilbert + SST-2): ~60 seconds
- **Full benchmark** (distilbert + SST-2): ~20 minutes
- **Large model** (Mistral-7B): ~6 hours

### Disk Usage
- **Base install**: ~10MB  
- **With PyTorch CPU**: ~500MB
- **With PyTorch CUDA**: ~2.5GB
- **After first model download**: +250MB-13GB

## Verification Commands

After installation, verify everything works:

```bash
# Test basic functionality
gradience --help                    # Should be fast (<3s)
gradience audit --help             # Should be fast (<3s)

# Test ML functionality (first run may be slower)
gradience-bench --help             # <15s first run, <5s cached

# Test with actual config
gradience check --help
gradience monitor --help

# Run smoke test (optional, takes ~60s)
gradience-bench --config gradience/bench/configs/distilbert_sst2_ci.yaml \
                --device cpu \
                --output /tmp/verify_test \
                --smoke
```

If all commands complete successfully, your installation is ready!

## Getting Help

If you're still having issues:

1. **Check PyPI version**: `pip show gradience` 
2. **Check Python version**: `python --version`
3. **Check PyTorch**: `python -c "import torch; print(torch.__version__)"`
4. **Check disk space**: `df -h`
5. **Check cache locations**: `echo $HF_HOME`

**Common solutions:**
- Clear pip cache: `pip cache purge`
- Fresh virtual environment: `python -m venv fresh_env && source fresh_env/bin/activate`
- Update pip: `pip install --upgrade pip`

**Still stuck?** File an issue at [github.com/johntnanney/gradience/issues](https://github.com/johntnanney/gradience/issues) with:
- Operating system and Python version
- Complete error message
- Output of `pip list | grep -E "gradience|torch|transformers"`