# RunPod Quick Start

**TL;DR for RunPod users: Don't let cache fill up `/root/` - redirect to `/workspace/`.**

## 🚀 RunPod Setup (Copy-Paste)

```bash
# 1. Configure cache directories FIRST (prevents disk quota issues)
source /workspace/gradience/scripts/setup_cache_env.sh

# 2. Standard setup
cd /workspace
git clone https://github.com/johntnanney/gradience.git
cd gradience

python -m venv .venv
source .venv/bin/activate
pip install -U pip  
pip install -e ".[hf,dev]"

# 3. Verify everything works
make verify-version
```

## 💾 Why Cache Configuration Matters on RunPod

**RunPod storage layout:**
- `/root/` - Small system disk (~10-50GB) 
- `/workspace/` - Large data disk (100GB-1TB+)

**Default behavior (BAD):**
- HuggingFace downloads to `/root/.cache/huggingface/` 
- Large models quickly fill system disk
- Training fails with "Disk quota exceeded"

**Fixed behavior (GOOD):**
- `setup_cache_env.sh` redirects everything to `/workspace/`
- Unlimited space for models and datasets
- Training runs complete successfully

## 🛠️ RunPod-Specific Commands

```bash
# Check disk usage
df -h /root
df -h /workspace

# Quick cache fix if you forgot to run setup_cache_env.sh
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
mkdir -p /workspace/hf_cache/{hub,datasets}

# Clean up if /root/ already filled up
rm -rf /root/.cache/huggingface/*
rm -rf /root/.cache/torch/*
```

## 🔧 Persistent Configuration

Add to your RunPod startup script or `.bashrc`:

```bash
# Add this to /root/.bashrc for automatic setup
if [ -d "/workspace/gradience" ]; then
    source /workspace/gradience/scripts/setup_cache_env.sh
fi
```

## 🚨 Troubleshooting

### "No space left on device" during model download
```bash
# Check what's using space
du -sh /root/.cache/* | sort -hr

# Emergency: move existing cache to workspace  
mkdir -p /workspace/hf_cache/hub
mv /root/.cache/huggingface/hub/* /workspace/hf_cache/hub/ 2>/dev/null || true
export HF_HUB_CACHE=/workspace/hf_cache/hub
```

### Models downloading to wrong location
```bash
# Check cache environment variables are set
env | grep -E "HF_|TORCH_" 

# Must set BEFORE importing transformers
export HF_HUB_CACHE=/workspace/hf_cache/hub
python -c "import transformers; ..."  # Now uses correct cache
```

### Jupyter notebook not respecting cache variables
```bash
# Set in notebook cell BEFORE any imports
import os
os.environ['HF_HUB_CACHE'] = '/workspace/hf_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/workspace/hf_cache/datasets'

import transformers  # Now uses correct cache
```

---

💡 **Remember**: Cache configuration must happen BEFORE importing HuggingFace libraries. Once imported, the cache location is fixed for that Python process.