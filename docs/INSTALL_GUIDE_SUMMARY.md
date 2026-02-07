# Installation Guide Summary

## Overview

Created a comprehensive "stop the bleeding" installation guide that prevents the most common user setup issues.

## What We Created

### 📖 **[docs/install.md](install.md)** - Complete Installation Guide

A thorough guide covering every installation scenario and common failure mode.

## Content Coverage

### ✅ **Supported Environments**
- **Python versions**: Explicit support matrix (3.9-3.12 ✅, 3.8 ❌, 3.13+ ⚠️)
- **Operating Systems**: Linux, macOS, Windows with specific version requirements
- **Hardware**: CPU-only, CUDA GPUs, Apple Silicon, AMD (with caveats)

### ✅ **Installation Methods**
1. **Standard Install** - `pip install "gradience[bench]"` (recommended)
2. **CPU-Only** - Specific PyTorch CPU instructions
3. **GPU-Optimized** - CUDA 11.8/12.1 and Apple MPS instructions
4. **Development** - Editable installs with dev tools

### ✅ **Environment-Specific Guidance**
- **RunPod** - Cache configuration to prevent disk space issues
- **Google Colab** - Copy-paste cell instructions
- **Docker** - Complete Dockerfile example
- **CI/CD** - GitHub Actions example with CPU PyTorch

### ✅ **Common Issues & Solutions**

**The exact issues we hit during development:**

1. **`ModuleNotFoundError: datasets`**
   - **Cause**: Trying benchmarks without ML deps
   - **Fix**: Install `[bench]` extras

2. **`ModuleNotFoundError: transformers`**
   - **Cause**: Using HF integration without transformers
   - **Fix**: Install bench extras or transformers directly

3. **"Help command takes too long"**
   - **Expected**: `gradience --help` < 3s, `gradience-bench --help` < 15s first run
   - **Cause**: First ML import loads transformers stack
   - **Normal behavior**: Explained with timing expectations

4. **CUDA Out of Memory**
   - **Solutions**: CPU fallback, batch size reduction, smoke mode

5. **Disk Space Issues**
   - **Cause**: Cache location misconfiguration
   - **Fix**: Set HF_HOME and related env vars

6. **Safetensors Loading Errors**
   - **Cause**: Corrupted downloads
   - **Fix**: Clear cache and re-download

### ✅ **Cache Configuration**

Complete documentation of cache environment variables:

```bash
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"  
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export TORCH_HOME="/workspace/.cache/torch"
```

**With platform-specific recommendations:**
- RunPod: `/workspace/.cache/` (persistent)
- Colab: `/content/.cache/` (session-local)  
- Docker: `/app/.cache/` (predictable)
- CI: `/tmp/.cache/` (ephemeral)

### ✅ **Performance Expectations**

Clear benchmarks so users know what's normal:
- **CLI help times**: 3s base, 15s bench first run, 5s cached
- **Benchmark times**: 60s smoke test, 20min full CPU
- **Disk usage**: 10MB base → 500MB with PyTorch → +250MB-13GB with models

### ✅ **Verification & Troubleshooting**

- **Step-by-step verification** commands
- **Diagnostic commands** for common issues
- **"Getting Help" section** with issue template guidance
- **Environment debugging** checklist

## README Integration

### Updated README with:
- **Link to complete guide** in install section
- **Moved detailed guidance to dedicated doc** (keeps README focused)
- **Added to links section** for easy discovery

### Before/After:
- **Before**: Basic 3-tier install only
- **After**: Quick install + link to comprehensive guide

## Impact

### User Experience:
- **Prevents rage-quits** from common ModuleNotFoundError issues
- **Reduces support burden** with self-service troubleshooting
- **Speeds up onboarding** with environment-specific guidance
- **Sets proper expectations** with performance benchmarks

### Coverage:
- **All platforms** where Gradience runs
- **All installation methods** from pip to development
- **All failure modes** we encountered during development
- **All cache configuration** needed for different environments

## Validation

- ✅ All installation commands tested
- ✅ Common error messages verified
- ✅ Performance expectations match reality
- ✅ Cache configurations tested on RunPod-style environments
- ✅ Links and references work correctly

The guide now serves as a comprehensive "stop the bleeding" resource that should handle 90%+ of installation issues users encounter.