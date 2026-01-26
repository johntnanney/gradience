#!/bin/bash

# RunPod Environment Setup Script
# 
# Standardizes HuggingFace cache locations to prevent disk quota issues on RunPod.
# 
# RunPod storage layout:
#   /root/     - Small system disk (~10-50GB)  
#   /workspace/ - Large data disk (100GB-1TB+)
#
# This script redirects all HF caches to /workspace/ to avoid filling /root/
#
# Usage:
#   source scripts/runpod/env.sh
#   # OR
#   source /workspace/gradience/scripts/runpod/env.sh

set -euo pipefail

# Color output for visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up RunPod environment for Gradience...${NC}"

# Detect environment
if [[ -d "/workspace" ]]; then
    CACHE_BASE="/workspace/hf_cache"
    echo -e "${GREEN}✅ RunPod environment detected${NC}"
else
    # Fallback for local development
    CACHE_BASE="$HOME/.cache/gradience"
    echo -e "${YELLOW}⚠️  Non-RunPod environment - using local cache${NC}"
fi

# Create cache directories
echo -e "${BLUE}📁 Creating cache directories...${NC}"
mkdir -p "$CACHE_BASE"/{hf_home,hub,datasets}

# Set HuggingFace environment variables (current standards)
export HF_HOME="$CACHE_BASE/hf_home"
export HF_HUB_CACHE="$CACHE_BASE/hub"  
export HF_DATASETS_CACHE="$CACHE_BASE/datasets"

# Legacy compatibility (some older tools still use these)
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"

# PyTorch cache (for compiled models, etc.)
export TORCH_HOME="$CACHE_BASE/torch"
mkdir -p "$TORCH_HOME"

# Report configuration
echo -e "${GREEN}✅ Cache environment configured:${NC}"
echo -e "   HF_HOME=${GREEN}$HF_HOME${NC}"
echo -e "   HF_HUB_CACHE=${GREEN}$HF_HUB_CACHE${NC}"
echo -e "   HF_DATASETS_CACHE=${GREEN}$HF_DATASETS_CACHE${NC}"
echo -e "   TORCH_HOME=${GREEN}$TORCH_HOME${NC}"

# Check disk space if on RunPod
if [[ -d "/workspace" ]]; then
    echo -e "\n${BLUE}💾 Disk space check:${NC}"
    echo -e "   System disk (/root):"
    df -h /root | grep -v Filesystem || echo "   (Could not check /root disk)"
    echo -e "   Data disk (/workspace):"
    df -h /workspace | grep -v Filesystem || echo "   (Could not check /workspace disk)"
fi

# Warn about existing cache in /root
if [[ -d "/root/.cache/huggingface" ]] && [[ "$(ls -A /root/.cache/huggingface 2>/dev/null)" ]]; then
    echo -e "\n${YELLOW}⚠️  WARNING: Existing HuggingFace cache found in /root/.cache/huggingface${NC}"
    echo -e "   This may consume system disk space. Consider cleaning up:"
    echo -e "   ${BLUE}rm -rf /root/.cache/huggingface/*${NC}"
    echo -e "   ${BLUE}rm -rf /root/.cache/torch/*${NC}"
fi

# Instructions for persistence
if [[ -d "/workspace" ]]; then
    echo -e "\n${BLUE}💡 To make this persistent, add to your RunPod startup:${NC}"
    echo -e "   echo 'source /workspace/gradience/scripts/runpod/env.sh' >> ~/.bashrc"
fi

echo -e "\n${GREEN}🎉 Environment setup complete!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "   1. Run your gradience commands"
echo -e "   2. Models will download to ${GREEN}/workspace/hf_cache/${NC}"
echo -e "   3. No more disk quota issues! 🚀"