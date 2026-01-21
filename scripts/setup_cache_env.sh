#!/bin/bash
# Gradience Cache Environment Setup
# 
# Purpose: Prevent "Disk quota exceeded" chaos by configuring
#          HuggingFace and PyTorch caching to use workspace storage
#
# Usage:
#   source scripts/setup_cache_env.sh
#   # or
#   bash scripts/setup_cache_env.sh

set -e

echo "🗂️  Setting up Gradience cache environment..."

# Determine cache root based on environment
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    # RunPod/cloud environment - use workspace
    CACHE_ROOT="/workspace"
    echo "📁 Detected cloud environment - using /workspace"
elif [ -n "$DOCKER_CONTAINER" ] || [ -f "/.dockerenv" ]; then
    # Docker environment
    CACHE_ROOT="/app/cache" 
    echo "🐳 Detected Docker environment - using /app/cache"
else
    # Local development
    CACHE_ROOT="$HOME/.cache/gradience"
    echo "💻 Detected local environment - using ~/.cache/gradience"
fi

# Set HuggingFace environment variables
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="$CACHE_ROOT/hf_cache/hf_home"
export HF_HUB_CACHE="$CACHE_ROOT/hf_cache/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_cache/datasets"

# Set PyTorch environment variables
export TORCH_HOME="$CACHE_ROOT/torch_cache"

# Set Gradience environment variables
export GRADIENCE_CACHE_DIR="$CACHE_ROOT/gradience_cache"

# Create cache directories
echo "📁 Creating cache directories..."
mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE" 
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$GRADIENCE_CACHE_DIR"

echo "✅ Cache environment configured!"
echo ""
echo "📋 Environment variables set:"
echo "   HF_HOME=$HF_HOME"
echo "   HF_HUB_CACHE=$HF_HUB_CACHE"
echo "   HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "   TORCH_HOME=$TORCH_HOME"
echo "   GRADIENCE_CACHE_DIR=$GRADIENCE_CACHE_DIR"
echo ""

# Check available space
if command -v df >/dev/null 2>&1; then
    echo "💾 Available space at cache root:"
    df -h "$CACHE_ROOT" 2>/dev/null | tail -1 || echo "   (Unable to check disk space)"
    echo ""
fi

echo "💡 To make this persistent, add to your shell profile:"
echo "   echo 'source $(pwd)/scripts/setup_cache_env.sh' >> ~/.bashrc"
echo ""
echo "🚀 Ready to run Gradience without storage issues!"