#!/bin/bash
#
# Blessed Evidence Pack Runner Script
#
# The canonical RunPod entrypoint for Gradience Evidence Pack demonstrations.
# Runs all three Evidence Pack experiments with proper environment setup.
# 
# Usage:
#   ./scripts/run_evidence_pack.sh /workspace/evidence_runs
#   ./scripts/run_evidence_pack.sh /tmp/evidence_demo
#
# What it does:
#   1. Sets up HF cache environment variables (RunPod-friendly)
#   2. Runs exp02 (DistilBERT CPU benchmark) 
#   3. Runs exp01_smoke (Mistral GPU smoke test)
#   4. Runs exp01_multiseed (Mistral multiseed evaluation)
#   5. Reports all output paths for easy access
#
# Stop doing bespoke commands by memory - use this blessed script!
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

echo -e "${BOLD}${BLUE}🧪 GRADIENCE EVIDENCE PACK RUNNER${NC}"
echo "=================================="
echo "Canonical RunPod entrypoint for Evidence Pack demonstrations"
echo "Timestamp: $(date)"
echo

# 1. Validate arguments
if [ $# -ne 1 ]; then
    echo -e "${RED}❌ Usage: $0 <EVID_ROOT>${NC}"
    echo
    echo "Examples:"
    echo "  $0 /workspace/evidence_runs"
    echo "  $0 /tmp/evidence_demo"
    echo
    exit 1
fi

EVID_ROOT="$1"

# Validate EVID_ROOT
if [ -z "$EVID_ROOT" ]; then
    echo -e "${RED}❌ EVID_ROOT cannot be empty${NC}"
    exit 1
fi

echo -e "${YELLOW}📂 Evidence Pack Root: ${EVID_ROOT}${NC}"
echo "   Creating directory structure..."
mkdir -p "$EVID_ROOT"

# 2. Set up HF cache environment variables (RunPod-friendly)
echo -e "${YELLOW}🔧 Setting up HuggingFace cache environment...${NC}"

# Source RunPod env script if available, otherwise set up manually
if [ -f "$SCRIPT_DIR/runpod/env.sh" ]; then
    echo "   Using RunPod environment script"
    source "$SCRIPT_DIR/runpod/env.sh"
else
    echo "   Setting up manual HF cache environment"
    # Determine cache base - prefer /workspace/ on RunPod, fallback to standard locations
    if [ -d "/workspace" ] && [ -w "/workspace" ]; then
        CACHE_BASE="/workspace/hf_cache"
        echo "   Detected RunPod environment - using /workspace/"
    else
        CACHE_BASE="$HOME/.cache/gradience_evidence"
        echo "   Using local environment - caching in $HOME"
    fi
    
    # Set all the HF environment variables
    export HF_HOME="${CACHE_BASE}/hf_home"
    export HF_HUB_CACHE="${CACHE_BASE}/hub"
    export HF_DATASETS_CACHE="${CACHE_BASE}/datasets"
    export TORCH_HOME="${CACHE_BASE}/torch"
    export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"  # Legacy compatibility
    
    # Create cache directories
    mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"
    
    echo "   Cache directories created:"
    echo "     HF_HOME: $HF_HOME"
    echo "     HF_HUB_CACHE: $HF_HUB_CACHE" 
    echo "     HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
    echo "     TORCH_HOME: $TORCH_HOME"
fi

echo

# 3. Define experiment paths
EXP02_OUTPUT="$EVID_ROOT/exp02_distilbert_sst2"
EXP01_SMOKE_OUTPUT="$EVID_ROOT/exp01_smoke_mistral_gsm8k"
EXP01_MULTISEED_OUTPUT="$EVID_ROOT/exp01_multiseed_mistral_gsm8k"

echo -e "${BLUE}🎯 Planned experiment outputs:${NC}"
echo "   • exp02 (DistilBERT): $EXP02_OUTPUT"
echo "   • exp01 smoke (Mistral): $EXP01_SMOKE_OUTPUT"
echo "   • exp01 multiseed (Mistral): $EXP01_MULTISEED_OUTPUT"
echo

# 4. Run experiments
echo -e "${BOLD}${GREEN}🚀 STARTING EVIDENCE PACK EXPERIMENTS${NC}"
echo "======================================"

# Track success/failure
EXPERIMENTS_SUCCESS=()
EXPERIMENTS_FAILED=()

# Experiment 1: exp02_distilbert_sst2 (CPU-friendly)
echo
echo -e "${YELLOW}📊 Experiment 1/3: DistilBERT SST2 (CPU-friendly)${NC}"
echo "Config: gradience/bench/configs/evidence/exp02_distilbert_sst2.yaml"
echo "Output: $EXP02_OUTPUT"
echo "Expected runtime: ~5-10 minutes"
echo

if python3 -m gradience.bench.run_bench \
    --config "gradience/bench/configs/evidence/exp02_distilbert_sst2.yaml" \
    --output "$EXP02_OUTPUT" 2>&1 | tee "$EVID_ROOT/exp02_log.txt"; then
    echo -e "${GREEN}✅ exp02 completed successfully${NC}"
    EXPERIMENTS_SUCCESS+=("exp02_distilbert_sst2")
else
    echo -e "${RED}❌ exp02 failed${NC}"
    EXPERIMENTS_FAILED+=("exp02_distilbert_sst2")
fi

# Experiment 2: exp01_smoke (GPU smoke test)  
echo
echo -e "${YELLOW}📊 Experiment 2/3: Mistral GSM8K Smoke Test (GPU)${NC}"
echo "Config: gradience/bench/configs/evidence/exp01_mistral_gsm8k_smoke.yaml"
echo "Output: $EXP01_SMOKE_OUTPUT"
echo "Expected runtime: ~5 minutes"
echo

if python3 -m gradience.bench.run_bench \
    --config "gradience/bench/configs/evidence/exp01_mistral_gsm8k_smoke.yaml" \
    --output "$EXP01_SMOKE_OUTPUT" 2>&1 | tee "$EVID_ROOT/exp01_smoke_log.txt"; then
    echo -e "${GREEN}✅ exp01_smoke completed successfully${NC}"
    EXPERIMENTS_SUCCESS+=("exp01_smoke")
else
    echo -e "${RED}❌ exp01_smoke failed${NC}"
    EXPERIMENTS_FAILED+=("exp01_smoke") 
fi

# Experiment 3: exp01_multiseed (comprehensive evaluation)
echo
echo -e "${YELLOW}📊 Experiment 3/3: Mistral GSM8K Multiseed (GPU)${NC}"
echo "Config: gradience/bench/configs/evidence/exp01_mistral_gsm8k_multiseed.yaml"  
echo "Output: $EXP01_MULTISEED_OUTPUT"
echo "Expected runtime: ~45-90 minutes"
echo

if python3 -m gradience.bench.run_bench \
    --config "gradience/bench/configs/evidence/exp01_mistral_gsm8k_multiseed.yaml" \
    --output "$EXP01_MULTISEED_OUTPUT" 2>&1 | tee "$EVID_ROOT/exp01_multiseed_log.txt"; then
    echo -e "${GREEN}✅ exp01_multiseed completed successfully${NC}"
    EXPERIMENTS_SUCCESS+=("exp01_multiseed")
else
    echo -e "${RED}❌ exp01_multiseed failed${NC}"
    EXPERIMENTS_FAILED+=("exp01_multiseed")
fi

# 5. Generate final report
echo
echo -e "${BOLD}${BLUE}📋 EVIDENCE PACK COMPLETION REPORT${NC}"
echo "================================="
echo "Timestamp: $(date)"
echo "Evidence Root: $EVID_ROOT"
echo

# Success summary
if [ ${#EXPERIMENTS_SUCCESS[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Successful experiments (${#EXPERIMENTS_SUCCESS[@]}/3):${NC}"
    for exp in "${EXPERIMENTS_SUCCESS[@]}"; do
        case $exp in
            "exp02_distilbert_sst2")
                echo "   • $exp → $EXP02_OUTPUT"
                ;;
            "exp01_smoke")
                echo "   • $exp → $EXP01_SMOKE_OUTPUT"
                ;;
            "exp01_multiseed")
                echo "   • $exp → $EXP01_MULTISEED_OUTPUT"
                ;;
        esac
    done
    echo
fi

# Failure summary
if [ ${#EXPERIMENTS_FAILED[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed experiments (${#EXPERIMENTS_FAILED[@]}/3):${NC}"
    for exp in "${EXPERIMENTS_FAILED[@]}"; do
        echo "   • $exp (check ${EVID_ROOT}/${exp}_log.txt)"
    done
    echo
fi

# Output paths for easy access
echo -e "${BLUE}📁 Output paths for analysis:${NC}"
echo "   Evidence Root: $EVID_ROOT"
echo "   Logs: $EVID_ROOT/*_log.txt"
echo

if [ -d "$EXP02_OUTPUT" ]; then
    echo "   exp02 results: $EXP02_OUTPUT/"
    echo "     • Bench report: $EXP02_OUTPUT/bench.md"
    echo "     • Machine data: $EXP02_OUTPUT/bench.json"
    echo "     • Adapter audit: $EXP02_OUTPUT/probe_r16/audit.json"
fi

if [ -d "$EXP01_SMOKE_OUTPUT" ]; then
    echo "   exp01_smoke results: $EXP01_SMOKE_OUTPUT/"
    echo "     • Bench report: $EXP01_SMOKE_OUTPUT/bench.md"
    echo "     • Machine data: $EXP01_SMOKE_OUTPUT/bench.json"
fi

if [ -d "$EXP01_MULTISEED_OUTPUT" ]; then
    echo "   exp01_multiseed results: $EXP01_MULTISEED_OUTPUT/"
    echo "     • Bench report: $EXP01_MULTISEED_OUTPUT/bench.md"
    echo "     • Machine data: $EXP01_MULTISEED_OUTPUT/bench.json"
    echo "     • Adapter audit: $EXP01_MULTISEED_OUTPUT/probe_r32/audit.json"
fi

echo
echo -e "${BLUE}💡 Next steps:${NC}"
echo "   • Review human-readable reports: */bench.md"
echo "   • Analyze machine data: */bench.json" 
echo "   • Inspect adapter audits: */probe_r*/audit.json"
echo "   • Package for transport: tar -czf evidence_pack_${TIMESTAMP}.tar.gz -C $(dirname $EVID_ROOT) $(basename $EVID_ROOT)"
echo

# Exit with appropriate code
if [ ${#EXPERIMENTS_FAILED[@]} -eq 0 ]; then
    echo -e "${BOLD}${GREEN}🎉 All Evidence Pack experiments completed successfully!${NC}"
    exit 0
else
    echo -e "${BOLD}${RED}⚠️  Some Evidence Pack experiments failed. Check logs for details.${NC}"
    exit 1
fi