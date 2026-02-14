#!/bin/bash
# ===========================================================================
# run_all.sh — Master runbook for merge-audit A40 GPU test protocol
#
# Run this from /workspace/gradience after setup is complete.
# Total runtime: ~1-2 hours including downloads and GPU verification.
#
# Usage:
#   cd /workspace/gradience
#   bash scripts/merge_audit_test/run_all.sh
# ===========================================================================

set -e

SCRIPT_DIR="scripts/merge_audit_test"
WORKSPACE="/workspace/merge_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  Merge-Audit A40 GPU Test Protocol"
echo "  $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 0: Verify environment
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 0: Environment Check ---"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0)}')" || {
    echo "ERROR: PyTorch/CUDA not available"
    exit 1
}

python -c "import gradience; print(f'Gradience {gradience.__version__}')" || {
    echo "ERROR: Gradience not installed. Run: pip install -e '.[bench]'"
    exit 1
}

python -c "from gradience.vnext.merge import merge_audit; print('merge_audit import OK')" || {
    echo "ERROR: merge_audit module not available"
    exit 1
}

# Check HF cache location — bench protocol may have set HF_HOME
if [ -n "$HF_HOME" ]; then
    echo "HF_HOME=$HF_HOME"
elif [ -d "/workspace/hf_cache" ]; then
    export HF_HOME="/workspace/hf_cache"
    export HF_HUB_CACHE="/workspace/hf_cache/hub"
    echo "Auto-detected HF cache at /workspace/hf_cache (from bench run)"
else
    echo "HF_HOME not set, using default: ~/.cache/huggingface"
fi

echo "Environment OK"

# ---------------------------------------------------------------------------
# Phase 1: Discover available adapters
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 1: Discover Adapters ---"
echo "  (Review output and update download_adapters.py ADAPTERS dict if needed)"
echo ""

python $SCRIPT_DIR/discover_adapters.py 2>&1 | tee $WORKSPACE/discover_log_${TIMESTAMP}.txt

echo ""
echo "  Discovery log saved to $WORKSPACE/discover_log_${TIMESTAMP}.txt"
echo ""
read -p "  Press Enter to continue with downloads (or Ctrl+C to edit configs first)..."

# ---------------------------------------------------------------------------
# Phase 2: Download adapters
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 2: Download Adapters ---"

python $SCRIPT_DIR/download_adapters.py 2>&1 | tee $WORKSPACE/download_log_${TIMESTAMP}.txt

echo ""
echo "  Download log saved to $WORKSPACE/download_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Phase 3: Run merge-audit on all pairs
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 3: Run Merge Audits ---"

python $SCRIPT_DIR/run_merge_audits.py 2>&1 | tee $WORKSPACE/audit_log_${TIMESTAMP}.txt

echo ""
echo "  Audit log saved to $WORKSPACE/audit_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Phase 4: Analyze results
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 4: Analyze Results ---"

python $SCRIPT_DIR/analyze_merge_results.py 2>&1 | tee $WORKSPACE/analysis_log_${TIMESTAMP}.txt

echo ""
echo "  Analysis log saved to $WORKSPACE/analysis_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Phase 5: GPU verification
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 5: GPU Verification ---"
echo "  (Loading base model + adapters on GPU, running inference)"
echo ""

# Check if Mistral-7B weights are already cached (e.g. from bench run)
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--mistralai--Mistral-7B-v0.1"
if [ -d "$HF_CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1)
    echo "  Mistral-7B weights cached: $HF_CACHE_DIR ($CACHE_SIZE)"
    echo "  GPU verification will skip the ~14GB download."
else
    echo "  Mistral-7B weights NOT cached."
    echo "  GPU verification will download ~14GB on first load."
    echo "  (If you ran the bench protocol first, check HF_HOME is set correctly.)"
fi
echo ""

python $SCRIPT_DIR/gpu_verification.py 2>&1 | tee $WORKSPACE/gpu_log_${TIMESTAMP}.txt

echo ""
echo "  GPU log saved to $WORKSPACE/gpu_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Phase 6: Package results
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 6: Package Results ---"

ARCHIVE="merge_audit_a40_results_${TIMESTAMP}.tar.gz"
cd /workspace
tar czf "$ARCHIVE" \
    merge_test/results/ \
    merge_test/*_log_*.txt \
    2>/dev/null

echo "  Results archived: /workspace/$ARCHIVE"
echo ""
echo "  Download with:"
echo "    scp root@<runpod-ip>:/workspace/$ARCHIVE ./"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  TEST PROTOCOL COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "  Key artifacts:"
echo "    /workspace/merge_test/results/*/merge_audit.json"
echo "    /workspace/merge_test/results/*/merge_audit.md"
echo "    /workspace/merge_test/results/run_summary.json"
echo "    /workspace/merge_test/results/analysis_summary.json"
echo ""
echo "  Remember to stop the pod when done!"
echo "============================================================"
