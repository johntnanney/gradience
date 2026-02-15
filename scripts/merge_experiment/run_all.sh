#!/bin/bash
# ===========================================================================
# run_all.sh — Master runbook for merge execution & validation experiment
#
# Run this from /workspace/gradience after setup is complete.
# Total runtime: ~2-4 hours including downloads, merges, and GPU evaluation.
#
# Usage:
#   cd /workspace/gradience
#   bash scripts/merge_experiment/run_all.sh
#
# Prerequisites:
#   - pip install -e ".[bench]"
#   - GPU available (A40 recommended)
#   - Set ADAPTER_A_REPO and ADAPTER_B_REPO environment variables
# ===========================================================================

set -e

SCRIPT_DIR="scripts/merge_experiment"
WORKSPACE="${MERGE_WORKSPACE:-/workspace/merge_experiment}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
OUTPUT_RANK="${OUTPUT_RANK:-8}"
OUTPUT_ALPHA="${OUTPUT_ALPHA:-16.0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-500}"

mkdir -p "$WORKSPACE"

echo "============================================================"
echo "  Merge Execution & Validation Experiment"
echo "  $(date)"
echo "  Workspace: $WORKSPACE"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 0: Verify environment
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 0: Environment Check ---"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

python -c "import gradience; print(f'Gradience {gradience.__version__}')" || {
    echo "ERROR: Gradience not installed. Run: pip install -e '.[bench]'"
    exit 1
}

python -c "from gradience.vnext.merge import merge_audit, execute_merge; print('merge imports OK')" || {
    echo "ERROR: merge modules not available"
    exit 1
}

python -c "import transformers, peft, datasets; print('bench deps OK')" || {
    echo "ERROR: Bench dependencies not installed. Run: pip install -e '.[bench]'"
    exit 1
}

# Check adapter repos
if [ -z "$ADAPTER_A_REPO" ] || [ -z "$ADAPTER_B_REPO" ]; then
    echo ""
    echo "ERROR: Set adapter HuggingFace repos:"
    echo "  export ADAPTER_A_REPO=<hf-repo-for-mnli-adapter>"
    echo "  export ADAPTER_B_REPO=<hf-repo-for-qnli-adapter>"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "Adapter A repo: $ADAPTER_A_REPO"
echo "Adapter B repo: $ADAPTER_B_REPO"
echo "Base model: $BASE_MODEL"
echo "Output rank: $OUTPUT_RANK"
echo "Max eval samples: $MAX_EVAL_SAMPLES"

# HF cache
if [ -n "$HF_HOME" ]; then
    echo "HF_HOME=$HF_HOME"
elif [ -d "/workspace/hf_cache" ]; then
    export HF_HOME="/workspace/hf_cache"
    export HF_HUB_CACHE="/workspace/hf_cache/hub"
    echo "Auto-detected HF cache at /workspace/hf_cache"
else
    echo "HF_HOME not set, using default: ~/.cache/huggingface"
fi

echo ""
echo "Environment OK"
echo ""
read -p "Press Enter to start experiment (or Ctrl+C to abort)..."

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
echo ""
echo "--- Starting experiment ---"

python $SCRIPT_DIR/run_experiment.py \
    --workspace "$WORKSPACE" \
    --base-model "$BASE_MODEL" \
    --adapter-a-repo "$ADAPTER_A_REPO" \
    --adapter-b-repo "$ADAPTER_B_REPO" \
    --output-rank "$OUTPUT_RANK" \
    --output-alpha "$OUTPUT_ALPHA" \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    2>&1 | tee "$WORKSPACE/experiment_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Package results
# ---------------------------------------------------------------------------
echo ""
echo "--- Packaging Results ---"

ARCHIVE="merge_experiment_results_${TIMESTAMP}.tar.gz"
cd /workspace
tar czf "$ARCHIVE" \
    merge_experiment/results/ \
    merge_experiment/*_log_*.txt \
    2>/dev/null || true

echo "  Results archived: /workspace/$ARCHIVE"
echo ""
echo "  Download with:"
echo "    scp root@<runpod-ip>:/workspace/$ARCHIVE ./"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  EXPERIMENT COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "  Key artifacts:"
echo "    $WORKSPACE/results/experiment_results.json"
echo "    $WORKSPACE/results/audit/merge_audit.json"
echo "    $WORKSPACE/results/merged_*/adapter_config.json"
echo "    $WORKSPACE/results/evaluations/*.json"
echo ""
echo "  Remember to stop the pod when done!"
echo "============================================================"
