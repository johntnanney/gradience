#!/bin/bash
# Lead-Lag V2: Resolution artifact test + overparameterization ablation
#
# 3 new conditions, all with --structural-every-n 25 (matching eval interval):
#   1. chat / r=8   — constrained capacity on easy task
#   2. math / r=32  — full capacity on hard task
#   3. math / r=8   — constrained capacity on hard task
#
# Existing data (chat / r=32, structural every 50) serves as baseline.
# Each condition runs 1 seed (42) for direction testing; add seeds later
# if the direction result is interesting.
#
# Expected runtime: ~4.5h per run on A100 80GB = ~13.5h total
#
# Usage on RunPod:
#   cd /workspace && git clone https://github.com/<user>/gradience.git
#   cd gradience && pip install -e ".[dev]"
#   pip install datasets transformers peft accelerate
#   tmux new -s training
#   bash scripts/m1_experiment/run_lead_lag_v2.sh 2>&1 | tee /workspace/lead_lag_v2.log

set -euo pipefail

CONFIG="scripts/m1_experiment/m1_config.yaml"
COMMON_ARGS="--eval-steps 25 --logging-steps 10 --structural-every-n 25"

echo "============================================================"
echo "LEAD-LAG V2: 3 new conditions (1 seed each)"
echo "  All with structural-every-n=25 (matching eval interval)"
echo "============================================================"

# --- Condition 1: chat / r=8 (constrained capacity, easy task) ---
echo ""
echo "[1/3] chat / r=8 / seed=42"
echo "------------------------------------------------------------"
python scripts/m1_experiment/phase1_train_telemetry.py \
    --config "$CONFIG" \
    --task chat \
    --lora-rank 8 \
    $COMMON_ARGS

# --- Condition 2: math / r=32 (full capacity, hard task) ---
echo ""
echo "[2/3] math / r=32 / seed=42"
echo "------------------------------------------------------------"
python scripts/m1_experiment/phase1_train_telemetry.py \
    --config "$CONFIG" \
    --task math \
    $COMMON_ARGS

# --- Condition 3: math / r=8 (constrained capacity, hard task) ---
echo ""
echo "[3/3] math / r=8 / seed=42"
echo "------------------------------------------------------------"
python scripts/m1_experiment/phase1_train_telemetry.py \
    --config "$CONFIG" \
    --task math \
    --lora-rank 8 \
    $COMMON_ARGS

echo ""
echo "============================================================"
echo "DONE. Output directories:"
echo "  adapters/chat_r8/seed_42/telemetry/run.jsonl"
echo "  adapters/math/seed_42/telemetry/run.jsonl"
echo "  adapters/math_r8/seed_42/telemetry/run.jsonl"
echo ""
echo "Download telemetry for analysis:"
echo "  scp -r runpod:/workspace/m1/adapters/{chat_r8,math,math_r8} results/lead_lag/workspace/m1/adapters/"
echo "============================================================"
