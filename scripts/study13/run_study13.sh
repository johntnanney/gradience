#!/usr/bin/env bash
# ============================================================
# Study 13: Grokking DFA Experiment — Multi-Seed Runner
# ============================================================
#
# Purpose: Run modular addition grokking with dense Hessian telemetry
# for windowed DFA analysis of the phase transition.
#
# Expected output:
#   results/study13/seed_<N>/telemetry.jsonl  (12 seeds)
#
# Estimated runtime:
#   ~45 min per seed on A100
#   ~9 hours sequential, ~2.5 hours with 4-way parallelism
# ============================================================

set -euo pipefail

# ---------------------------
# Configuration
# ---------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/study13_train_grokking.py"

# Seeds
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53}"

# Training params
MAX_STEPS="${MAX_STEPS:-80000}"
HESSIAN_EVERY="${HESSIAN_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LR="${LR:-0.001}"
WD="${WD:-0.1}"
TRAIN_FRAC="${TRAIN_FRAC:-0.3}"

# Results root
ROOT="${ROOT:-results/study13}"
mkdir -p "$ROOT"

# Parallelism (set to 1 for sequential)
N_PARALLEL="${N_PARALLEL:-1}"

echo "============================================================"
echo "Study 13: Grokking DFA Experiment"
echo "  Seeds:        $SEEDS"
echo "  Max steps:    $MAX_STEPS"
echo "  Hessian freq: every $HESSIAN_EVERY steps"
echo "  LR: $LR  WD: $WD  train_frac: $TRAIN_FRAC"
echo "  Output:       $ROOT"
echo "  Parallelism:  $N_PARALLEL"
echo "============================================================"
echo

# ---------------------------
# Run function
# ---------------------------
run_one () {
  local seed="$1"
  local out_dir="${ROOT}/seed_${seed}"

  # Resume-safe: skip if telemetry already has enough records
  if [[ -f "${out_dir}/telemetry.jsonl" ]]; then
    local n_lines
    n_lines=$(wc -l < "${out_dir}/telemetry.jsonl")
    if [[ $n_lines -gt 100 ]]; then
      echo "[SKIP] seed_${seed} (telemetry.jsonl exists, ${n_lines} lines)"
      return 0
    fi
  fi

  echo
  echo "============================================================"
  echo "[RUN] seed=${seed}  out_dir=${out_dir}"
  echo "============================================================"

  mkdir -p "$out_dir"

  # Record wall time
  local t0
  t0=$(python3 -c "import time; print(time.time())")

  set +e
  python3 "$TRAIN_SCRIPT" \
    --seed="$seed" \
    --out_dir="$out_dir" \
    --max_steps="$MAX_STEPS" \
    --hessian_every="$HESSIAN_EVERY" \
    --eval_every="$EVAL_EVERY" \
    --lr="$LR" \
    --wd="$WD" \
    --train_frac="$TRAIN_FRAC" \
    2>&1 | tee "${out_dir}/console.log"
  local rc=${PIPESTATUS[0]}
  set -e

  local t1
  t1=$(python3 -c "import time; print(time.time())")
  T0="$t0" T1="$t1" python3 -c \
    "import os; t0=float(os.environ['T0']); t1=float(os.environ['T1']); print(f'wall_seconds {t1-t0:.1f}')" \
    > "${out_dir}/time.txt"

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] seed_${seed} (exit=$rc)" >&2
    echo "seed_${seed} FAILED rc=$rc" >> "${ROOT}/failures.log"
    return 1
  fi

  # Quick check
  local n_lines
  n_lines=$(wc -l < "${out_dir}/telemetry.jsonl" 2>/dev/null || echo 0)

  # Check if it grokked
  local grokked
  grokked=$(python3 -c "
import json
with open('${out_dir}/summary.json') as f:
    s = json.load(f)
print('YES' if s.get('grokked') else 'NO')
" 2>/dev/null || echo "UNKNOWN")

  echo "[OK] seed_${seed}: ${n_lines} records, grokked=${grokked}"
}

# ---------------------------
# Main loop
# ---------------------------
main () {
  local n_total=0
  local n_done=0
  local n_fail=0

  for seed in $SEEDS; do
    n_total=$((n_total + 1))
  done

  echo "Total runs planned: $n_total"
  echo

  if [[ $N_PARALLEL -eq 1 ]]; then
    # Sequential execution
    for seed in $SEEDS; do
      if run_one "$seed"; then
        n_done=$((n_done + 1))
      else
        n_fail=$((n_fail + 1))
      fi
      echo "  Progress: $((n_done + n_fail))/$n_total (${n_fail} failures)"
    done
  else
    # Parallel execution
    local pids=()
    local seed_list=($SEEDS)
    local i=0

    while [[ $i -lt ${#seed_list[@]} ]]; do
      # Launch up to N_PARALLEL jobs
      pids=()
      for (( j=0; j<N_PARALLEL && i<${#seed_list[@]}; j++, i++ )); do
        local seed="${seed_list[$i]}"
        run_one "$seed" &
        pids+=($!)
        echo "  Launched seed_${seed} (PID ${pids[-1]})"
      done

      # Wait for batch
      for pid in "${pids[@]}"; do
        if wait "$pid"; then
          n_done=$((n_done + 1))
        else
          n_fail=$((n_fail + 1))
        fi
      done
      echo "  Progress: $((n_done + n_fail))/$n_total (${n_fail} failures)"
    done
  fi

  echo
  echo "============================================================"
  echo "COMPLETE"
  echo "  Total: $n_total"
  echo "  Success: $n_done"
  echo "  Failed: $n_fail"
  echo "  Output: $ROOT"
  echo "============================================================"

  # Summary table
  echo
  echo "Per-seed summary:"
  for seed in $SEEDS; do
    local summary="${ROOT}/seed_${seed}/summary.json"
    if [[ -f "$summary" ]]; then
      python3 -c "
import json
with open('$summary') as f:
    s = json.load(f)
grok = f'GROKKED@{s[\"grok_step\"]}' if s.get('grokked') else 'NO_GROK'
print(f'  seed {s[\"seed\"]:3d}: val_acc={s[\"final_val_acc\"]:.4f}  {grok}  ({s[\"elapsed_seconds\"]:.0f}s)')
" 2>/dev/null || echo "  seed ${seed}: NO SUMMARY"
    else
      echo "  seed ${seed}: NOT RUN"
    fi
  done

  if [[ $n_fail -gt 0 ]]; then
    echo
    echo "FAILURES:"
    cat "${ROOT}/failures.log"
    exit 1
  fi
}

main
