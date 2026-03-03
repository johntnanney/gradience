#!/usr/bin/env bash
# ============================================================
# Study 12: Replication Experiment
# ============================================================
#
# Purpose: Replicate the Study 11 regime classification with 10 seeds
# per regime (up from 3), using the alwayson spectral arm only.
# This resolves the n=15 limitation that prevented statistical
# significance in the geometry-vs-loss comparison.
#
# Expected output:
#   results/study12_replication/<regime>_<params>_seed<N>/telemetry.jsonl
#   50 runs total (5 regimes × 10 seeds)
#
# Estimated runtime: 4–8 hours single-GPU, <1 hour if parallelised
# ============================================================

set -euo pipefail

cd /workspace/nanogpt_gradience

# ---------------------------
# Common training config
# ---------------------------
CONFIG="config/train_shakespeare_char.py"
MAX_ITERS="${MAX_ITERS:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# Spectral settings (identical to Study 11 rngfix_alwayson arm)
SPECTRAL_MAX_MATRICES="${SPECTRAL_MAX_MATRICES:-8}"
SPECTRAL_POWER_ITERS="${SPECTRAL_POWER_ITERS:-8}"
SPECTRAL_POWER_THRESHOLD="${SPECTRAL_POWER_THRESHOLD:-0}"

# ---------------------------
# Seeds: 10 per regime
# ---------------------------
# Seeds 1337–1339 match Study 11 exactly (replication of originals).
# Seeds 1340–1346 are new (7 additional seeds per regime).
SEEDS="${SEEDS:-1337 1338 1339 1340 1341 1342 1343 1344 1345 1346}"
REGIMES="${REGIMES:-baseline low_wd high_wd low_lr high_lr}"

# Results root
ROOT="results/study12_replication"
mkdir -p "$ROOT"

echo "============================================================"
echo "Study 12: Replication Experiment"
echo "  Regimes: $REGIMES"
echo "  Seeds:   $SEEDS"
echo "  Config:  $CONFIG"
echo "  Max iters: $MAX_ITERS"
echo "  Output:  $ROOT"
echo "============================================================"

# ---------------------------
# Regime parameter table
# (identical to Study 11)
# ---------------------------
declare -A LR WD LRTAG WDTAG

# Baseline: lr=1e-3, wd=0.1
LR[baseline]="0.001"
WD[baseline]="0.1"
LRTAG[baseline]="1em03"
WDTAG[baseline]="0p1"

# Low weight decay: lr=1e-3, wd=0
LR[low_wd]="0.001"
WD[low_wd]="0.0"
LRTAG[low_wd]="1em03"
WDTAG[low_wd]="0"

# High weight decay: lr=1e-3, wd=1.0
LR[high_wd]="0.001"
WD[high_wd]="1.0"
LRTAG[high_wd]="1em03"
WDTAG[high_wd]="1"

# Low LR: lr=1e-4, wd=0.1
LR[low_lr]="0.0001"
WD[low_lr]="0.1"
LRTAG[low_lr]="1em04"
WDTAG[low_lr]="0p1"

# High LR: lr=1e-2, wd=0.1
LR[high_lr]="0.01"
WD[high_lr]="0.1"
LRTAG[high_lr]="1em02"
WDTAG[high_lr]="0p1"

# ---------------------------
# Run function
# ---------------------------
run_one () {
  local regime="$1"
  local seed="$2"
  local run_id="${regime}_lr${LRTAG[$regime]}_wd${WDTAG[$regime]}_seed${seed}"
  local out_dir="${ROOT}/${run_id}"

  # Skip if already complete (resume-safe)
  if [[ -f "${out_dir}/telemetry.jsonl" ]]; then
    local n_lines=$(wc -l < "${out_dir}/telemetry.jsonl")
    if [[ $n_lines -gt 10 ]]; then
      echo "[SKIP] ${run_id} (telemetry.jsonl exists, ${n_lines} lines)"
      return 0
    fi
  fi

  mkdir -p "$out_dir"

  echo
  echo "============================================================"
  echo "[RUN] run_id=${run_id}"
  echo "  lr=${LR[$regime]}  wd=${WD[$regime]}  seed=${seed}"
  echo "  out_dir=${out_dir}"
  echo "============================================================"

  # Alwayson arm: spectral computed every LOG_INTERVAL steps
  local args=(
    python train_gradience.py "$CONFIG"
    --out_dir="$out_dir"
    --max_iters="$MAX_ITERS"
    --eval_interval="$EVAL_INTERVAL"
    --log_interval="$LOG_INTERVAL"
    --seed="$seed"
    --learning_rate="${LR[$regime]}"
    --weight_decay="${WD[$regime]}"
    --wandb_log=False
    --always_save_checkpoint=False
    --compile=False
    --telemetry_enabled=True
    --telemetry_spectral_enabled=True
    --controller_enabled=False
    --spectral_interval="$LOG_INTERVAL"
    --spectral_max_matrices="$SPECTRAL_MAX_MATRICES"
    --spectral_power_iters="$SPECTRAL_POWER_ITERS"
    --spectral_power_threshold="$SPECTRAL_POWER_THRESHOLD"
  )

  # Record wall time
  t0=$(python -c "import time; print(time.time())")
  set +e
  "${args[@]}" 2>&1 | tee "${out_dir}/console.log"
  rc=${PIPESTATUS[0]}
  set -e
  t1=$(python -c "import time; print(time.time())")
  T0="$t0" T1="$t1" python -c "import os; t0=float(os.environ['T0']); t1=float(os.environ['T1']); print(f'wall_seconds {t1-t0:.1f}')" > "${out_dir}/time.txt"

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${run_id} (exit=$rc)" >&2
    # Don't exit — continue to next run
    echo "$run_id FAILED rc=$rc" >> "${ROOT}/failures.log"
    return 1
  fi

  # Quick sanity check
  local n_lines=$(wc -l < "${out_dir}/telemetry.jsonl" 2>/dev/null || echo 0)
  echo "[OK] ${run_id}: ${n_lines} telemetry records"
}

# ---------------------------
# Main loop
# ---------------------------
main () {
  local n_total=0
  local n_done=0
  local n_fail=0

  for regime in $REGIMES; do
    for seed in $SEEDS; do
      n_total=$((n_total + 1))
    done
  done

  echo "Total runs planned: $n_total"
  echo

  for regime in $REGIMES; do
    for seed in $SEEDS; do
      if run_one "$regime" "$seed"; then
        n_done=$((n_done + 1))
      else
        n_fail=$((n_fail + 1))
      fi
      echo "  Progress: $((n_done + n_fail))/$n_total (${n_fail} failures)"
    done
  done

  echo
  echo "============================================================"
  echo "COMPLETE"
  echo "  Total: $n_total"
  echo "  Success: $n_done"
  echo "  Failed: $n_fail"
  echo "  Output: $ROOT"
  echo "============================================================"

  if [[ $n_fail -gt 0 ]]; then
    echo "FAILURES:"
    cat "${ROOT}/failures.log"
    exit 1
  fi
}

main
