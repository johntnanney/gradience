#!/usr/bin/env bash
set -euo pipefail

# Parse arguments
USE_HF=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --hf)
      USE_HF="1"
      shift
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--hf]"
      echo "  --hf    Use HuggingFace Trainer instead of toy_lora_run"
      exit 1
      ;;
  esac
done

# Always run from repo root
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if [[ -n "$USE_HF" ]]; then
  echo "========================================================================"
  echo "GRADIENCE SMOKE TEST (HuggingFace Mode)"
  echo "========================================================================"
else
  echo "========================================================================"
  echo "GRADIENCE SMOKE TEST"
  echo "========================================================================"
fi
echo "repo: $ROOT"

# Basic preflight: deps + CLI availability
python3 - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "peft", "safetensors", "datasets"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)

if missing:
    print("❌ Missing dependencies:", ", ".join(missing))
    print("Install (example): pip install torch transformers peft safetensors datasets")
    sys.exit(1)

print("✅ Python deps ok")
PY

if ! command -v gradience >/dev/null 2>&1; then
  echo "❌ gradience command not found."
  echo "Install in this repo with: pip install -e ."
  exit 1
fi

# Pick device conservatively:
# - prefer cuda if available
# - else prefer mps if available
# - else cpu
DEVICE="${GRADIENCE_SMOKE_DEVICE:-}"
if [ -z "$DEVICE" ]; then
  DEVICE="$(python3 - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
)"
fi

TASK="${GRADIENCE_SMOKE_TASK:-sst2}"
MAX_STEPS="${GRADIENCE_SMOKE_MAX_STEPS:-25}"
TRAIN_SAMPLES="${GRADIENCE_SMOKE_TRAIN_SAMPLES:-128}"
EVAL_SAMPLES="${GRADIENCE_SMOKE_EVAL_SAMPLES:-128}"

OUT_BASE="${GRADIENCE_SMOKE_OUT_BASE:-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$USE_HF" ]]; then
  OUT="${OUT_BASE}/smoke_hf_${STAMP}"
else
  OUT="${OUT_BASE}/smoke_${STAMP}"
fi

echo "out:    $OUT"
if [[ -z "$USE_HF" ]]; then
  echo "device: $DEVICE"
  echo "task:   $TASK"
  echo "steps:  $MAX_STEPS"
fi
echo

mkdir -p "$OUT"

if [[ -n "$USE_HF" ]]; then
  # 1) HF trainer run (CPU-only, fast)
  echo "---- (1/6) HF trainer run ----"
  GRADIENCE_OUTPUT_DIR="$OUT" python3 examples/vnext/hf_trainer_example.py
  
  test -f "$OUT/run.jsonl" || { echo "❌ Expected $OUT/run.jsonl"; exit 1; }
  test -f "$OUT/adapter_config.json" || { echo "❌ Expected $OUT/adapter_config.json"; exit 1; }
  
  # For HF mode, we audit the output dir directly (not peft/ subdir)
  PEFT_DIR="$OUT"
  TRAINING_DIR=""
else
  # 1) Toy LoRA run (writes run.jsonl + peft/ + training/)
  echo "---- (1/6) toy run ----"
  python3 examples/vnext/toy_lora_run.py \
    --out "$OUT" \
    --device "$DEVICE" \
    --max-steps "$MAX_STEPS" \
    --train-samples "$TRAIN_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES"

  test -f "$OUT/run.jsonl" || { echo "❌ Expected $OUT/run.jsonl"; exit 1; }
  test -d "$OUT/peft" || { echo "❌ Expected $OUT/peft/"; exit 1; }
  test -d "$OUT/training" || { echo "❌ Expected $OUT/training/"; exit 1; }
  
  PEFT_DIR="$OUT/peft"
  TRAINING_DIR="$OUT/training"
fi

# 2) Check (pre-flight)
echo "---- (2/6) check ----"
if [[ -n "$USE_HF" ]]; then
  # HF mode: check with peft-dir only (no training-dir since it's embedded in adapter)
  gradience check --task sst2 --peft-dir "$PEFT_DIR" | tee "$OUT/check.txt" >/dev/null
else
  # Traditional mode: use both peft and training dirs
  gradience check --task "$TASK" --peft-dir "$PEFT_DIR" --training-dir "$TRAINING_DIR" | tee "$OUT/check.txt" >/dev/null
fi

# 3) Monitor (pre-audit)
echo "---- (3/6) monitor (pre-audit) ----"
gradience monitor "$OUT/run.jsonl" --json > "$OUT/monitor_pre_audit.json"
python3 - <<'PY' "$OUT/monitor_pre_audit.json"
import json,sys
p=sys.argv[1]
with open(p,"r",encoding="utf-8") as f:
    json.load(f)
print("✅ monitor JSON parses:", p)
PY

# 4) Audit (json)
echo "---- (4/6) audit ----"
gradience audit --peft-dir "$PEFT_DIR" --json > "$OUT/audit.json"
python3 - <<'PY' "$OUT/audit.json"
import json,sys
p=sys.argv[1]
with open(p,"r",encoding="utf-8") as f:
    json.load(f)
print("✅ audit JSON parses:", p)
PY

# 5) Append audit event to telemetry
echo "---- (5/6) audit --append ----"
gradience audit --peft-dir "$PEFT_DIR" --append "$OUT/run.jsonl" | tee "$OUT/audit_append.txt" >/dev/null

# 6) Monitor again (post-audit) and verify lora_audit shows up somewhere
echo "---- (6/6) monitor (post-audit) ----"
gradience monitor "$OUT/run.jsonl" --json > "$OUT/monitor_post_audit.json"
python3 - <<'PY' "$OUT/monitor_post_audit.json"
import json,sys
p=sys.argv[1]
with open(p,"r",encoding="utf-8") as f:
    j=json.load(f)

def contains(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(contains(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(contains(v, key) for v in obj)
    return False

if not contains(j, "lora_audit"):
    raise SystemExit("❌ expected monitor summary to contain 'lora_audit' after --append")
print("✅ lora_audit present in post-audit monitor summary")
print("✅ monitor post-audit JSON parses:", p)
PY

echo
if [[ -n "$USE_HF" ]]; then
  echo "✅ HF SMOKE TEST PASSED"
  echo "Artifacts:"
  echo "  $OUT/run.jsonl"
  echo "  $OUT/adapter_config.json"
  echo "  $OUT/check.txt"
  echo "  $OUT/monitor_pre_audit.json"
  echo "  $OUT/audit.json"
  echo "  $OUT/monitor_post_audit.json"
  echo ""
  echo "This validates:"
  echo "  • HF callback: GradienceCallback()"
  echo "  • Telemetry generation + validation"
  echo "  • Monitor/audit compatibility with HF adapters"
else
  echo "✅ SMOKE TEST PASSED"
  echo "Artifacts:"
  echo "  $OUT/run.jsonl"
  echo "  $OUT/check.txt"
  echo "  $OUT/monitor_pre_audit.json"
  echo "  $OUT/audit.json"
  echo "  $OUT/monitor_post_audit.json"
fi