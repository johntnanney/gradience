#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

echo "========================================================================"
echo "GRADIENCE SMOKE TEST"
echo "========================================================================"
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
OUT="${OUT_BASE}/smoke_${STAMP}"

echo "out:    $OUT"
echo "device: $DEVICE"
echo "task:   $TASK"
echo "steps:  $MAX_STEPS"
echo

mkdir -p "$OUT"

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

# 2) Check (pre-flight)
echo "---- (2/6) check ----"
gradience check --task "$TASK" --peft-dir "$OUT/peft" --training-dir "$OUT/training" | tee "$OUT/check.txt" >/dev/null

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
gradience audit --peft-dir "$OUT/peft" --json > "$OUT/audit.json"
python3 - <<'PY' "$OUT/audit.json"
import json,sys
p=sys.argv[1]
with open(p,"r",encoding="utf-8") as f:
    json.load(f)
print("✅ audit JSON parses:", p)
PY

# 5) Append audit event to telemetry
echo "---- (5/6) audit --append ----"
gradience audit --peft-dir "$OUT/peft" --append "$OUT/run.jsonl" | tee "$OUT/audit_append.txt" >/dev/null

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
echo "✅ SMOKE TEST PASSED"
echo "Artifacts:"
echo "  $OUT/run.jsonl"
echo "  $OUT/check.txt"
echo "  $OUT/monitor_pre_audit.json"
echo "  $OUT/audit.json"
echo "  $OUT/monitor_post_audit.json"