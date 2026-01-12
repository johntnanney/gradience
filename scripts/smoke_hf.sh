#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

echo "========================================================================"
echo "GRADIENCE HF SMOKE TEST"
echo "========================================================================"
echo "repo: $ROOT"

# HF-specific preflight: transformers, peft, datasets
python3 - <<'PY'
import importlib, sys
mods = ["transformers", "peft", "datasets"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)

if missing:
    print("❌ Missing HF dependencies:", ", ".join(missing))
    print("Install with: pip install transformers peft datasets")
    sys.exit(1)

print("✅ HuggingFace deps ok")
PY

if ! command -v gradience >/dev/null 2>&1; then
  echo "❌ gradience command not found."
  echo "Install in this repo with: pip install -e ."
  exit 1
fi

# Test HF callback import (critical regression catcher)
echo "---- Testing HF callback import ----"
python3 - <<'PY'
try:
    from gradience.vnext.integrations.hf import GradienceCallback
    print("✅ HF callback import successful")
except ImportError as e:
    print(f"❌ HF callback import failed: {e}")
    import sys
    sys.exit(1)
PY

OUT_BASE="${GRADIENCE_SMOKE_OUT_BASE:-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_BASE}/smoke_hf_${STAMP}"

echo "out: $OUT"
echo

mkdir -p "$OUT"

# 1) Run HF trainer example (CPU-only, fast)
echo "---- (1/6) HF trainer example ----"
cd "$ROOT"
GRADIENCE_OUTPUT_DIR="$OUT" python3 examples/vnext/hf_trainer_example.py

# Check expected outputs exist
test -f "$OUT/run.jsonl" || { echo "❌ Expected $OUT/run.jsonl"; exit 1; }
test -f "$OUT/adapter_config.json" || { echo "❌ Expected $OUT/adapter_config.json"; exit 1; }

echo "✅ HF callback generated telemetry and adapter"

# 2) Validate telemetry schema
echo "---- (2/6) validate telemetry ----"
python3 - <<PY "$OUT/run.jsonl"
import sys
from gradience.vnext.telemetry_reader import TelemetryReader

telemetry_file = sys.argv[1]
reader = TelemetryReader(telemetry_file, strict_schema=True)
issues = reader.validate()

if issues:
    print(f"❌ Telemetry validation failed: {issues}")
    sys.exit(1)

print("✅ Telemetry schema validation passed")
PY

# 3) Monitor (verify HF telemetry works with monitor)
echo "---- (3/6) monitor HF telemetry ----"
gradience monitor "$OUT/run.jsonl" --json > "$OUT/monitor.json"
python3 - <<'PY' "$OUT/monitor.json" "$OUT/run.jsonl"
import json, sys
monitor_file = sys.argv[1]
telemetry_file = sys.argv[2]

# Check monitor JSON parses
with open(monitor_file, "r", encoding="utf-8") as f:
    monitor_data = json.load(f)

# Check framework metadata in original telemetry
with open(telemetry_file, "r", encoding="utf-8") as f:
    first_line = f.readline()
    run_start = json.loads(first_line)
    
if not run_start.get("meta", {}).get("framework") == "huggingface":
    print(f"❌ Expected framework=huggingface in telemetry meta")
    sys.exit(1)

print("✅ HuggingFace framework detected in telemetry")
print("✅ Monitor JSON parses:", monitor_file)
PY

# 4) Audit the PEFT adapter
echo "---- (4/6) audit PEFT adapter ----"
gradience audit --peft-dir "$OUT" --json > "$OUT/audit.json"
python3 - <<'PY' "$OUT/audit.json"
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    audit = json.load(f)

# Basic audit validation
if "total_lora_params" not in audit:
    print(f"❌ Expected total_lora_params in audit output")
    sys.exit(1)

print("✅ Audit JSON parses:", p)
print(f"✅ Found {audit['total_lora_params']} LoRA parameters")
PY

# 5) Test rank suggestions on HF adapter
echo "---- (5/6) rank suggestions ----"
python3 - <<'PY' "$OUT/audit.json"
import json, sys
sys.path.append('/Users/john/code/gradience')

from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit

with open(sys.argv[1], "r") as f:
    audit = json.load(f)

try:
    suggestions = suggest_global_ranks_from_audit(audit)
    print(f"✅ Rank suggestions: r={suggestions.current_r} → {suggestions.suggested_r_median}/{suggestions.suggested_r_p90}")
except Exception as e:
    print(f"❌ Rank suggestion failed: {e}")
    sys.exit(1)
PY

# 6) Test config extraction from HF callback
echo "---- (6/6) config extraction ----"
python3 - <<'PY' "$OUT/run.jsonl"
import sys
from gradience.vnext.telemetry_reader import TelemetryReader

reader = TelemetryReader(sys.argv[1])
config = reader.latest_config()

# Verify HF-specific config was captured
if not config.model_name:
    print("❌ Expected model_name in config")
    sys.exit(1)

if not config.optimizer.lr:
    print("❌ Expected learning_rate in optimizer config")
    sys.exit(1)

print(f"✅ Config extracted: model={config.model_name}, lr={config.optimizer.lr}")
print(f"✅ LoRA config: r={config.lora.r}, alpha={config.lora.alpha}")
PY

echo
echo "✅ HF SMOKE TEST PASSED"
echo "Artifacts:"
echo "  $OUT/run.jsonl"
echo "  $OUT/adapter_config.json" 
echo "  $OUT/monitor.json"
echo "  $OUT/audit.json"
echo
echo "This validates:"
echo "  • HF callback import (critical regression catcher)"
echo "  • One-line integration: GradienceCallback()"
echo "  • Telemetry generation + validation"
echo "  • Monitor/audit compatibility with HF-generated data"
echo "  • Config extraction from TrainingArguments"
echo "  • LoRA adapter compatibility"