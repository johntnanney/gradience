#!/usr/bin/env bash
set -euo pipefail

# No-tmux friendly bench runner with comprehensive state tracking
# Prevents "where was I?" archaeology when kicked off pods
#
# Usage: scripts/bench/run_seed_nohup.sh --config CONFIG --output OUTPUT [OPTIONS]
#
# Always writes state files:
#   OUTPUT/_pid.txt       - Process ID for monitoring/killing
#   OUTPUT/STAGE.txt      - Current execution stage  
#   OUTPUT/_exit_code.txt - Final exit code
#   OUTPUT/nohup.log      - Complete execution log

# Default values
CONFIG=""
OUTPUT=""
BACKGROUND=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2" 
      shift 2
      ;;
    --background|-bg)
      BACKGROUND=true
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/bench/run_seed_nohup.sh --config CONFIG --output OUTPUT [OPTIONS]

No-tmux friendly bench runner with comprehensive state tracking.
Prevents "where was I?" archaeology when kicked off pods.

Required Arguments:
  --config CONFIG      Path to bench config YAML file
  --output OUTPUT      Output directory for results and state files

Optional Arguments:
  --background, -bg    Run in background with nohup (default: foreground)
  --verbose, -v        Enable verbose logging
  -h, --help          Show this help

State Files (written to OUTPUT/):
  _pid.txt            Process ID for monitoring/killing  
  STAGE.txt           Current execution stage
  _exit_code.txt      Final exit code (0=success, >0=error)
  nohup.log           Complete execution log

Environment Variables:
  GRADIENCE_NOHUP_NICE     Nice level for background process (default: 10)

Examples:
  # Foreground (SSH session)  
  scripts/bench/run_seed_nohup.sh --config configs/mistral_gsm8k.yaml --output runs/test

  # Background (survives disconnection)
  scripts/bench/run_seed_nohup.sh --config configs/mistral_gsm8k.yaml --output runs/test --background

  # Monitor background run
  tail -f runs/test/nohup.log
  cat runs/test/STAGE.txt
  kill $(cat runs/test/_pid.txt)  # stop if needed
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$CONFIG" ]]; then
    echo "❌ --config is required"
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    echo "❌ --output is required"  
    exit 1
fi

# Always run from repo root
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# Validate config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT"

# Convert to absolute paths for nohup safety
OUTPUT_ABS="$(cd "$OUTPUT" && pwd)"
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# Setup state tracking
PID_FILE="$OUTPUT_ABS/_pid.txt"
STAGE_FILE="$OUTPUT_ABS/STAGE.txt" 
EXIT_CODE_FILE="$OUTPUT_ABS/_exit_code.txt"
LOG_FILE="$OUTPUT_ABS/nohup.log"

# Function to update stage
update_stage() {
    local stage="$1"
    echo "$stage" > "$STAGE_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [STAGE] $stage" >> "$LOG_FILE"
    if [[ "$VERBOSE" == "true" ]]; then
        echo "🔄 Stage: $stage"
    fi
}

# Function to write exit code and cleanup
write_exit_code() {
    local code="$1"
    echo "$code" > "$EXIT_CODE_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [EXIT] Code: $code" >> "$LOG_FILE"
    
    # Remove PID file when done
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi
}

# Main execution function
run_bench() {
    # Initialize log file
    echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Bench runner starting" > "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Config: $CONFIG_ABS" >> "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Output: $OUTPUT_ABS" >> "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] PID: $$" >> "$LOG_FILE"
    
    # Write PID immediately
    echo "$$" > "$PID_FILE"
    
    # Setup trap to ensure cleanup on exit
    trap 'write_exit_code $?' EXIT
    
    update_stage "initializing"
    
    # Preflight checks
    update_stage "preflight_checks"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running preflight checks..." >> "$LOG_FILE"
    
    # Check dependencies
    python3 - <<'PY' >> "$LOG_FILE" 2>&1
import importlib, sys
required_modules = ["torch", "transformers", "peft", "safetensors", "datasets", "yaml"]
missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module}")
    except Exception as e:
        missing.append(module)
        print(f"❌ {module}: {e}")

if missing:
    print(f"❌ Missing dependencies: {missing}")
    sys.exit(1)
else:
    print("✅ All dependencies available")
PY
    
    if [[ $? -ne 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Preflight dependency check failed" >> "$LOG_FILE"
        exit 1
    fi
    
    # Check config file validity
    python3 - <<PY "$CONFIG_ABS" >> "$LOG_FILE" 2>&1
import yaml, sys
config_path = sys.argv[1]
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config valid: {config_path}")
    print(f"   Model: {config.get('model', {}).get('name', 'unknown')}")
    print(f"   Device: {config.get('runtime', {}).get('device', 'auto')}")
except Exception as e:
    print(f"❌ Config validation failed: {e}")
    sys.exit(1)
PY
    
    if [[ $? -ne 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Config validation failed" >> "$LOG_FILE"
        exit 1
    fi
    
    # Check bench CLI
    if ! python3 -m gradience.bench.run_bench --help >/dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] gradience.bench.run_bench not available" >> "$LOG_FILE"
        exit 1
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Preflight checks passed" >> "$LOG_FILE"
    
    # Run preflight check if available
    update_stage "preflight_validation"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running bench preflight validation..." >> "$LOG_FILE"
    
    python3 - <<PY "$CONFIG_ABS" >> "$LOG_FILE" 2>&1
import yaml, sys
sys.path.insert(0, '.')
try:
    from gradience.bench.protocol import run_bench_preflight_check
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_name = config.get('model', {}).get('name', 'unknown')
    run_bench_preflight_check(config, model_name)
    print("✅ Bench preflight validation passed")
except ImportError:
    print("ℹ️  Bench preflight check not available (older version)")
except Exception as e:
    print(f"⚠️  Preflight validation warning: {e}")
    # Don't fail - some warnings are acceptable
PY
    
    # Start main bench execution
    update_stage "bench_execution"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting bench execution..." >> "$LOG_FILE"
    
    # Run the actual bench command
    python3 -m gradience.bench.run_bench \
        --config "$CONFIG_ABS" \
        --output "$OUTPUT_ABS" \
        --verbose \
        >> "$LOG_FILE" 2>&1
    
    local bench_exit_code=$?
    
    if [[ $bench_exit_code -eq 0 ]]; then
        update_stage "completed"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Bench execution completed successfully" >> "$LOG_FILE"
    else
        update_stage "failed"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Bench execution failed with exit code: $bench_exit_code" >> "$LOG_FILE"
        exit $bench_exit_code
    fi
    
    # Validate outputs
    update_stage "output_validation"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Validating outputs..." >> "$LOG_FILE"
    
    local validation_errors=0
    
    # Check for required files
    for required_file in "bench.json" "bench.md" "runs.json"; do
        if [[ -f "$OUTPUT_ABS/$required_file" ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ✅ $required_file" >> "$LOG_FILE"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] ❌ $required_file missing" >> "$LOG_FILE"
            validation_errors=$((validation_errors + 1))
        fi
    done
    
    # Check for variant directories
    local variant_count=$(find "$OUTPUT_ABS" -maxdepth 1 -name "*_r*" -type d | wc -l)
    if [[ $variant_count -gt 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ✅ Found $variant_count compressed variants" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] ❌ No compressed variants found" >> "$LOG_FILE"
        validation_errors=$((validation_errors + 1))
    fi
    
    if [[ $validation_errors -eq 0 ]]; then
        update_stage "success"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Output validation passed" >> "$LOG_FILE"
    else
        update_stage "validation_failed"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Output validation failed ($validation_errors errors)" >> "$LOG_FILE"
        exit 1
    fi
}

# Print initial information
echo "========================================================================"
echo "GRADIENCE NOHUP-FRIENDLY BENCH RUNNER"  
echo "========================================================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo ""

if [[ "$BACKGROUND" == "true" ]]; then
    echo "🌙 Launching in background..."
    echo ""
    echo "State files:"
    echo "   📄 PID:        $PID_FILE"
    echo "   📄 Stage:      $STAGE_FILE"
    echo "   📄 Exit code:  $EXIT_CODE_FILE"
    echo "   📄 Log:        $LOG_FILE"
    echo ""
    echo "Monitor with:"
    echo "   tail -f $LOG_FILE"
    echo "   watch cat $STAGE_FILE"
    echo ""
    echo "Stop with:"
    echo "   kill \$(cat $PID_FILE)"
    echo ""
    
    # Use nice level for background processes
    NICE_LEVEL="${GRADIENCE_NOHUP_NICE:-10}"
    
    # Create a wrapper script for background execution
    WRAPPER_SCRIPT="$OUTPUT_ABS/_nohup_wrapper.sh"
    cat > "$WRAPPER_SCRIPT" <<'WRAPPER'
#!/bin/bash
set -euo pipefail

# Source the original script functions and variables
CONFIG_ABS="$1"
OUTPUT_ABS="$2"
VERBOSE="$3"

PID_FILE="$OUTPUT_ABS/_pid.txt"
STAGE_FILE="$OUTPUT_ABS/STAGE.txt" 
EXIT_CODE_FILE="$OUTPUT_ABS/_exit_code.txt"
LOG_FILE="$OUTPUT_ABS/nohup.log"

# Write our PID
echo "$$" > "$PID_FILE"

# Function to update stage
update_stage() {
    local stage="$1"
    echo "$stage" > "$STAGE_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [STAGE] $stage" >> "$LOG_FILE"
    if [[ "$VERBOSE" == "true" ]]; then
        echo "🔄 Stage: $stage" >> "$LOG_FILE"
    fi
}

# Function to write exit code and cleanup
write_exit_code() {
    local code="$1"
    echo "$code" > "$EXIT_CODE_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [EXIT] Code: $code" >> "$LOG_FILE"
    
    # Remove PID file when done
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi
}

# Setup trap to ensure cleanup on exit
trap 'write_exit_code $?' EXIT

# Initialize log file
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Background bench runner starting" > "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Config: $CONFIG_ABS" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Output: $OUTPUT_ABS" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] PID: $$" >> "$LOG_FILE"

update_stage "initializing"

# Preflight checks
update_stage "preflight_checks"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running preflight checks..." >> "$LOG_FILE"

# Check dependencies
python3 - <<'PY' >> "$LOG_FILE" 2>&1
import importlib, sys
required_modules = ["torch", "transformers", "peft", "safetensors", "datasets", "yaml"]
missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module}")
    except Exception as e:
        missing.append(module)
        print(f"❌ {module}: {e}")

if missing:
    print(f"❌ Missing dependencies: {missing}")
    sys.exit(1)
else:
    print("✅ All dependencies available")
PY

if [[ $? -ne 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Preflight dependency check failed" >> "$LOG_FILE"
    exit 1
fi

# Check config file validity
python3 - <<PY "$CONFIG_ABS" >> "$LOG_FILE" 2>&1
import yaml, sys
config_path = sys.argv[1]
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config valid: {config_path}")
    print(f"   Model: {config.get('model', {}).get('name', 'unknown')}")
    print(f"   Device: {config.get('runtime', {}).get('device', 'auto')}")
except Exception as e:
    print(f"❌ Config validation failed: {e}")
    sys.exit(1)
PY

if [[ $? -ne 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Config validation failed" >> "$LOG_FILE"
    exit 1
fi

# Check bench CLI
if ! python3 -m gradience.bench.run_bench --help >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] gradience.bench.run_bench not available" >> "$LOG_FILE"
    exit 1
fi
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Preflight checks passed" >> "$LOG_FILE"

# Run preflight check if available
update_stage "preflight_validation"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running bench preflight validation..." >> "$LOG_FILE"

python3 - <<PY "$CONFIG_ABS" >> "$LOG_FILE" 2>&1
import yaml, sys
sys.path.insert(0, '.')
try:
    from gradience.bench.protocol import run_bench_preflight_check
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_name = config.get('model', {}).get('name', 'unknown')
    run_bench_preflight_check(config, model_name)
    print("✅ Bench preflight validation passed")
except ImportError:
    print("ℹ️  Bench preflight check not available (older version)")
except Exception as e:
    print(f"⚠️  Preflight validation warning: {e}")
    # Don't fail - some warnings are acceptable
PY

# Start main bench execution
update_stage "bench_execution"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting bench execution..." >> "$LOG_FILE"

# Run the actual bench command
python3 -m gradience.bench.run_bench \
    --config "$CONFIG_ABS" \
    --output "$OUTPUT_ABS" \
    --verbose \
    >> "$LOG_FILE" 2>&1

local bench_exit_code=$?

if [[ $bench_exit_code -eq 0 ]]; then
    update_stage "completed"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Bench execution completed successfully" >> "$LOG_FILE"
else
    update_stage "failed"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Bench execution failed with exit code: $bench_exit_code" >> "$LOG_FILE"
    exit $bench_exit_code
fi

# Validate outputs
update_stage "output_validation"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Validating outputs..." >> "$LOG_FILE"

local validation_errors=0

# Check for required files
for required_file in "bench.json" "bench.md" "runs.json"; do
    if [[ -f "$OUTPUT_ABS/$required_file" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ✅ $required_file" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] ❌ $required_file missing" >> "$LOG_FILE"
        validation_errors=$((validation_errors + 1))
    fi
done

# Check for variant directories
local variant_count=$(find "$OUTPUT_ABS" -maxdepth 1 -name "*_r*" -type d | wc -l)
if [[ $variant_count -gt 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ✅ Found $variant_count compressed variants" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] ❌ No compressed variants found" >> "$LOG_FILE"
    validation_errors=$((validation_errors + 1))
fi

if [[ $validation_errors -eq 0 ]]; then
    update_stage "success"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Output validation passed" >> "$LOG_FILE"
else
    update_stage "validation_failed"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Output validation failed ($validation_errors errors)" >> "$LOG_FILE"
    exit 1
fi

# Clean up wrapper script
rm -f "$OUTPUT_ABS/_nohup_wrapper.sh"
WRAPPER
    
    chmod +x "$WRAPPER_SCRIPT"
    
    # Launch in background
    nohup nice -n "$NICE_LEVEL" "$WRAPPER_SCRIPT" "$CONFIG_ABS" "$OUTPUT_ABS" "$VERBOSE" >/dev/null 2>&1 &
    
    # Write the background PID
    BACKGROUND_PID=$!
    
    # Wait a moment for the wrapper to start and write its own PID
    sleep 1
    
    echo "🚀 Background process launched (PID: $BACKGROUND_PID)"
    echo ""
else
    echo "🖥️  Running in foreground..."
    echo ""
    echo "Log file: $LOG_FILE"
    echo "Stage tracking: $STAGE_FILE"
    echo ""
    
    # Run in foreground
    run_bench
    
    echo ""
    echo "✅ Bench execution completed"
fi