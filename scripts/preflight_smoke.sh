#!/usr/bin/env bash
# preflight_smoke.sh — End-to-end smoke test for the Gradience artifact spine.
#
# Exercises: audit-adapter → merge-audit → summarize-inventory
# Uses the bundled examples/adapters/tiny_lora fixture.
#
# Exit 0 = all green, non-zero = something broke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "=== Gradience Preflight Smoke Test ==="
echo "Working directory: $WORK_DIR"
echo ""

ADAPTER_DIR="$REPO_ROOT/examples/adapters/tiny_lora"
QA_DIR="$WORK_DIR/qa"
REPORT_DIR="$WORK_DIR/reports"
INVENTORY_DIR="$WORK_DIR/inventory"
mkdir -p "$QA_DIR" "$REPORT_DIR" "$INVENTORY_DIR"

# --- Step 1: audit-adapter ---
echo "Step 1: audit-adapter"
python3 -m gradience audit-adapter \
    --peft-dir "$ADAPTER_DIR" \
    --out "$QA_DIR/tiny_lora_qa.json"
echo "  -> OK"
echo ""

# --- Step 2: merge-audit ---
echo "Step 2: merge-audit"
python3 -m gradience merge-audit \
    --adapter-a "$ADAPTER_DIR" \
    --adapter-b "$ADAPTER_DIR" \
    --emit-report "$REPORT_DIR/self_merge_report.json"
echo "  -> OK"
echo ""

# --- Step 3: summarize-inventory ---
echo "Step 3: summarize-inventory"
python3 -m gradience summarize-inventory \
    --qa-dir "$QA_DIR" \
    --report-dir "$REPORT_DIR" \
    --emit-report "$INVENTORY_DIR/inventory_summary.json"
echo "  -> OK"
echo ""

# --- Step 4: Validate outputs ---
echo "Step 4: Validate JSON schemas"

validate_schema() {
    local file="$1"
    local expected_schema="$2"
    local actual
    actual=$(python3 -c "import json; print(json.load(open('$file'))['schema'])")
    if [ "$actual" != "$expected_schema" ]; then
        echo "  FAIL: $file — expected schema '$expected_schema', got '$actual'"
        exit 1
    fi
    echo "  $file -> $expected_schema OK"
}

validate_schema "$QA_DIR/tiny_lora_qa.json" "gradience.adapter_qa/v1"
validate_schema "$REPORT_DIR/self_merge_report.json" "gradience.merge_qa_report/v1"
validate_schema "$INVENTORY_DIR/inventory_summary.json" "gradience.inventory_summary/v1"
echo ""

echo "=== All preflight checks passed ==="
