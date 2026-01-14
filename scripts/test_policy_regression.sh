#!/bin/bash
# Policy regression test runner
# Protects empirically calibrated safe uniform baseline decisions

set -e

echo "🛡️  Running Safe Uniform Policy Regression Tests"
echo "================================================"
echo "These tests protect empirically calibrated safety decisions:"
echo "  ✓ r=20 remains primary safe uniform baseline"
echo "  ✓ r=16 remains marked unsafe (0% pass rate)"
echo "  ✓ Safety criteria thresholds remain intact"
echo "  ✓ Empirical evidence stays documented"
echo ""

# Run the regression tests
python -m pytest tests/test_bench/test_policy_regression.py -v

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All policy regression tests passed!"
    echo "   Calibrated safety decisions are protected from drift."
else
    echo ""
    echo "❌ Policy regression tests failed!"
    echo "   ⚠️  This indicates potential accidental changes to calibrated safety policy."
    echo "   🔍 Review changes to:"
    echo "      - gradience/bench/policies/safe_uniform.yaml"
    echo "      - VALIDATION_POLICY.md"
    echo "      - gradience/bench/README.md"
    echo ""
    echo "   📋 If changes are intentional (e.g., new validation data):"
    echo "      1. Update empirical evidence in policy YAML"
    echo "      2. Update expected values in regression tests"
    echo "      3. Document rationale for policy change"
    exit 1
fi