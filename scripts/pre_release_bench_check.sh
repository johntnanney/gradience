#!/bin/bash
# Pre-release bench validation check
# Ensures bench-related components are ready for release

set -e

echo "🔍 Pre-Release Bench Validation Check"
echo "====================================="
echo ""

# Track overall status
OVERALL_STATUS=0

# Function to run check with status tracking
run_check() {
    local check_name="$1"
    local check_command="$2"
    local required="$3"  # "required" or "optional"
    
    echo "📋 $check_name"
    echo "   Command: $check_command"
    
    if eval "$check_command" >/dev/null 2>&1; then
        echo "   ✅ PASS"
    else
        echo "   ❌ FAIL"
        if [ "$required" = "required" ]; then
            OVERALL_STATUS=1
            echo "   ⚠️  This is a REQUIRED check - release should be blocked"
        else
            echo "   ℹ️  This is optional - release can proceed with caution"
        fi
    fi
    echo ""
}

# Always required: Policy regression tests
run_check "Policy Regression Tests" \
    "python -m pytest tests/test_bench/test_policy_regression.py -q" \
    "required"

# Check if policy files are consistent
run_check "Policy File Consistency" \
    "python -c 'import yaml; yaml.safe_load(open(\"gradience/bench/policies/safe_uniform.yaml\"))'" \
    "required"

# Check if any rank suggestion code changed (heuristic)
if git diff --name-only HEAD~1 2>/dev/null | grep -E "(protocol|audit|bench)" >/dev/null 2>&1; then
    echo "🧪 RANK SUGGESTION CODE CHANGES DETECTED"
    echo "  Modified files affecting rank suggestions detected."
    echo "  Consider running mini-validation if algorithm changes were made:"
    echo ""
    echo "    python -m gradience.bench.run_bench \\"
    echo "      --config gradience/bench/configs/distilbert_sst2_mini_validation.yaml \\"
    echo "      --output bench_runs/pre_release_validation \\"
    echo "      --device cpu"
    echo ""
    echo "  📖 See RELEASE_CHECKLIST.md for detailed guidance"
    echo ""
fi

# Optional: Check if documentation is consistent
run_check "Documentation Consistency (VALIDATION_POLICY.md mentions r=20)" \
    "grep -q 'Uniform r=20' VALIDATION_POLICY.md" \
    "optional"

run_check "Documentation Consistency (Bench README mentions r=20)" \
    "grep -q '\\*\\*r=20\\*\\*' gradience/bench/README.md" \
    "optional"

echo "📊 SUMMARY"
echo "========="

if [ $OVERALL_STATUS -eq 0 ]; then
    echo "✅ All required bench validation checks passed!"
    echo "   Release can proceed from bench perspective."
    echo ""
    echo "🔗 Next steps:"
    echo "   1. Complete other release checklist items (RELEASE_CHECKLIST.md)"
    echo "   2. Run full test suite"
    echo "   3. Update version numbers and release notes"
else
    echo "❌ Required bench validation checks failed!"
    echo "   Release should be blocked until issues are resolved."
    echo ""
    echo "🔧 Common fixes:"
    echo "   1. Run ./scripts/test_policy_regression.sh to see detailed failures"
    echo "   2. Check if policy files were accidentally modified"
    echo "   3. Review RELEASE_CHECKLIST.md for policy change procedures"
    exit 1
fi