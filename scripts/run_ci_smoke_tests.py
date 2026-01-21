#!/usr/bin/env python3
"""
CI Smoke Test Runner

Runs the essential smoke tests that verify core pipeline logic works
without requiring GPUs. Designed to run in under 5 minutes.

Usage:
    python scripts/run_ci_smoke_tests.py
    python scripts/run_ci_smoke_tests.py --verbose
    python scripts/run_ci_smoke_tests.py --timing
"""

import subprocess
import sys
import time
from pathlib import Path


def run_pytest(test_paths, args=None):
    """Run pytest on specified paths with optional arguments."""
    if args is None:
        args = []
    
    cmd = [sys.executable, "-m", "pytest"] + test_paths + args
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    return result, end_time - start_time


def main():
    """Run CI smoke tests."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    timing = "--timing" in sys.argv
    
    # Define core smoke test suites
    smoke_tests = [
        # Comprehensive CPU pipeline verification
        "tests/test_cpu_smoke_comprehensive.py",
        
        # Config parsing and routing
        "tests/test_bench_config_parsing.py",
        
        # GSM8K task profile (formatting, masking)
        "tests/test_gsm8k_profile.py",
        
        # Audit with minimal artifacts
        "tests/test_audit_json_invariants.py",
        
        # Aggregator backward compatibility  
        "tests/test_bench_aggregator_compatibility.py",
        
        # Basic CLI functionality
        "tests/test_cli_smoke.py",
        
        # Core validation protocol
        "tests/test_validation_protocol.py",
    ]
    
    # Filter to only existing test files
    existing_tests = []
    for test_path in smoke_tests:
        if Path(test_path).exists():
            existing_tests.append(test_path)
        elif verbose:
            print(f"Skipping {test_path} (not found)")
    
    if not existing_tests:
        print("❌ No smoke test files found!")
        return 1
    
    # Prepare pytest arguments
    pytest_args = []
    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.extend(["-q", "--tb=short"])
    
    # Add performance optimizations for CI
    pytest_args.extend([
        "--disable-warnings",  # Suppress deprecation warnings for speed
        "--tb=no",              # No traceback for passed tests
        "-x",                   # Stop on first failure for fast feedback
    ])
    
    print(f"🚀 Running {len(existing_tests)} smoke test suites...")
    print(f"📁 Tests: {', '.join(Path(t).name for t in existing_tests)}")
    
    # Run all tests
    result, duration = run_pytest(existing_tests, pytest_args)
    
    # Show results
    if result.returncode == 0:
        print("✅ All smoke tests passed!")
        status = "PASS"
    else:
        print("❌ Some smoke tests failed!")
        status = "FAIL"
        
        # Show output on failure
        if result.stdout:
            print("\n📤 Test Output:")
            print(result.stdout)
        if result.stderr:
            print("\n📤 Test Errors:")
            print(result.stderr)
    
    # Performance summary
    if timing or verbose or duration > 60:
        print(f"\n⏱️  Duration: {duration:.2f}s")
        if duration > 300:  # 5 minutes
            print("⚠️  WARNING: Tests took longer than 5 minutes!")
        elif duration < 60:
            print("✨ Fast enough for CI!")
    
    # Performance recommendations
    if duration > 180:  # 3 minutes
        print("\n💡 Performance Tips:")
        print("   - Use pytest-xdist for parallel execution: pip install pytest-xdist")
        print("   - Run with: pytest -n auto tests/")
        print("   - Consider splitting into smaller test suites")
    
    return result.returncode


if __name__ == "__main__":
    exit(main())