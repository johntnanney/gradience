#!/usr/bin/env python3
"""
Local CI test runner - simulates what GitHub Actions will run.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Return code: {result.returncode}")
            if result.stdout:
                print(f"   STDOUT: {result.stdout}")
            if result.stderr:
                print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run local CI tests."""
    print("🧪 Running local CI simulation...")
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Test results
    results = []
    
    # 1. Test imports and basic functionality
    results.append(run_command([
        sys.executable, "-c", 
        "import gradience; import gradience.bench.protocol; print('✅ Core imports work')"
    ], "Core package imports"))
    
    # 2. Test config validation
    results.append(run_command([
        sys.executable, "-c",
        """
import yaml
from pathlib import Path
configs = list(Path('gradience/bench/configs').rglob('*.yaml'))
print(f'✅ Found {len(configs)} config files')
for c in configs[:3]:
    with open(c) as f:
        yaml.safe_load(f)
print('✅ Config parsing works')
"""
    ], "Config validation"))
    
    # 3. Test pytest (if tests exist)
    if Path("tests").exists() and list(Path("tests").glob("test_*.py")):
        results.append(run_command([
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
        ], "Unit tests"))
    else:
        print("ℹ️  No tests directory found, skipping pytest")
        results.append(True)  # Don't fail CI for missing tests
    
    # 4. Test linting (if ruff available)
    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=True)
        results.append(run_command([
            "ruff", "check", "gradience/", "--output-format=text"
        ], "Code linting"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ℹ️  Ruff not available, skipping linting")
        results.append(True)
    
    # 5. Test bench smoke (optional - might take too long)
    print("\n🔄 Testing bench smoke run...")
    print("   This may take several minutes...")
    
    smoke_success = run_command([
        sys.executable, "-m", "gradience.bench.run_bench",
        "--config", "gradience/bench/configs/distilbert_sst2_ci.yaml",
        "--output", "ci_test_output",
        "--smoke",
        "--ci"
    ], "CPU bench smoke test")
    
    if smoke_success:
        # Validate output files
        output_dir = Path("ci_test_output")
        required_files = ["bench.json", "bench.md", "runs.json"]
        
        missing_files = []
        for file in required_files:
            if not (output_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing output files: {missing_files}")
            results.append(False)
        else:
            print("✅ All required output files generated")
            results.append(True)
    else:
        results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 CI Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All local CI tests passed! Ready for GitHub Actions.")
        return 0
    else:
        print("❌ Some tests failed. Fix issues before pushing.")
        return 1

if __name__ == "__main__":
    exit(main())