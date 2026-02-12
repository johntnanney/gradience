#!/usr/bin/env python3
"""
Local test script for pip install readiness.

This script tests the same scenarios as the CI pipeline but can be run locally
for quick validation before pushing.

Usage:
    python scripts/test_pip_install_ready.py [--test {base,bench,packaging,all}]
"""

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import os


def run_cmd(cmd, description, cwd=None, timeout=30):
    """Run command and return success status."""
    print(f"\n🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   📝 {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ Failed (exit code {result.returncode})")
            if result.stderr.strip():
                print(f"   📝 Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout ({timeout}s)")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_base_install():
    """Test A: Base install functionality."""
    print("\n" + "="*50)
    print("🧪 TEST A: Base Install")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_venv"
        
        # Create virtual environment
        if not run_cmd(f"python3 -m venv {venv_path}", "Creating virtual environment"):
            return False
            
        # Activate and install
        activate_cmd = f"source {venv_path}/bin/activate"
        if not run_cmd(f"{activate_cmd} && pip install .", "Installing base package"):
            return False
            
        # Test import
        if not run_cmd(f"{activate_cmd} && python -c \"import gradience; print(f'Version: {{gradience.__version__}}')\"", "Testing import"):
            return False
            
        # Test CLI
        if not run_cmd(f"{activate_cmd} && gradience --help", "Testing CLI help", timeout=10):
            return False
            
        # Verify minimal dependencies
        result = subprocess.run(
            f"{activate_cmd} && pip list", 
            shell=True, capture_output=True, text=True
        )
        
        if "torch" in result.stdout or "transformers" in result.stdout:
            print("   ❌ Heavy dependencies found in base install")
            return False
        else:
            print("   ✅ No heavy dependencies in base install")
            
        print("\n✅ Base install test PASSED")
        return True


def test_bench_extras():
    """Test B: Bench extras functionality.""" 
    print("\n" + "="*50)
    print("🧪 TEST B: Bench Extras")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "bench_venv"
        
        # Create virtual environment
        if not run_cmd(f"python3 -m venv {venv_path}", "Creating virtual environment"):
            return False
            
        # Install with CPU torch first
        activate_cmd = f"source {venv_path}/bin/activate"
        if not run_cmd(f"{activate_cmd} && pip install torch --index-url https://download.pytorch.org/whl/cpu", "Installing CPU PyTorch"):
            return False
            
        # Install bench extras
        if not run_cmd(f"{activate_cmd} && pip install '.[bench]'", "Installing bench extras"):
            return False
            
        # Test bench CLI
        if not run_cmd(f"{activate_cmd} && gradience-bench --help", "Testing bench CLI", timeout=10):
            return False
            
        # Test config access
        config_test = '''
import gradience.bench.configs as configs
import os
config_dir = os.path.dirname(configs.__file__)
yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
print(f"Found {len(yaml_files)} config files")
assert len(yaml_files) > 40, f"Expected >40 configs, got {len(yaml_files)}"
print("✅ Config access test passed")
'''
        if not run_cmd(f"{activate_cmd} && python -c '{config_test}'", "Testing config access"):
            return False
            
        print("\n✅ Bench extras test PASSED")
        return True


def test_packaging_correctness():
    """Test C: Packaging correctness."""
    print("\n" + "="*50) 
    print("🧪 TEST C: Packaging Correctness")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir) / "packaging_test"
        work_dir.mkdir()
        
        # Copy source to temp directory
        src_dir = Path.cwd()
        shutil.copytree(src_dir, work_dir / "gradience_src", 
                       ignore=shutil.ignore_patterns('*.git*', 'dist', 'build', '*.egg-info', '__pycache__', '.venv*'))
        
        os.chdir(work_dir / "gradience_src")
        
        # Build packages
        if not run_cmd("python3 -m pip install build", "Installing build tools"):
            return False
            
        if not run_cmd("python3 -m build", "Building wheel and sdist", timeout=60):
            return False
            
        # Create test environment
        venv_path = work_dir / "wheel_test_venv"
        if not run_cmd(f"python3 -m venv {venv_path}", "Creating test environment"):
            return False
            
        # Install from wheel
        activate_cmd = f"source {venv_path}/bin/activate"
        if not run_cmd(f"{activate_cmd} && pip install dist/*.whl", "Installing from wheel"):
            return False
            
        # Test importlib.resources access
        resources_test = '''
import importlib.resources
import gradience.bench.configs
import gradience.bench.policies

# Test main configs
configs_files = list(importlib.resources.files(gradience.bench.configs).iterdir())
yaml_files = [f for f in configs_files if f.name.endswith(".yaml")]
print(f"Main configs: {len(yaml_files)} files")
assert len(yaml_files) > 40, f"Expected >40 main configs, got {len(yaml_files)}"

# Test evidence subdirectory
evidence_files = list(importlib.resources.files(gradience.bench.configs).joinpath("evidence").iterdir())
evidence_yamls = [f for f in evidence_files if f.name.endswith(".yaml")]
print(f"Evidence configs: {len(evidence_yamls)} files")
assert len(evidence_yamls) >= 3, f"Expected >=3 evidence configs, got {len(evidence_yamls)}"

# Test policies
policy_files = list(importlib.resources.files(gradience.bench.policies).iterdir())
policy_yamls = [f for f in policy_files if f.name.endswith(".yaml")]
print(f"Policy files: {len(policy_yamls)} files")
assert len(policy_yamls) >= 1, f"Expected >=1 policy, got {len(policy_yamls)}"

print("✅ All importlib.resources tests passed")
'''
        
        if not run_cmd(f"{activate_cmd} && python -c '{resources_test}'", "Testing importlib.resources access"):
            return False
            
        # Test fresh install from different directory
        os.chdir("/tmp")
        fresh_test = '''
import gradience
print(f"Fresh install version: {gradience.__version__}")

import gradience.bench.configs
import os
config_dir = os.path.dirname(gradience.bench.configs.__file__)
yaml_count = len([f for f in os.listdir(config_dir) if f.endswith(".yaml")])
print(f"Fresh install configs: {yaml_count}")
assert yaml_count > 40, f"Fresh install missing configs: {yaml_count}"
print("✅ Fresh installation test passed")
'''
        
        if not run_cmd(f"{activate_cmd} && python -c '{fresh_test}'", "Testing fresh install"):
            return False
            
        print("\n✅ Packaging correctness test PASSED")
        return True


def test_performance():
    """Test performance requirements."""
    print("\n" + "="*50)
    print("🧪 PERFORMANCE: CLI Speed Requirements") 
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "perf_venv"
        
        # Create and install
        if not run_cmd(f"python3 -m venv {venv_path}", "Creating environment"):
            return False
            
        activate_cmd = f"source {venv_path}/bin/activate"
        if not run_cmd(f"{activate_cmd} && pip install .", "Installing package"):
            return False
            
        # Test base CLI speed (< 3s for base)
        if not run_cmd(f"timeout 3s bash -c '{activate_cmd} && gradience --help'", "Testing gradience --help speed"):
            return False
            
        # Install bench and test speed
        if not run_cmd(f"{activate_cmd} && pip install torch --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch"):
            return False
            
        if not run_cmd(f"{activate_cmd} && pip install '.[bench]'", "Installing bench extras"):
            return False
            
        # First run can be slower (< 15s) as it loads ML deps
        if not run_cmd(f"timeout 15s bash -c '{activate_cmd} && gradience-bench --help'", "Testing gradience-bench --help first run", timeout=20):
            return False
            
        # Second run should be faster (< 5s) 
        if not run_cmd(f"timeout 5s bash -c '{activate_cmd} && gradience-bench --help'", "Testing gradience-bench --help second run"):
            return False
            
        print("\n✅ Performance requirements PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test pip install readiness locally")
    parser.add_argument('--test', choices=['base', 'bench', 'packaging', 'performance', 'all'], 
                       default='all', help='Which test to run')
    args = parser.parse_args()
    
    original_cwd = Path.cwd()
    
    try:
        print("🚀 Testing Pip Install Readiness")
        print(f"   Current directory: {original_cwd}")
        
        results = {}
        
        if args.test in ['base', 'all']:
            results['base'] = test_base_install()
            
        if args.test in ['bench', 'all']:
            results['bench'] = test_bench_extras()
            
        if args.test in ['packaging', 'all']:
            results['packaging'] = test_packaging_correctness()
            
        if args.test in ['performance', 'all']:
            results['performance'] = test_performance()
            
        # Summary
        print("\n" + "="*50)
        print("📊 FINAL RESULTS")
        print("="*50)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {test_name.upper()}: {status}")
            if not passed:
                all_passed = False
                
        if all_passed:
            print("\n🎉 ALL TESTS PASSED - Package is pip install ready!")
            return 0
        else:
            print("\n💥 SOME TESTS FAILED - Package needs fixes")
            return 1
            
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    sys.exit(main())