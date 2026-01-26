"""
Gradience Bench (v0.1)

Bench is a small harness for validating Gradience recommendations by running:
probe (high rank) -> audit -> compress -> retrain -> eval -> report

This package is intentionally minimal in v0.1 (one model, one task).
"""

import os
from pathlib import Path

__all__ = ['get_scripts_path', 'list_available_scripts']

def get_scripts_path():
    """Get the path to bench runner scripts (available in source installs)."""
    try:
        # Try to find scripts directory relative to package
        package_dir = Path(__file__).parent.parent.parent
        scripts_dir = package_dir / "scripts"
        if scripts_dir.exists():
            return str(scripts_dir)
        else:
            return None
    except Exception:
        return None

def list_available_scripts():
    """List available bench runner scripts."""
    scripts_path = get_scripts_path()
    if not scripts_path:
        print("⚠️  Scripts not found. Available in source installations only.")
        print("   Install from source: pip install git+https://github.com/your-repo/gradience.git")
        return
    
    scripts_path = Path(scripts_path)
    
    print(f"📁 Scripts location: {scripts_path}")
    print("\n🚀 Available bench runners:")
    
    # GPU smoke test
    gpu_smoke = scripts_path / "bench" / "run_gpu_smoke.sh"
    if gpu_smoke.exists():
        print(f"   {gpu_smoke}")
        print("     GPU smoke test - fast validation of full GPU pipeline")
    
    # Nohup runner
    nohup_runner = scripts_path / "bench" / "run_seed_nohup.sh"  
    if nohup_runner.exists():
        print(f"   {nohup_runner}")
        print("     No-tmux friendly runner with state tracking")
    
    # RunPod environment
    runpod_env = scripts_path / "runpod" / "env.sh"
    if runpod_env.exists():
        print(f"   {runpod_env}")
        print("     RunPod environment setup (HF cache configuration)")
    
    print("\n📖 Usage examples:")
    print("   scripts/bench/run_gpu_smoke.sh --config configs/gpu_smoke/mistral_gsm8k_gpu_smoke.yaml --output runs/test")
    print("   scripts/bench/run_seed_nohup.sh --config your_config.yaml --output runs/experiment --background")
    print("   source scripts/runpod/env.sh  # On RunPod")