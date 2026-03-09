#!/usr/bin/env python3
"""
Validate RunPod experiment configurations locally before uploading.

Quick smoke test to ensure configurations are well-formed and will work on GPU.
"""

import sys
from pathlib import Path

import yaml


def validate_config(config_path: Path) -> bool:
    """Validate a single configuration file."""
    print(f"🔍 Validating {config_path.name}...")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["model", "task", "train", "lora", "compression", "runtime"]
        missing_sections = [s for s in required_sections if s not in config]
        if missing_sections:
            print(f"   ❌ Missing sections: {missing_sections}")
            return False

        # Check model configuration
        model = config["model"]
        supported_models = ["distilbert-base-uncased", "mistralai/Mistral-7B-v0.1"]
        if model["name"] not in supported_models:
            print(f"   ❌ Unsupported model: {model['name']}")
            print(f"       Supported: {supported_models}")
            return False

        if model["torch_dtype"] != "bf16":
            print(f"   ⚠️  Expected bf16 for GPU, got {model['torch_dtype']}")

        # Check device setting
        runtime = config["runtime"]
        if runtime["device"] != "cuda":
            print(f"   ❌ Expected cuda device, got {runtime['device']}")
            return False

        # Check heartbeat/progress tracking
        if not runtime.get("enable_heartbeat", False):
            print("   ❌ Heartbeat monitoring not enabled")
            return False

        if not runtime.get("enable_progress_tracking", False):
            print("   ❌ Progress tracking not enabled")
            return False

        # Check compression settings
        compression = config["compression"]
        if not compression.get("candidate_policies", []):
            print("   ❌ No candidate policies specified")
            return False

        if not compression.get("fast_mode", False):
            print("   ❌ Fast mode not enabled")
            return False

        # Check seeds
        seeds = compression.get("seeds", [])
        train_seed = config["train"].get("seed", None)

        if seeds and train_seed:
            print("   ❌ Both compression.seeds and train.seed specified - ambiguous")
            return False

        if seeds:
            seed_type = "multi-seed"
            seed_count = len(seeds)
        elif train_seed:
            seed_type = "single-seed"
            seed_count = 1
        else:
            print("   ❌ No seed configuration found")
            return False

        # Estimate runtime
        max_steps = config["train"]["max_steps"]
        if seed_type == "multi-seed":
            estimated_minutes = seed_count * (max_steps / 20)  # Rough estimate
        else:
            estimated_minutes = max_steps / 20

        print(f"   ✅ Valid {seed_type} configuration")
        print(f"   📊 {seed_count} seed(s), {max_steps} steps per seed")
        print(f"   ⏱️  Estimated runtime: ~{estimated_minutes:.1f} minutes on GPU")
        print(f"   💰 Estimated cost: ~${estimated_minutes * 0.015:.3f} on RunPod")

        return True

    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False


def main():
    """Validate all RunPod configurations."""
    print("RunPod Configuration Validator")
    print("=" * 40)

    configs = [
        Path("runpod_distilbert_sst2.yaml"),
        Path("runpod_distilbert_sst2_dev.yaml"),
        Path("runpod_mistral_gsm8k_public.yaml"),
        Path("runpod_mistral_gsm8k_dev.yaml"),
    ]

    all_valid = True
    for config_path in configs:
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            all_valid = False
            continue

        valid = validate_config(config_path)
        all_valid = all_valid and valid
        print()

    if all_valid:
        print("🎉 All configurations are valid!")
        print()
        print("📋 Next steps for RunPod:")
        print("1. Upload gradience codebase to RunPod workspace")
        print("2. Upload configuration files to workspace")
        print("3. Run: python -m gradience.bench.run_bench --config <config> --output <output>")
        print()
        print("💡 Recommended workflow:")
        print("• Start with runpod_distilbert_sst2_dev.yaml for rapid iteration")
        print("• Once validated, run runpod_distilbert_sst2.yaml for full multi-seed results")
        print("• For public demos: runpod_mistral_gsm8k_dev.yaml → runpod_mistral_gsm8k_public.yaml")
        print("• Analyze results: python scripts/analyze_public_artifacts.py <output_dir>")

        return 0
    else:
        print("❌ Some configurations have errors - please fix before running")
        return 1


if __name__ == "__main__":
    sys.exit(main())
