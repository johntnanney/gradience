"""Preflight checks and artifact cleanup for bench protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_artifact_hygiene_cleanup(output_dir: Path, config: dict[str, Any]) -> None:
    """
    Clean up heavy adapter weights and checkpoints while preserving scientific artifacts.

    Deletes:
    - adapter_model.safetensors / adapter_model.bin (hundreds of MB)
    - checkpoint-* directories (if keep_checkpoints=false)

    Preserves:
    - bench.json, bench.md (scientific results)
    - */audit.json, */eval.json (evidence)
    - compression_configs.json (configuration record)
    - run.jsonl (telemetry, optional but useful)
    - adapter_config.json (small config files)
    """
    runtime_config = config.get("runtime", {})
    keep_adapter_weights = runtime_config.get("keep_adapter_weights", True)  # Default to keep for compatibility
    keep_checkpoints = runtime_config.get("keep_checkpoints", True)  # Default to keep for compatibility

    if keep_adapter_weights and keep_checkpoints:
        # Nothing to clean up
        return

    cleaned_files = []
    saved_space = 0

    try:
        for variant_dir in output_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            # Clean adapter weights
            if not keep_adapter_weights:
                # Look for adapter weights in common locations
                adapter_patterns = ["adapter_model.safetensors", "adapter_model.bin", "pytorch_adapter.bin"]

                for pattern in adapter_patterns:
                    # Check in variant root
                    adapter_file = variant_dir / pattern
                    if adapter_file.exists():
                        file_size = adapter_file.stat().st_size
                        adapter_file.unlink()
                        cleaned_files.append(str(adapter_file.relative_to(output_dir)))
                        saved_space += file_size

                    # Check in peft/ subdirectory
                    peft_adapter = variant_dir / "peft" / pattern
                    if peft_adapter.exists():
                        file_size = peft_adapter.stat().st_size
                        peft_adapter.unlink()
                        cleaned_files.append(str(peft_adapter.relative_to(output_dir)))
                        saved_space += file_size

                    # Check in checkpoint directories
                    for checkpoint_dir in variant_dir.glob("checkpoint-*"):
                        if checkpoint_dir.is_dir():
                            checkpoint_adapter = checkpoint_dir / pattern
                            if checkpoint_adapter.exists():
                                file_size = checkpoint_adapter.stat().st_size
                                checkpoint_adapter.unlink()
                                cleaned_files.append(str(checkpoint_adapter.relative_to(output_dir)))
                                saved_space += file_size

            # Clean checkpoint directories
            if not keep_checkpoints:
                for checkpoint_dir in variant_dir.glob("checkpoint-*"):
                    if checkpoint_dir.is_dir():
                        # Calculate directory size before deletion
                        dir_size = sum(f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file())

                        # Remove the entire checkpoint directory
                        import shutil

                        shutil.rmtree(checkpoint_dir)
                        cleaned_files.append(str(checkpoint_dir.relative_to(output_dir)) + "/")
                        saved_space += dir_size

    except Exception as e:
        print(f"\u26a0\ufe0f  Artifact cleanup encountered error: {e}")
        return

    # Report cleanup results
    if cleaned_files:
        saved_mb = saved_space / (1024 * 1024)
        print(f"\U0001f9f9 Artifact hygiene: Cleaned {len(cleaned_files)} items, saved {saved_mb:.1f} MB")
        if len(cleaned_files) <= 10:
            # Show details if not too many files
            for item in cleaned_files:
                print(f"   - {item}")
        else:
            # Summarize if many files
            weight_files = [f for f in cleaned_files if not f.endswith("/")]
            checkpoint_dirs = [f for f in cleaned_files if f.endswith("/")]
            if weight_files:
                print(f"   - {len(weight_files)} adapter weight files")
            if checkpoint_dirs:
                print(f"   - {len(checkpoint_dirs)} checkpoint directories")


def run_bench_preflight_check(config: dict[str, Any], model_name: str) -> None:
    """
    Preflight checks to catch common failure modes before expensive training.

    Checks for:
    - PyTorch device availability and GPU info
    - Disk space on critical paths (/tmp, /workspace if exists)
    - HF cache accessibility and potential corruption
    - Model loading sanity check for safetensors metadata issues

    Fails fast with helpful diagnostics to save hours of debugging.
    """
    import os
    import shutil  # noqa: F401
    import subprocess
    import tempfile
    from pathlib import Path

    print("\U0001f50d Running preflight checks...")

    # 1. PyTorch device check
    try:
        import torch

        print(f"\u2705 PyTorch {torch.__version__} available")

        # Check CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"\u2705 CUDA available: {gpu_count} GPU(s)")
            print(f"   Current device: {current_device} ({gpu_name})")
        else:
            print("\u26a0\ufe0f  CUDA not available - will run on CPU (much slower)")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("\u2705 MPS (Apple Silicon) available")

        # Determine runtime device from config
        runtime_device = config.get("runtime", {}).get("device", "auto")
        if runtime_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Config specifies device: cuda but CUDA is not available. "
                "Either install CUDA PyTorch or change device to 'cpu' in config."
            )

    except ImportError as e:
        raise RuntimeError(f"PyTorch not available: {e}") from e

    # 2. Disk space checks
    critical_paths = ["/tmp"]
    if os.path.exists("/workspace"):
        critical_paths.append("/workspace")

    for path in critical_paths:
        if os.path.exists(path):
            try:
                # Use df command for reliable disk space info
                result = subprocess.run(["df", "-h", path], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) >= 2:
                        # Parse df output: Filesystem Size Used Avail Use% Mounted
                        fields = lines[1].split()
                        if len(fields) >= 4:
                            avail = fields[3]
                            use_pct = fields[4]
                            print(f"\u2705 Disk space {path}: {avail} available ({use_pct} used)")

                            # Warn if very low space
                            if use_pct.rstrip("%").isdigit() and int(use_pct.rstrip("%")) > 95:
                                print(f"\u26a0\ufe0f  WARNING: {path} is {use_pct} full - may cause download failures")
                        else:
                            print(f"\u2705 {path} accessible (could not parse disk usage)")
                    else:
                        print(f"\u2705 {path} accessible (unusual df output)")
                else:
                    print(f"\u26a0\ufe0f  Could not check disk space for {path}: {result.stderr}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                print(f"\u26a0\ufe0f  Could not check disk space for {path}: {e}")
        else:
            print(f"\u2139\ufe0f  {path} does not exist (skipping)")

    # 3. HF cache checks
    try:
        import os
        from pathlib import Path

        from transformers import AutoTokenizer  # noqa: F401

        # Get HF cache directory (try multiple methods for different HF versions)
        try:
            # Try new HuggingFace Hub API
            from huggingface_hub import HF_HOME

            cache_dir = Path(HF_HOME) if HF_HOME else None
        except ImportError:
            cache_dir = None

        if not cache_dir:
            # Fallback to environment variable or default
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = Path(hf_home)
            else:
                # Default HF cache location
                cache_dir = Path.home() / ".cache" / "huggingface"

        try:
            print(f"\u2705 HuggingFace cache: {cache_dir}")

            if cache_dir.exists():
                # Check if writable
                test_file = cache_dir / f".preflight_test_{os.getpid()}"
                try:
                    test_file.touch()
                    test_file.unlink()
                    print("\u2705 HF cache directory writable")
                except (OSError, PermissionError) as e:
                    print(f"\u274c HF cache directory not writable: {e}")
                    raise RuntimeError(f"HuggingFace cache directory not writable: {cache_dir}") from e
            else:
                print(f"\u2139\ufe0f  HF cache directory will be created: {cache_dir}")

        except Exception as e:
            print(f"\u26a0\ufe0f  Could not determine HF cache location: {e}")

    except ImportError:
        print("\u26a0\ufe0f  HuggingFace transformers not available for cache check")

    # 3.5. HF Cache Environment Validation (RunPod survival check)
    hf_env_vars = {
        "HF_HOME": "Primary HF cache directory",
        "HF_HUB_CACHE": "Model weights cache",
        "HF_DATASETS_CACHE": "Dataset cache",
        "TORCH_HOME": "PyTorch cache",
    }

    print("\U0001f50d Validating HuggingFace cache environment...")
    env_issues = []
    runpod_detected = os.path.exists("/workspace")

    for env_var, _description in hf_env_vars.items():
        value = os.environ.get(env_var)
        if value:
            cache_path = Path(value)

            # Check if path is under /root/ on RunPod (danger zone)
            if runpod_detected and str(cache_path).startswith("/root/"):
                env_issues.append(
                    f"{env_var}={value} (\u26a0\ufe0f  points to /root/ - will fill system disk on RunPod)"
                )
            elif cache_path.exists() and not os.access(cache_path, os.W_OK):
                env_issues.append(f"{env_var}={value} (\u274c not writable)")
            else:
                print(f"\u2705 {env_var}: {value}")
        else:
            if runpod_detected:
                env_issues.append(f"{env_var} not set (\u26a0\ufe0f  will default to /root/.cache on RunPod)")
            else:
                print(f"\u2139\ufe0f  {env_var}: not set (will use defaults)")

    # Report environment issues with remediation
    if env_issues:
        print("\n\u26a0\ufe0f  HuggingFace cache environment issues detected:")
        for issue in env_issues:
            print(f"   - {issue}")

        print("\n\U0001f4a1 SOLUTION:")
        print("   Set environment variables to use a persistent cache directory:")
        if runpod_detected:
            print("   # For RunPod environments:")
            print("   export HF_HOME=/workspace/hf_cache/hf_home")
            print("   export HF_HUB_CACHE=/workspace/hf_cache/hub")
            print("   export HF_DATASETS_CACHE=/workspace/hf_cache/datasets")
            print("   export TORCH_HOME=/workspace/hf_cache/torch")
        else:
            print("   # For other environments, choose a persistent location:")
            print("   export HF_HOME=/path/to/persistent/hf_cache/hf_home")
            print("   export HF_HUB_CACHE=/path/to/persistent/hf_cache/hub")
            print("   export HF_DATASETS_CACHE=/path/to/persistent/hf_cache/datasets")
            print("   export TORCH_HOME=/path/to/persistent/hf_cache/torch")
    else:
        print("\u2705 HuggingFace cache environment properly configured")

    # 4. Model loading sanity check
    try:
        print(f"\U0001f50d Testing model loading: {model_name}")

        # Try to load just the config (fast) to catch common issues
        from transformers import AutoConfig

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Set a temporary cache dir to isolate the test
                os.environ["HF_HOME"] = temp_dir
                try:
                    config_test = AutoConfig.from_pretrained(model_name)
                    print(f"\u2705 Model config loads successfully: {config_test.model_type}")
                finally:
                    # Restore original cache
                    if "HF_HOME" in os.environ:
                        del os.environ["HF_HOME"]

        except Exception as e:
            error_msg = str(e).lower()
            if "incomplete" in error_msg and "metadata" in error_msg:
                print(f"\u274c Safetensors metadata corruption detected for {model_name}")
                print("\U0001f4a1 SOLUTION: Delete the corrupted cache:")
                print(f"   rm -rf ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
                print("   Or nuke entire cache: rm -rf ~/.cache/huggingface/")
                raise RuntimeError(f"HuggingFace cache corruption: {e}") from e
            else:
                print(f"\u26a0\ufe0f  Model loading issue: {e}")
                # Don't fail on other model loading issues as they might resolve during training

    except ImportError:
        print("\u26a0\ufe0f  Cannot test model loading - HuggingFace transformers not available")
    except Exception as e:
        print(f"\u26a0\ufe0f  Model loading check failed: {e}")

    print("\u2705 Preflight checks complete!\n")
