# Changelog

All notable changes to Gradience are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-01-26

This is an operationally significant release focused on production reliability and infrastructure robustness for cloud deployments, particularly RunPod environments. The primary goal is to eliminate "it worked on my pod" failures and reduce support load through better defaults and diagnostics.

### 🔄 **BREAKING CHANGES**

#### UDR Explicit Opt-In Policy
- **BREAKING**: UDR (Utilization and Decomposition Ratio) computation now requires explicit opt-in
- **Before**: UDR enabled by default, causing resource hangs on memory-constrained pods
- **After**: Must set `audit.compute_udr: true` AND `audit.base_model: "model-name"` to enable UDR
- **Migration**: Add explicit audit configuration to existing configs that rely on UDR
- **Rationale**: Prevents expensive base model loading on resource-constrained environments

### ✨ **New Features**

#### Preflight Validation System
- Added comprehensive preflight checks before expensive training operations
- Validates PyTorch device availability, disk space, HuggingFace cache health
- Detects safetensors corruption with specific remediation commands
- Provides actionable error messages instead of cryptic failures
- Usage: `python -c "from gradience.bench.protocol import run_bench_preflight_check; run_bench_preflight_check()"`

#### GPU Smoke Test Suite
- New official GPU smoke test configuration: `gradience/bench/configs/gpu_smoke/mistral_gsm8k_gpu_smoke.yaml`
- Fast GPU pipeline validation (~3-5 minutes vs hours for full runs)
- Dedicated runner script: `scripts/bench/run_gpu_smoke.sh`
- Validates full pipeline: model loading → training → audit → compression → evaluation
- 20 training steps, 32 train samples, 64 eval samples for rapid iteration

#### Artifact Hygiene Defaults
- New runtime options to prevent disk space exhaustion:
  - `runtime.keep_adapter_weights: false` - Removes heavy adapter files (hundreds of MB)
  - `runtime.keep_checkpoints: false` - Removes intermediate checkpoints
- Preserves scientific evidence (JSON reports, metrics) while cleaning artifacts
- Prevents "volume full → cache corrupt → safetensors header error" loops
- Enabled by default in CI and smoke test configs

#### "No tmux" Friendly Runner
- New wrapper script: `scripts/bench/run_seed_nohup.sh`
- Writes clear state tracking files: `_pid.txt`, `STAGE.txt`, `_exit_code.txt`, `nohup.log`
- Prevents "where was I?" archaeology when kicked off pods
- Supports foreground and background execution modes
- Comprehensive error handling and stage reporting

### 📚 **Documentation & Infrastructure**

#### RunPod Production Guide
- Comprehensive RunPod survival documentation: `docs/runpod.md`
- Covers dual-disk layout (`/root/` vs `/workspace/`), persistence strategies
- Environment variable standardization using current HuggingFace standards
- Troubleshooting guide for common RunPod failure modes
- Cache management and corruption recovery procedures

#### Standardized HuggingFace Cache
- Environment setup script: `scripts/runpod/env.sh`
- Uses current HF standards: `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`
- Deprecates old `TRANSFORMERS_CACHE` patterns
- Configures optimal cache locations for RunPod dual-disk layout
- Prevents cache corruption and disk quota issues

#### Contributing Guidelines
- New `CONTRIBUTING.md` with development hygiene rules
- Artifact hygiene guidelines (what to commit vs avoid)
- Bug report requirements with specific file attachments
- Test running instructions for GPU and CPU environments
- PR template and review checklist

#### GitHub Actions CI
- CPU-only invariants testing workflow: `.github/workflows/ci.yml`
- Tests Python 3.10, 3.11, 3.12 compatibility
- Runs pytest, ruff linting, mypy type checking
- CPU bench smoke test with config validation
- Prevents "it worked on my pod" regressions without requiring GPU infrastructure

### 🛠️ **Improvements**

#### Enhanced Configuration Validation
- All YAML configs now validated in CI pipeline
- Better error messages for configuration issues
- Explicit device configuration requirements (`runtime.device: "cpu"` or `"cuda"`)
- Validation for UDR opt-in requirements

#### Packaging & Distribution
- Updated `MANIFEST.in` to include all new configs, scripts, and documentation
- Ensures pip installs contain RunPod scripts and GPU smoke tests
- Enhanced `.gitignore` with RunPod-specific patterns
- Prevents accidental commits of cache directories and session artifacts

#### Test Infrastructure
- New test categories: basic functionality, UDR policy enforcement, config validation
- Local CI simulation script: `test_ci_locally.py`
- Comprehensive test coverage for new opt-in policy
- Smoke test validation for output artifact generation

### 🔧 **Technical Details**

#### Version Management
- Canonical version in `pyproject.toml` (single source of truth)
- `importlib.metadata.version("gradience")` provides runtime version access
- Fallback version in `gradience/__init__.py` for development installs
- Consistent version bumping discipline

#### Error Handling Improvements
- Explicit error messages for UDR misconfiguration
- Preflight validation with remediation instructions
- Better diagnostics for HuggingFace cache issues
- Clear safetensors corruption detection and recovery

### 📋 **Migration Guide**

For existing users upgrading from 0.5.x:

1. **UDR Configuration**: If your configs rely on UDR computation, add explicit opt-in:
   ```yaml
   audit:
     compute_udr: true
     base_model: "your-model-name"  # Required when compute_udr: true
   ```

2. **RunPod Users**: Run the environment setup on first use:
   ```bash
   source scripts/runpod/env.sh
   ```

3. **CI/Testing**: Use new smoke tests for faster validation:
   ```bash
   # GPU environments
   scripts/bench/run_gpu_smoke.sh
   
   # CPU environments  
   python test_ci_locally.py
   ```

### 🙏 **Acknowledgments**

This release addresses real-world operational pain points identified through production deployments. Special thanks to RunPod users who provided detailed failure reports and infrastructure insights.

---

## [0.5.0] - Previous Release

Previous stable release. See git history for changes prior to this changelog introduction.