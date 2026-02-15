# Changelog

All notable changes to Gradience are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

*No unreleased changes yet.*

## [0.9.1] - 2026-02-07

### Added
- **Two-Tier Selection System**: Defensible recommendations based on multi-seed validation evidence
  - Tier A (validated safe): 3/3 seeds pass tolerance threshold
  - Tier B (conditionally promising): 2/3 seeds pass with detailed explanations
  - Evidence-based transparency in markdown reports with honest validation statements
- **Candidate Diversity Improvements**: Enhanced deduplication logic prevents rank collisions
  - "Second rung" mapping for policy collisions without additional compute cost
  - More aggressive rank selection when conflicts occur
  - Preserved diversity across compression candidates in FAST mode
- **Scientific Honesty in Validation Evidence**: Per-rank validation evidence section
  - Explicit per-rank validation statements matching evidence pack
  - Clear indicators for validated safe, conditionally promising, unreliable, and failed variants
  - Complete transparency about which compression levels can be trusted across independent random seeds

### Changed
- **Multi-seed aggregation logic**: Improved statistical validation with honest verdict reporting
- **Markdown report generation**: Enhanced with per-rank evidence section before two-tier recommendations
- **Fast mode candidate selection**: Better diversity preservation through collision resolution

### Fixed
- **Rank collision handling**: Prevents duplicate ranks in candidate generation
- **Evidence transparency**: Eliminates over-optimistic claims in benchmark reports

## [0.9.0] - 2026-02-06

### Added
- **PyPI-Ready Documentation Suite**: Complete professional documentation for package distribution
  - Comprehensive troubleshooting guide with symptom→fix patterns
  - RunPod cloud GPU setup guide (optional, cloud-specific)  
  - Artifacts & evidence documentation for research reproducibility
  - CLI reference with exit codes and fast mode explanations
  - Configuration reference with guaranteed package examples
  - Installation guide with environment-specific instructions
- **Install Tier Tag System**: All documentation commands tagged with (base), (bench), or (gpu) requirements
- **Idiot-proof Installation Guidance**: 
  - Explicit PyTorch installation guidance (CPU vs CUDA)
  - ModuleNotFoundError callouts in README
  - Clear bench extras story with troubleshooting

### Changed
- **README Restructure**: Transformed from "lab notebook" to PyPI-first landing page
  - Professional user onboarding flow: Who it's for → What you get → Install → Quickstart
  - Performance guarantees documented (CLI operations, dependency loading)
  - Clear 3-tier installation system (base, bench extras, development)
  - Organized documentation section with logical progression
- **PyPI Compatibility**: All relative links converted to absolute GitHub URLs
- **Quickstart Workflow**: Uses importlib.resources for guaranteed config availability after pip install
- **Citation Block**: Version-aware citations matching current release (v0.9.0, 2026)

### Fixed
- **Package Distribution**: Comprehensive CI testing matrix for "pip install ready" validation
  - Tests across multiple Python versions (3.9-3.12) and platforms
  - Validates base install, bench extras, packaging correctness, and CLI performance
  - Local testing script for pre-release validation
  - Console script verification and help command timing
- **PyPI Landing Page**: Eliminates broken relative links and ensures quickstart works from installed wheel

### Documentation  
- **Professional Footer**: Added license, citation (APA + BibTeX), changelog links
- **Security & Responsible Use**: Brief guidance on validation, resource awareness, data privacy
- **Release Management**: Links to GitHub releases for detailed release notes
- **Release Checklist**: Systematic version management process to prevent citation inconsistencies

## [0.8.6] - 2026-02-03

### Added
- Comprehensive monitoring and reliability features for production use
- Professional release notes for v0.8.5

## [0.8.5] - 2026-01-31

### Fixed
- Per-layer rank validation false failures with degrade-to-uniform fallback logic
- Version bump for clean release

## [0.8.4-patch1] - 2026-01-30

### Fixed
- Empty `rank_pattern` crash and `rank_pattern` KeyError in compression pipeline
- `run_bench --help` made fast by deferring heavy imports from module load time

## [0.8.4] - 2026-01-30

### Fixed
- `rank_pattern` KeyError with explicit schema contract for compression configs
- Probe quality gate "undertrained" check corrected

### Added
- Blessed Evidence Pack runner as canonical RunPod entrypoint
- Evidence Pack configs canonicalized in dedicated directory
- Truth commands as acceptance gate scripts

### Changed
- Eliminated hardcoded paths in dev scripts and tests for CI portability
- Restricted pytest collection to `tests/` directory only

## [0.8.3] - 2026-01-28

### Fixed
- `policy_global_suggestions` robustness in `generate_compression_configs`

## [0.8.2] - 2026-01-28

### Changed
- `protocol.py` refactored to use true module alias instead of star-import

## [0.8.1] - 2026-01-28

### Fixed
- UDR norms computation via `state_dict` for GPU RunPod compatibility
- Module name canonicalization for adapter layer matching
- Schema stability improvements for audit output

## [0.8.0] - 2026-01-28

### Added
- Comprehensive JSON bloat prevention and UX enhancements for benchmark output
- Cleaner, more compact audit and bench JSON artifacts

## [v0.7.1] - 2026-01-26

### Added - LoRA Gain Audit Complete Implementation

**🎯 Lean Playbook Implementation Complete**

Comprehensive LoRA gain/magnitude audit functionality following the 4-step lean playbook for maximum adoption:

#### Step 1: Blessed Demo Command
- `make demo-gain-audit` - Single command demo (~30 seconds)
- Shows mean magnitudes, top layers, and concentration analysis
- Runs fastest CPU config (DistilBERT SST2 mini smoke)

#### Step 2: Sensitivity Validation  
- `make sensitivity-check` - Proves metrics respond to known changes (~60 seconds)
- Mathematical validation: >100% magnitude changes for rank variations
- Confirms metrics compute real, responsive values vs static/cached data

#### Step 3: Human Interpretability
- New `## Magnitude diagnostics (LoRA ΔW)` section in bench.md
- Answers key questions in <30 seconds:
  - "Which layers did most adapting?" → Top 5 layers by energy
  - "Is adaptation concentrated?" → HHI index with plain English interpretation

#### Step 4: Power User Utilities
- `scripts/inspect_audit.py` - Standalone inspector with glob pattern support
- No jq dependency, handles any audit.json nesting structure

### Technical Features
- **Math utilities**: Efficient LoRA ΔW norm computation without materializing ΔW matrices
- **Composition analysis**: Energy concentration using Herfindahl-Hirschman Index (HHI)
- **Complete integration**: audit.json → bench.json → bench.md pipeline
- **Config flag**: `audit.enable_composition_analysis` (default: true)

Sample output:
```
📈 Gain Audit Results
Update Magnitude: ||ΔW||_F: 0.017, ||ΔW||_2: 0.012
Top Layers: Layer 4 (18.9%), Layer 3 (17.0%), Layer 2 (16.9%)
Concentration: HHI 0.168 → ✅ Well distributed adaptation
```

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