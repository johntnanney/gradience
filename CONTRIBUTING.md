# Contributing to Gradience

🚀 **Welcome! Thanks for wanting to make Gradience better.**

This guide helps you get set up and contributing quickly. Whether you're fixing bugs, adding features, or improving documentation, we've streamlined the process to minimize friction.

## 📋 Quick Start Checklist

- [ ] Python 3.10+ installed
- [ ] Git configured with your name/email  
- [ ] Fork and clone the repository
- [ ] Run `make setup` for development environment
- [ ] Run `make test-smoke` to verify setup (~6 seconds)
- [ ] Make your changes
- [ ] Run `make check` before submitting PR

## 🛠️ Development Setup

### One-Command Setup

```bash
git clone https://github.com/your-username/gradience.git
cd gradience
make setup
```

This creates a virtual environment, installs all dependencies with the `[hf,dev]` extras, and sets you up for development.

### Manual Setup (if needed)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[hf,dev]"
```

### Cache Configuration (Recommended)

Prevent "Disk quota exceeded" chaos during development:

```bash
make setup-cache
```

This configures HuggingFace, PyTorch, and Gradience caches to use workspace storage instead of filling up system disks.

### Verify Installation

```bash
# Quick smoke test (~6 seconds)
make test-smoke

# Full verification
make check
```

## 🧪 Testing

### Test Commands

| Command | Purpose | Runtime |
|---------|---------|---------|
| `make test-smoke` | CPU-only smoke tests (no GPU required) | ~6s |
| `make test-quick` | Fast tests without coverage | ~30s |
| `make test` | Full test suite with coverage | ~2-5min |

### Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=gradience --cov-report=html

# Run specific test categories  
pytest tests/test_basic_functionality.py -v  # Basic imports/configs
pytest tests/test_udr_opt_in_policy.py -v   # UDR policy enforcement

# Local CI simulation (full validation)
python test_ci_locally.py
```

### GPU Testing

```bash
# Quick GPU smoke test (~3-5 minutes)
scripts/bench/run_gpu_smoke.sh

# Manual GPU smoke test
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/gpu_smoke/mistral_gsm8k_gpu_smoke.yaml \
  --output test_output \
  --smoke

# Validate output files exist
ls test_output/  # Should contain: bench.json, bench.md, runs.json
```

### CPU Testing (CI-Safe)

```bash
# CPU smoke test (safe for GitHub Actions)
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2_ci.yaml \
  --output test_output \
  --smoke --ci
```

### Before Submitting PRs

```bash
# Run all quality checks
make check

# This runs: lint, format-check, type-check, and tests
```

### Test Categories

1. **Smoke Tests**: Fast CPU-only tests that verify core logic
   ```bash
   python scripts/run_ci_smoke_tests.py --timing
   ```

2. **Unit Tests**: Component-specific tests with mocks/fixtures
   ```bash
   python -m pytest tests/test_*.py -v
   ```

3. **Integration Tests**: End-to-end bench runs (require time/compute)
   ```bash
   python -m pytest test_bench_*_integration.py -v
   ```

### Running Bench Locally

To test your changes with a real bench run:

```bash
# Small GSM8K smoke bench (~5 minutes)
python -m gradience.bench \
    --config configs/smoke_gsm8k.yaml \
    --output-dir bench_output/

# Check the bench.json artifact
cat bench_output/bench.json
```

## 🗂️ File and Artifact Hygiene

### ❌ DO NOT COMMIT

**Never commit these files/directories:**

```bash
# Cache directories (potentially gigabytes)
.cache/
__pycache__/
*.pyc
.pytest_cache/

# HuggingFace cache (multi-GB)
hf_cache/
hf_home/
hub/
datasets/
transformers_cache/  # deprecated location

# Model artifacts (hundreds of MB)
*.safetensors
*.bin
adapter_model.*
pytorch_model.*

# Training artifacts  
checkpoints/
ckpts_*/
snapshots_*/
*.log
nohup.out

# RunPod session artifacts
_pid.txt
_exit_code.txt  
STAGE.txt
*.swp
*.swo
.bash_history
.python_history

# Temporary outputs
test_output/
ci_test_output/
bench_output_*/
```

Our `.gitignore` catches most of these, but be vigilant about:
- Large model files (check `git status` before committing)
- Personal RunPod session artifacts
- Accidentally staged log files

### ✅ DO COMMIT

**Always include these in bug reports and PRs:**

```bash
# Configuration files
*.yaml
*.yml

# Scientific evidence (small JSON files)
bench.json           # Aggregated benchmark results
bench.md            # Human-readable report  
runs.json           # Individual run metadata
audit.json          # LoRA audit summary

# Documentation and scripts
docs/
scripts/
*.md
```

## 📝 Code Style & Standards

### Automatic Formatting

```bash
# Format all code
make format

# Check formatting without changes
make format-check
```

### Style Requirements

- **Linting**: Use `ruff` for fast Python linting
- **Type Hints**: Add type annotations for public functions
- **Import Sorting**: Automatic with `ruff`
- **Line Length**: 100 characters (configured in `pyproject.toml`)

### Example Function

```python
def audit_lora_adapter(
    peft_dir: str | Path, 
    *, 
    base_model_id: Optional[str] = None
) -> LoRAAuditResult:
    """
    Audit a PEFT LoRA adapter for rank compression opportunities.
    
    Args:
        peft_dir: Path to PEFT adapter directory
        base_model_id: Optional base model identifier for UDR computation
        
    Returns:
        Audit result with compression suggestions
        
    Raises:
        ValueError: If adapter config is invalid
    """
    # Implementation here...
```

### Code Quality Checks

```bash
# Linting
ruff check .

# Type checking  
mypy gradience/

# All checks together
make lint
```

## 🏗️ Architecture Overview

Understanding the codebase structure helps target your contributions:

```
gradience/
├── gradience/
│   ├── vnext/              # Current API (stable)
│   │   ├── telemetry/      # JSONL telemetry output
│   │   ├── integrations/   # HuggingFace callbacks
│   │   ├── audit/          # LoRA adapter analysis
│   │   └── rank_suggestion/ # Conservative compression
│   ├── bench/              # Benchmarking framework
│   │   ├── task_profiles/  # Task-specific logic (GSM8K, GLUE)
│   │   └── protocol.py     # Core bench protocol
│   ├── spectral.py         # Legacy spectral analysis
│   ├── structural.py       # Legacy structural analysis  
│   └── cli.py             # CLI interface
├── tests/                  # Comprehensive test suite
├── scripts/               # Development utilities
└── docs/                  # Documentation
```

### Key Components

- **Task Profiles** (`bench/task_profiles/`): Add new datasets/tasks here
- **Audit Logic** (`vnext/audit/`): Core LoRA analysis algorithms
- **Rank Suggestions** (`vnext/rank_suggestion/`): Conservative compression logic
- **Protocol** (`bench/protocol.py`): Benchmarking pipeline orchestration

## 📊 Bench Artifact Changes

When your changes affect bench output:

### 1. Generate Before/After Artifacts

```bash
# Checkout main branch
git checkout main
python -m gradience.bench --config configs/smoke_gsm8k.yaml --output-dir before/

# Checkout your branch  
git checkout your-feature-branch
python -m gradience.bench --config configs/smoke_gsm8k.yaml --output-dir after/

# Compare artifacts
diff before/bench.json after/bench.json
```

### 2. Document Changes in PR

```markdown
## Bench Artifact Changes

- **bench_version**: Updated from 0.1 to 0.2
- **New fields**: Added `config_metadata.primary_metric_key`
- **Breaking changes**: None (backward compatible)
- **Example diff**: 
  ```json
  + "config_metadata": {
  +   "primary_metric_key": "eval_exact_match"
  + }
  ```
```

## 🐛 Bug Reports

### Required Files

When reporting bugs, attach these files (they're small and contain scientific evidence):

1. **bench.json** - Aggregated results with metrics
2. **bench.md** - Human-readable benchmark report
3. **audit.json** - LoRA audit summary (if UDR enabled)
4. **Config file** - The exact YAML config used
5. **Last 200 log lines**:
   ```bash
   tail -n 200 nohup.log > last_200_lines.txt
   # or from tmux session
   tail -n 200 your_log_file.txt > last_200_lines.txt
   ```

### Environment Info

Include this diagnostic output:

```bash
# Python and package versions
python --version
pip list | grep -E "(torch|transformers|gradience|peft)"

# GPU info (if relevant)
nvidia-smi

# Disk space
df -h /workspace /root

# HF cache status  
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
ls -la $HF_HOME/hub/ | head -5  # Sample cache contents
```

### RunPod Issues

For RunPod-specific issues, see [`docs/runpod.md`](docs/runpod.md) first. Include:

- Pod type and region
- Disk layout (`df -h /workspace /root`)
- Whether you ran `source scripts/runpod/env.sh`
- Contents of `_exit_code.txt` (if using no-tmux runner)

## 🐛 Debugging

### Common Issues

**Import Errors**
```bash
# Ensure development install
pip install -e ".[hf,dev]"

# Check Python path
python -c "import gradience; print(gradience.__file__)"
```

**Test Failures**
```bash
# Run specific test with verbose output
python -m pytest tests/test_specific.py::test_function -v -s

# Debug with pdb
python -m pytest tests/test_specific.py::test_function --pdb
```

**Cache Issues**
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Or use configured cache
make setup-cache
```

### Preflight Validation

Run the built-in preflight check before reporting issues:

```bash
python -c "from gradience.bench.protocol import run_bench_preflight_check; run_bench_preflight_check()"
```

This validates PyTorch devices, disk space, HF cache health, and environment variables.

### Performance Debugging

```bash
# Profile test performance
python -m pytest tests/ --durations=10

# Memory profiling
python -m memory_profiler your_script.py
```

## 📚 Adding Features

### New Task Profile

1. **Create Profile Class** (`gradience/bench/task_profiles/my_task.py`)
   ```python
   from .base import TaskProfile
   
   class MyTaskProfile(TaskProfile):
       name = "my_task"
       primary_metric = "accuracy"
       
       def load(self, cfg): 
           # Load dataset
       
       def tokenize(self, ds, tokenizer, cfg):
           # Task-specific tokenization
       
       def probe_gate(self, eval_results, cfg):
           # Quality gate logic
   ```

2. **Register Profile** (`gradience/bench/task_profiles/registry.py`)
   ```python
   TASK_PROFILES["my_task"] = MyTaskProfile
   ```

3. **Add Tests** (`tests/test_my_task_profile.py`)
   ```python
   def test_my_task_tokenization():
       # Test tokenization logic
   
   def test_my_task_probe_gate():
       # Test quality gate
   ```

### New Audit Metric

1. **Implement Metric** (`gradience/vnext/audit/metrics.py`)
   ```python
   def compute_my_metric(lora_A: torch.Tensor, lora_B: torch.Tensor) -> float:
       """Compute custom LoRA analysis metric."""
       # Implementation
   ```

2. **Add to Audit Result** (`gradience/vnext/audit/lora_audit.py`)
   ```python
   # Include in LayerStats
   my_metric=compute_my_metric(lora_A, lora_B)
   ```

3. **Update Tests** (`tests/test_audit_*.py`)
   ```python
   def test_my_metric_computation():
       # Verify metric calculation
   ```

## 🔄 Pull Request Process

### Before Submitting

1. **Rebase on main**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run quality checks**
   ```bash
   make check
   ```

3. **Test bench artifacts** (if relevant)
   ```bash
   make test-smoke
   python scripts/run_ci_smoke_tests.py --timing
   ```

### PR Description Template

Use our PR template (auto-populated) or include:

- **What changed**: High-level summary
- **Why**: Problem solved or feature motivation  
- **How**: Implementation approach
- **Tests**: What testing was done
- **Breaking changes**: Any API/artifact changes
- **Bench artifacts**: Before/after if applicable

### Review Process

1. **Automated checks** run on all PRs
2. **Manual review** by maintainers
3. **Bench validation** for protocol changes
4. **Merge** once approved and checks pass

## ❓ Getting Help

### Community

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs, request features  
- **Discord/Slack**: Real-time help (link in README)

### Code Questions

- **Architecture**: Check `docs/architecture.md`
- **API**: See `gradience/__init__.py` for public API
- **Examples**: Browse `examples/` directory

### Performance Questions

- **Benchmarking**: See `docs/benchmarking.md`
- **Storage**: See `docs/storage_and_caching.md`
- **CI**: See `docs/cpu_smoke_tests.md`

## 🏆 Recognition

Contributors are recognized in:

- **CHANGELOG.md**: Feature/fix credits
- **README.md**: Core contributor acknowledgments  
- **Release notes**: Major contribution highlights

---

**Thanks for contributing to Gradience! Your improvements help the entire ML community build better, more efficient models.** 🚀