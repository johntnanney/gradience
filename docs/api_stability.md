# API Stability Guide

This document provides detailed guidance on Gradience's API stability tiers and how to write code that won't break across releases.

## 🎯 Quick Summary

- **Use `gradience.api`** for programmatic access (stable)
- **Pin tags** for reproducible results: `pip install git+https://github.com/johntnanney/gradience.git@v0.4.4`
- **Avoid internal imports** like `gradience.bench.protocol`

## 📋 Stability Tiers

### ✅ Stable (Backward Compatible)

These interfaces are guaranteed stable across minor releases:

#### Command-Line Interface
```bash
# Audit a LoRA adapter
gradience audit --peft-dir /path/to/adapter --layers

# Monitor training telemetry  
gradience monitor run.jsonl --verbose

# Run bench protocol
python -m gradience.bench.run_bench --config config.yaml --output results/

# Aggregate multiple runs
python -m gradience.bench.aggregate run1/ run2/ run3/ --output aggregate/
```

#### Python API (gradience.api)
```python
import gradience.api as gradience

# Run audit programmatically
result = gradience.audit(peft_dir="adapter/", layers=True)

# Run bench protocol
artifacts = gradience.run_bench(config="config.yaml", output="results/")
bench_data = gradience.load_bench_report("results/")

# Aggregate multiple runs
agg_artifacts = gradience.aggregate_bench_runs(
    runs=["run1/", "run2/", "run3/"], 
    output="aggregate/"
)
```

#### Configuration Schema
```yaml
# Top-level structure is stable
model:
  name: "microsoft/DialoGPT-small"
  type: "causal_lm"

task:
  dataset: "gsm8k"
  subset: "main"

lora:
  probe_r: 16
  alpha: 16
  
train:
  batch_size: 4
  learning_rate: 1e-4
```

#### Artifact Schema (Core Keys)
```json
// bench.json - core structure guaranteed
{
  "bench_version": "0.1",
  "timestamp": "...",
  "env": { "python_version": "...", "torch_version": "..." },
  "model": "...",
  "task": "...",
  "probe": { "rank": 16, "accuracy": 0.85 },
  "compressed": {
    "uniform_median": {
      "rank": 8,
      "param_reduction": 0.5,
      "verdict": "PASS"
    }
  }
}
```

### 🟨 Provisional (May Evolve)

Use these, but expect occasional changes:

#### HuggingFace Integration
```python
from gradience.vnext.integrations.hf import GradienceCallback

# Stable usage pattern
trainer = Trainer(..., callbacks=[GradienceCallback()])

# Config may evolve
callback = GradienceCallback(config={
    "enable_memory_tracking": True,  # may change
    "log_frequency": 10              # may change
})
```

#### Task Profiles  
```python
from gradience.bench.task_profiles import get_task_profile

# Interface may evolve as we add more tasks
profile = get_task_profile("gsm8k_causal_lm")
```

### 🧪 Experimental (No Guarantees)

Opt-in features that may disappear:

```python
# Experimental features (use at your own risk)
from gradience.vnext.experimental.guards import ExperimentalGuard

# These may change or be removed without notice
```

### 🚫 Internal (Will Break)

Don't import these directly:

```python
# ❌ Don't do this - will break
from gradience.bench.protocol import create_canonical_bench_report
from gradience.spectral import compute_spectral_metrics

# ✅ Do this instead
import gradience.api as gradience
result = gradience.audit(peft_dir="adapter/")
```

## 🔧 Migration Patterns

### From Internal APIs to Stable APIs

```python
# ❌ Old (breaks)
from gradience.bench.protocol import run_bench_protocol
result = run_bench_protocol(config, output_dir)

# ✅ New (stable)
import gradience.api as gradience
artifacts = gradience.run_bench(config="config.yaml", output="output/")
```

### From Direct CLI to API Wrapper

```python
# ❌ Fragile subprocess calls
import subprocess
subprocess.run(["python", "-m", "gradience", "audit", "--peft-dir", "adapter/"])

# ✅ Stable API wrapper  
import gradience.api as gradience
gradience.audit(peft_dir="adapter/", layers=True)
```

## 📦 Reproducibility Best Practices

### Pin Tags for Publications

```bash
# ✅ Pin exact version for reproducible results
pip install git+https://github.com/johntnanney/gradience.git@v0.4.4

# ❌ Don't use floating versions for papers
pip install git+https://github.com/johntnanney/gradience.git@main
```

### Save Complete Environment

```python
import gradience.api as gradience

# Run with logging for full reproducibility
artifacts = gradience.run_bench(
    config="config.yaml",
    output="results/",
    log_path="results/bench.log"  # Captures all CLI output
)

# The bench.json includes complete environment metadata
bench_data = gradience.load_bench_report("results/")
print(f"Git commit: {bench_data['env']['git_commit']}")
print(f"PyTorch version: {bench_data['env']['torch_version']}")
```

### Configuration Management

```python
# ✅ Embed complete config in artifacts
artifacts = gradience.run_bench(config="config.yaml", output="results/")
bench_data = gradience.load_bench_report("results/")

# Config is fully embedded for reproducibility
embedded_config = bench_data["config_metadata"]["embedded_config"]
config_hash = bench_data["config_metadata"]["config_hash"]
```

## 🔍 Testing Your Code Against Updates

### Compatibility Testing

```python
import gradience.api as gradience
import tempfile
from pathlib import Path

def test_api_compatibility():
    """Test that your code works with gradience.api."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        
        # Test audit API
        result = gradience.audit(
            peft_dir="examples/adapters/tiny_lora/",
            layers=True,
            check=False  # Don't fail on errors
        )
        assert result.returncode == 0
        
        # Test artifact loading
        artifacts = gradience.run_bench(
            config="examples/configs/smoke_gsm8k.yaml",
            output=output_dir,
            smoke=True,
            check=False
        )
        
        if artifacts.bench_json.exists():
            data = gradience.load_bench_report(output_dir)
            assert "bench_version" in data
            assert "env" in data
```

### Forward Compatibility Patterns

```python
def robust_bench_loading(output_dir):
    """Load bench data defensively."""
    data = gradience.api.load_bench_report(output_dir)
    
    # ✅ Check for required fields
    required_fields = ["bench_version", "env", "probe"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # ✅ Use .get() for optional fields
    git_commit = data["env"].get("git_commit", "unknown")
    compression_data = data.get("compressed", {})
    
    # ✅ Handle schema evolution
    if "config_metadata" in data:
        # New format with embedded config
        config = data["config_metadata"]["embedded_config"]
    else:
        # Fallback for older format
        config = {"legacy": True}
    
    return {
        "git_commit": git_commit,
        "probe_accuracy": data["probe"]["accuracy"],
        "config": config
    }
```

## 📚 Example Usage Patterns

### Research Pipeline

```python
import gradience.api as gradience
from pathlib import Path

def run_research_pipeline(configs, base_output_dir):
    """Run a research pipeline with stable APIs."""
    results = []
    
    for i, config_path in enumerate(configs):
        output_dir = Path(base_output_dir) / f"run_{i}"
        
        # Run bench with full logging
        artifacts = gradience.run_bench(
            config=config_path,
            output=output_dir,
            log_path=output_dir / "gradience.log"
        )
        
        # Load and store results
        bench_data = gradience.load_bench_report(output_dir)
        results.append({
            "config": config_path,
            "output_dir": output_dir,
            "probe_accuracy": bench_data["probe"]["accuracy"],
            "best_compression": bench_data["summary"]["best_compression"],
            "git_commit": bench_data["env"]["git_commit"]
        })
    
    # Aggregate all runs
    if len(results) > 1:
        agg_artifacts = gradience.aggregate_bench_runs(
            runs=[r["output_dir"] for r in results],
            output=Path(base_output_dir) / "aggregate"
        )
        
        agg_data = gradience.load_bench_aggregate(agg_artifacts.output_dir)
        print(f"Statistical power: {agg_data['summary']['statistical_power']}")
    
    return results
```

### CI Integration

```python
def ci_smoke_test():
    """Smoke test for CI using stable APIs."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "smoke_test"
        
        try:
            # Quick smoke test
            artifacts = gradience.run_bench(
                config="examples/configs/smoke_gsm8k.yaml",
                output=output_dir,
                smoke=True,
                ci=True
            )
            
            # Validate artifacts exist
            assert artifacts.bench_json.exists()
            assert artifacts.bench_md.exists()
            
            # Quick validation
            data = gradience.load_bench_report(output_dir)
            assert data["bench_version"] == "0.1"
            assert "probe" in data
            
            print("✅ CI smoke test passed")
            return True
            
        except Exception as e:
            print(f"❌ CI smoke test failed: {e}")
            return False
```

---

**Remember**: The stable APIs are designed to be simple and robust. If you need advanced functionality not available through `gradience.api`, consider requesting it as a feature rather than importing internals.