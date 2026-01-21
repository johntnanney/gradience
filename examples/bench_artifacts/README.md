# Bench Artifacts Examples

This directory contains example output from Gradience bench runs, showing the complete artifact structure and format.

## 📁 Files

- **bench.json**: Single-seed bench result with comprehensive metadata
- **bench_aggregate.json**: Multi-seed aggregated result with statistics
- **bench_report.md**: Human-readable markdown report

## 🎯 Key Features Demonstrated

### Self-Describing Metadata
All bench artifacts include comprehensive metadata for full reproducibility:
- **Git tracking**: commit hash and tag information
- **Environment**: Python, PyTorch, CUDA versions and GPU details
- **Model versioning**: HuggingFace model revision hashes
- **Dataset versioning**: Dataset revision and split information
- **Configuration**: Complete config embedding with hash

### Compression Analysis
Example shows the complete compression validation pipeline:
- **Probe quality gate**: Validates initial adapter quality
- **Multiple variants**: Tests different compression strategies
- **Statistical analysis**: Provides confidence measures
- **Conservative recommendations**: Safe compression suggestions

### Schema Examples

#### Environment Metadata
```json
{
  "env": {
    "python_version": "3.10.12",
    "torch_version": "2.1.0+cu118",
    "cuda_available": true,
    "gpu_devices": [{
      "name": "NVIDIA GeForce RTX 4090",
      "total_memory": 25757220864
    }],
    "git_commit": "a1b2c3d4...",
    "git_tag": "v0.4.4",
    "model_info": {
      "model_id": "microsoft/DialoGPT-small",
      "revision": "f6d5bc5e..."
    }
  }
}
```

#### Compression Results
```json
{
  "compressed": {
    "uniform_median": {
      "rank": 8,
      "params": 221184,
      "accuracy": 0.2089,
      "delta_vs_probe": -0.0067,
      "param_reduction": 0.5,
      "verdict": "PASS"
    }
  }
}
```

#### Multi-Seed Statistics
```json
{
  "probe": {
    "accuracy": {
      "mean": 0.2089,
      "std": 0.0123,
      "values": [0.2156, 0.2034, 0.2078]
    }
  }
}
```

## 🔍 Usage Examples

### Inspect Schema Structure
```bash
# View complete structure
cat bench.json | jq .

# Extract specific fields
cat bench.json | jq '.config_metadata.primary_metric_key'
cat bench.json | jq '.env.git_commit'
cat bench.json | jq '.compressed.uniform_median.param_reduction'
```

### Validate Your Artifacts
```bash
# Compare your bench.json against examples
diff <(cat your_bench.json | jq 'keys | sort') \
     <(cat examples/bench_artifacts/bench.json | jq 'keys | sort')

# Check required metadata fields
python -c "
import json
with open('your_bench.json') as f:
    data = json.load(f)

required_fields = ['bench_version', 'env', 'config_metadata', 'compressed']
for field in required_fields:
    assert field in data, f'Missing required field: {field}'
print('✅ All required fields present')
"
```

### Extract Metrics for Analysis
```bash
# Extract compression metrics across runs
jq -r '.compressed | to_entries[] | "\(.key): \(.value.param_reduction)"' bench.json

# Get environment summary
jq -r '.env | "Python: \(.python_version), PyTorch: \(.torch_version), GPU: \(.gpu_devices[0].name)"' bench.json
```

## 📊 Understanding the Results

### Probe Quality Assessment
- **eval_exact_match ≥ 0.15**: Minimum threshold for GSM8K
- **utilization_mean**: How efficiently LoRA rank is used (0.72 = 72%)
- **energy_rank_90_p50**: Median of 90% energy preservation ranks

### Compression Validation
- **uniform_median (r=8)**: 50% parameter reduction
- **uniform_p90 (r=12)**: Conservative 25% reduction
- **verdict: PASS**: Compression maintains acceptable quality

### Statistical Confidence
- **n_seeds: 3**: Sufficient for defensible claims
- **statistical_power: sufficient**: Results are statistically valid
- **std values**: Measure consistency across seeds

---

These examples demonstrate the complete Gradience bench artifact format, providing templates for understanding and validating your own bench results.