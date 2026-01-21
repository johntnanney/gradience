# Mechanism Testing Experiment

This experiment tests whether audit-guided per-layer rank patterns provide real benefit beyond simple compression by comparing with shuffled controls.

## Hypothesis

- **H0**: Audit-guided placement provides no benefit beyond heterogeneity (per_layer ≈ per_layer_shuffled)
- **H1**: Audit-guided placement provides real benefit (per_layer > per_layer_shuffled)

## Experimental Design

### Fixed Invariants
- **Model**: Mistral-7B-v0.1 (consistent architecture)
- **Task**: GSM8K mathematical reasoning (challenging for compression)  
- **Probe rank**: r=64 (sufficient capacity for audit)
- **Training steps**: 1200 (enough for convergence)
- **Seeds**: 42, 43, 44 (statistical power)

### Variants Tested
1. **probe**: Baseline with r=64 uniform ranks
2. **per_layer**: Audit-guided heterogeneous ranks 
3. **per_layer_shuffled**: Same rank multiset, shuffled assignment (control)

### Key Scientific Control

The `per_layer_shuffled` variant uses the **exact same rank multiset** as `per_layer` but redistributes ranks randomly across modules. This isolates the mechanism:

- If audit-guided placement matters → per_layer > per_layer_shuffled
- If any heterogeneity is enough → per_layer ≈ per_layer_shuffled

## Running the Experiment

### Prerequisites
```bash
# Install dependencies
pip install -e .[hf,bench]

# Login to HuggingFace (for Mistral access)
huggingface-cli login

# Ensure CUDA available (recommended)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Run Experiment
```bash
# Run complete 3-seed experiment (~6-8 hours on single GPU)
./experiments/run_mechanism_test.sh

# Results will be saved to experiments/mechanism_test_results/
```

### Analyze Results
```bash
# Statistical analysis with hypothesis testing
python experiments/analyze_mechanism_test.py experiments/mechanism_test_results/aggregated_results.json
```

## Success Criteria

### Compression Efficacy
- **per_layer** must outperform **probe** by >2% accuracy
- Statistical significance: p < 0.05
- Effect size: Cohen's d > 0.5

### Mechanism Benefit  
- **per_layer** must outperform **per_layer_shuffled** by >1% accuracy
- Statistical significance: p < 0.05
- Practical significance threshold: 1%

## Expected Outcomes

### Scenario 1: Real Mechanism 
```
probe:              0.650 ± 0.010
per_layer:          0.675 ± 0.008  (+2.5% vs probe ✓)
per_layer_shuffled: 0.660 ± 0.012  (+1.5% vs shuffled ✓)
```
**Conclusion**: Audit-guided placement provides real adaptive regularization benefit

### Scenario 2: Heterogeneity Only
```
probe:              0.650 ± 0.010
per_layer:          0.672 ± 0.008  (+2.2% vs probe ✓) 
per_layer_shuffled: 0.670 ± 0.012  (+0.2% vs shuffled ❌)
```
**Conclusion**: Any rank heterogeneity helps, audit guidance doesn't matter

### Scenario 3: No Benefit
```
probe:              0.650 ± 0.010
per_layer:          0.655 ± 0.012  (+0.5% vs probe ❌)
per_layer_shuffled: 0.653 ± 0.015  (+0.2% vs shuffled ❌)
```
**Conclusion**: Per-layer compression ineffective for this task/model

## Files

- `mechanism_test_config.yaml`: Experimental configuration
- `run_mechanism_test.sh`: Multi-seed experiment runner
- `analyze_mechanism_test.py`: Statistical analysis with hypothesis testing
- `README.md`: This documentation

## Implementation Details

The shuffled control is implemented in `gradience/bench/protocol.py`:

```python
def _create_shuffled_rank_pattern(original_pattern, seed):
    """Create shuffled control preserving rank multiset."""
    # Extract ranks and shuffle with deterministic seed
    ranks = list(original_pattern.values())
    random.Random(seed + 10000).shuffle(ranks)
    
    # Recombine with same module names
    return dict(zip(original_pattern.keys(), ranks))
```

This ensures the control has:
- ✅ Same total parameters as per_layer
- ✅ Same rank distribution 
- ✅ Same heterogeneity level
- ❌ Different audit-guided placement

The difference isolates the **placement mechanism** from other factors.