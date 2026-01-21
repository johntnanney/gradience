# UDR/SDI Example: Comparing Fine-Tuning Approaches

This example shows how to use Update Dominance Ratio (UDR) and Spectral Drift Index (SDI) to compare different fine-tuning approaches on the same task.

## Scenario: Comparing Conservative vs Aggressive LoRA

Let's say you've fine-tuned the same base model with two different LoRA configurations:

1. **Conservative**: r=4, α=8 (α/r = 2.0)
2. **Aggressive**: r=16, α=64 (α/r = 4.0)

## Step 1: Run Audit with UDR

```bash
# Conservative run
gradience audit \
  --peft-dir ./runs/conservative/adapter \
  --base-model microsoft/DialoGPT-medium \
  --json > conservative_audit.json

# Aggressive run  
gradience audit \
  --peft-dir ./runs/aggressive/adapter \
  --base-model microsoft/DialoGPT-medium \
  --json > aggressive_audit.json
```

## Step 2: Compare UDR Distributions

```python
import json

# Load results
with open('conservative_audit.json') as f:
    conservative = json.load(f)

with open('aggressive_audit.json') as f:
    aggressive = json.load(f)

# Compare summary metrics
print("=== UDR Comparison ===")
print(f"Conservative UDR median: {conservative['udr_median']:.3f}")
print(f"Aggressive UDR median:   {aggressive['udr_median']:.3f}")

print(f"Conservative layers > 0.3 UDR: {conservative['fraction_udr_gt_0_3']:.1%}")
print(f"Aggressive layers > 0.3 UDR:   {aggressive['fraction_udr_gt_0_3']:.1%}")
```

## Step 3: Interpretation

**Example output:**
```
=== UDR Comparison ===
Conservative UDR median: 0.145
Aggressive UDR median:   0.487

Conservative layers > 0.3 UDR: 15%
Aggressive layers > 0.3 UDR:   73%
```

**What this tells us:**

- **Conservative (UDR ~0.15)**: Updates are small perturbations (~15% of base weight magnitude)
- **Aggressive (UDR ~0.49)**: Updates approach half the magnitude of base weights
- **73% of aggressive layers** have UDR > 0.3, suggesting significant departures from base model

## Step 4: Module-Level Analysis

```python
# Find modules with highest UDR differences
conservative_layers = {l['name']: l.get('udr', 0) for l in conservative['layers']}
aggressive_layers = {l['name']: l.get('udr', 0) for l in aggressive['layers']}

# Compare per module
print("\n=== Biggest UDR Differences ===")
for name in conservative_layers:
    if name in aggressive_layers:
        diff = aggressive_layers[name] - conservative_layers[name]
        print(f"{name}: +{diff:.3f}")
```

**Example output:**
```
=== Biggest UDR Differences ===
model.layers.5.self_attn.q_proj: +0.421
model.layers.8.self_attn.v_proj: +0.386
model.layers.12.mlp.gate_proj: +0.335
```

This reveals which modules are most affected by the different LoRA configurations.

## Step 5: Correlation with Performance

```python
# If you have performance metrics
conservative_accuracy = 0.847
aggressive_accuracy = 0.823

print(f"\n=== Performance vs UDR ===")
print(f"Conservative: {conservative_accuracy:.1%} accuracy, UDR median {conservative['udr_median']:.3f}")
print(f"Aggressive:   {aggressive_accuracy:.1%} accuracy, UDR median {aggressive['udr_median']:.3f}")

# Higher UDR doesn't guarantee better performance!
# Use UDR to understand HOW the adaptation works, not WHETHER it works
```

## Practical Insights

**When UDR is useful:**

1. **Debugging convergence**: High UDR in some modules might indicate they're working too hard
2. **Comparing hyperparameters**: UDR reveals the spectral impact of α/r choices
3. **Architecture decisions**: Compare UDR across different target_modules
4. **Stability analysis**: Very high UDR (>10) might indicate instability

**When UDR is not useful:**

1. **Performance prediction**: UDR doesn't directly predict accuracy/loss
2. **Absolute thresholds**: No universal "good" vs "bad" UDR values
3. **Single-run analysis**: UDR is most valuable for comparisons

## Integration with Gradience Bench

If you're using Gradience Bench, UDR metrics are automatically included in comparative reports:

```bash
# Bench automatically computes UDR when base_model is specified
python -m gradience.bench \
  --config experiments.yaml \
  --base-model microsoft/DialoGPT-medium
```

The resulting reports will include UDR distributions in the analysis sidebar, helping you understand not just *which* approach performs better, but *how* it achieves that performance through spectral characteristics.