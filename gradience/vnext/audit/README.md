# vNext LoRA Audit

This module computes **LoRA delta (ΔW = B@A)** spectral/efficiency metrics without
forming dense ΔW.

It is designed to support the vNext "Efficiency Auditor" and telemetry metrics events:
`event="metrics", kind="lora_audit"`.

Main entrypoint:

```python
from gradience.vnext.audit import audit_lora_peft_dir

result = audit_lora_peft_dir("./peft_out", include_top_singular_values=4)
print(result.to_summary_dict(include_layers=False))
```

## Update Dominance Ratio (UDR) and Spectral Drift Index (SDI)

UDR/SDI provide **optional** instrumentation to quantify how much LoRA updates dominate their base model weights.

### What it measures

- **UDR** = `||ΔW||₂ / ||W_base||₂` (spectral dominance ratio)
- **SDI** = `log₁₀(UDR + ε)` (logarithmic scale for interpretation)

Where `ΔW = (α/r) × B × A` is the LoRA update with proper PEFT scaling.

### What it's good for

- **"Are two runs comparable?"** — Compare UDR distributions across experiments
- **"Is the adapter doing a small perturbation or a big rewrite?"** — Low vs high UDR
- **"Which modules dominate the update budget?"** — Per-layer UDR breakdown

### What it's not

- **Not a performance predictor** — UDR doesn't directly predict accuracy/loss
- **Not a certification gate** — Don't use UDR thresholds to pass/fail runs

### How to run

**Basic audit (no UDR):**
```bash
gradience audit --peft-dir /path/to/adapter
```

**Audit with UDR (requires base model):**
```bash
gradience audit --peft-dir /path/to/adapter --base-model mistralai/Mistral-7B-v0.1
```

**With custom cache location:**
```bash
gradience audit --peft-dir /path/to/adapter \
  --base-model gpt2 \
  --base-norms-cache /workspace/gradience_cache/base_norms
```

**Disable UDR computation:**
```bash
gradience audit --peft-dir /path/to/adapter --no-udr
```

### Interpretation cheat sheet

- **UDR < 0.1**: Adapter is a small perturbation to base weights
- **UDR ~ 1.0**: Adapter update magnitude similar to base weight magnitude  
- **UDR > 3.0**: Adapter dominates the local weight update scale
- **Compare UDR distributions across modules**, not just summary statistics
- **SDI** provides log-scale view: SDI = 0 means UDR = 1, SDI = 1 means UDR = 10

### Output schema

**Summary metrics** (in audit JSON):
```json
{
  "udr_mean": 0.245,
  "udr_median": 0.180, 
  "udr_p90": 0.512,
  "udr_max": 0.893,
  "sdi_mean": -0.621,
  "fraction_udr_gt_0_1": 0.667,
  "fraction_udr_gt_0_3": 0.250,
  "n_layers_with_udr": 12
}
```

**Per-layer metrics** (in `layers` array):
```json
{
  "name": "model.layers.0.self_attn.q_proj",
  "udr": 0.142,
  "sdi": -0.847,
  "delta_sigma_max": 2.84,
  "base_sigma_max": 20.01,
  "scale": 2.0
}
```

### Requirements

- Base model norms computation requires `transformers` library
- First run with `--base-model` creates a cache for subsequent runs
- UDR computation adds ~10-30% overhead to audit time
- Works with any PEFT adapter (LoRA, QLoRA, etc.)

Notes:
- `.safetensors` requires the `safetensors` package.
- Computation uses r×r eigendecompositions in float64 by default (cheap/stable).
