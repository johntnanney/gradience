# Spectral Audit API

::: gradience.vnext.audit

The audit module performs SVD-based spectral analysis on PEFT LoRA adapters.

## Quick usage

```python
from gradience.vnext.audit import audit_lora_peft_dir

result = audit_lora_peft_dir("./my-adapter")

print(f"Layers analyzed: {len(result.layers)}")
print(f"Mean stable rank: {result.summary['stable_rank_mean']:.2f}")
print(f"Mean utilization: {result.summary['utilization_mean']:.2f}")

for layer in result.layers:
    print(f"  {layer.name}: stable_rank={layer.stable_rank:.2f}, "
          f"energy_rank_90={layer.energy_rank_90:.1f}")
```

## Functions

### `audit_lora_peft_dir()`

Main entry point. Loads a PEFT adapter directory and audits all LoRA layers.

```python
def audit_lora_peft_dir(
    peft_dir: str | Path,
    *,
    device: str = "cpu",
    compute_udr: bool = True,
    base_norms: dict[str, float] | None = None,
) -> LoRAAuditResult
```

| Parameter | Description |
|-----------|-------------|
| `peft_dir` | Path to PEFT adapter directory (must contain `adapter_config.json` and weights) |
| `device` | Torch device for computation (default: `"cpu"`) |
| `compute_udr` | Whether to compute Unused Dimension Ratio (requires base model norms) |
| `base_norms` | Pre-computed base model weight norms for UDR |

**Returns:** `LoRAAuditResult`

---

### `audit_lora_state_dict()`

Audit raw weight matrices directly (useful when you already have the state dict loaded).

```python
def audit_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    adapter_config: LoRAAdapterConfig,
    *,
    device: str = "cpu",
) -> LoRAAuditResult
```

---

### Helper functions

| Function | Description |
|----------|-------------|
| `find_peft_files(peft_dir)` | Locate adapter config and weight files in a directory |
| `load_peft_adapter_config(peft_dir)` | Load and parse `adapter_config.json` |
| `load_adapter_state_dict(peft_dir)` | Load adapter weights via safetensors |
| `iter_lora_pairs(state_dict)` | Iterate (name, A, B) tuples for each LoRA layer |
| `orient_lora_factors(A, B)` | Canonical orientation of LoRA factor matrices |
| `infer_module_type(name)` | Detect module type from parameter name (e.g., `q_proj`, `v_proj`) |

## Data classes

### `LoRAAuditResult`

Top-level audit result containing per-layer metrics and summary statistics.

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `layers` | `list[LoRALayerAudit]` | Per-layer audit results |
| `summary` | `dict[str, Any]` | Aggregate statistics (means, medians, percentiles) |
| `config` | `LoRAAdapterConfig` | Parsed adapter configuration |

### `LoRALayerAudit`

Per-layer spectral metrics.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Layer name (e.g., `model.layers.0.self_attn.q_proj`) |
| `module_type` | `str` | Module type (e.g., `q_proj`, `v_proj`, `k_proj`) |
| `r` | `int` | Configured LoRA rank |
| `stable_rank` | `float` | Effective dimensionality |
| `energy_rank_50` | `float` | Rank at 50% energy |
| `energy_rank_75` | `float` | Rank at 75% energy |
| `energy_rank_90` | `float` | Rank at 90% energy |
| `energy_rank_95` | `float` | Rank at 95% energy |
| `utilization` | `float` | `stable_rank / r` |
| `rank_waste` | `float` | `1 - utilization` |
| `singular_values` | `list[float]` | Full singular value spectrum |
| `udr` | `float \| None` | Unused Dimension Ratio (if computed) |

### `LoRAAdapterConfig`

Parsed adapter configuration from `adapter_config.json`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `r` | `int` | LoRA rank |
| `lora_alpha` | `float` | LoRA alpha scaling |
| `target_modules` | `list[str]` | Target module names |

### `AdapterQAArtifact`

QA eligibility artifact for a single adapter (from `build_qa_artifact()`).
