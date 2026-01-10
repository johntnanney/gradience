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

Notes:
- `.safetensors` requires the `safetensors` package.
- Computation uses r×r eigendecompositions in float64 by default (cheap/stable).
