# Design Principles

## Restraint-first philosophy

Gradience is a **measurement instrument**, not an optimizer. It provides spectral observations and conservative suggestions, leaving decisions to the researcher.

Key implications:

- **Report, don't act.** The default is to measure and present findings. Automatic interventions (like the experimental Guard) are off by default and clearly flagged.
- **Conservative suggestions.** Rank suggestions round up to standard LoRA buckets and never recommend increasing rank. This is intentionally cautious.
- **Evidence over heuristics.** Every recommendation carries its evidence trail (`evidence` dict) so the researcher can audit the reasoning.

## Measurement approach

### SVD-based spectral analysis

All spectral metrics are derived from the singular value decomposition of LoRA weight matrices. Given a LoRA layer with factors A (r×d_in) and B (d_out×r):

1. Compute the effective weight delta: ΔW = B·A
2. Perform SVD: ΔW = UΣV^T
3. Extract metrics from the singular value spectrum σ₁ ≥ σ₂ ≥ ... ≥ σᵣ

### Core metrics

- **Stable rank**: `‖ΔW‖²_F / σ₁²` — Continuous measure of effective dimensionality
- **Energy rank at p%**: Minimum k such that `Σᵢ₌₁ᵏ σᵢ² ≥ p% · Σᵢ₌₁ⁿ σᵢ²`
- **Utilization**: `stable_rank / r` — What fraction of allocated rank is used
- **UDR** (Unused Dimension Ratio): Finer measure of wasted capacity considering the full spectrum

### Subspace analysis (merge audit)

For comparing two adapters, Gradience computes:

- **Principal angles** between the column spaces of ΔW_A and ΔW_B
- **Directional agreement** via cosine similarity of the dominant singular vectors
- **Magnitude balance** as the ratio of spectral norms

## Schema design

### Telemetry (JSONL)

The telemetry schema (`gradience.vnext.telemetry/v1`) follows these principles:

- **Append-only JSONL** — Each line is a self-contained JSON object with a schema version, timestamp, and event type
- **Forward-compatible** — Extra keys are allowed and ignored by older readers
- **Schema-versioned** — Breaking changes require a schema version bump

### Configuration (YAML)

Bench configurations are YAML files validated against a strict schema. This ensures:

- Invalid configs fail fast with clear error messages
- All config options are documented and type-checked
- Default values are explicit, not implicit

## Error handling

All Gradience exceptions inherit from `GradienceError`, enabling broad catches while still allowing specific handling:

```python
from gradience.exceptions import GradienceError, AuditError, ConfigError

try:
    result = audit_lora_peft_dir(path)
except AuditError as e:
    # Handle audit-specific failure
except GradienceError as e:
    # Catch-all for any Gradience error
```

Each exception type also inherits from a standard Python exception (`ValueError`, `RuntimeError`) for compatibility with code that catches standard types.
