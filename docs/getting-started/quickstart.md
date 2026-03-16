# Quick Start

This guide walks you through Gradience's core workflows in under 5 minutes.

## 1. Audit a LoRA adapter

The `audit` command performs SVD-based spectral analysis on a PEFT LoRA adapter:

```bash
gradience audit --peft-dir ./my-lora-adapter --layers
```

This produces:

- **Per-layer metrics**: stable rank, energy rank (at multiple percentiles), utilization ratio, rank waste
- **Summary statistics**: mean/median/p90 across all layers
- **Rank suggestions**: conservative compression targets based on energy concentration

Add `--json` to get machine-readable output:

```bash
gradience audit --peft-dir ./my-lora-adapter --layers --json > audit.json
```

### What the metrics mean

| Metric | Description |
|--------|-------------|
| `stable_rank` | Effective dimensionality (ratio of squared Frobenius norm to squared spectral norm) |
| `energy_rank_90` | Number of singular values needed to capture 90% of total energy |
| `utilization` | `stable_rank / r` — fraction of allocated rank actually used |
| `rank_waste` | `1 - utilization` — fraction of allocated rank that is unused |
| `UDR` | Unused Dimension Ratio — a finer measure of wasted capacity |

## 2. Compare two adapters for merge compatibility

The `merge-audit` command measures geometric compatibility between two adapters:

```bash
gradience merge-audit --adapter-a ./adapter-1 --adapter-b ./adapter-2
```

This produces:

- **Per-layer analysis**: principal angles, directional agreement, magnitude balance
- **Compatibility verdict**: `safe`, `redundant`, or `conflicting`
- **Output files**: `merge_audit.json` and `merge_audit.md`

## 3. Monitor training telemetry

If you've recorded telemetry during training (via the [HuggingFace callback](huggingface.md) or `TelemetryWriter`):

```bash
gradience monitor run.jsonl --verbose
```

This analyzes spectral evolution over time and flags potential issues.

## 4. Validate a configuration

```bash
gradience check config.yaml
```

Validates your bench/experiment YAML configuration before running.

## Using the Python API

All CLI commands are also available programmatically:

```python
import gradience.api as gapi

# Run an audit
result = gapi.audit(peft_dir="./my-adapter", layers=True)
assert result.success

# Run the bench protocol
artifacts = gapi.run_bench(config="config.yaml", output="results/")
print(f"Results: {artifacts.bench_json}")

# Load and inspect results
report = gapi.load_bench_report("results/")
```

## Next steps

- [HuggingFace Integration](huggingface.md) — Record telemetry during training
- [CLI Reference](../guide/cli.md) — Full command documentation
- [API Reference](../api/index.md) — Python API details
- [Configuration](../guide/configuration.md) — YAML config schema
