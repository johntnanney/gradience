# Gradience Public API

**Version**: v1 (gradience.vnext.telemetry/v1 schema)

This document defines the **public API surface** that Gradience commits to stability for. Everything not listed here is **internal** and may change without notice.

## Public API (Stability Guaranteed)

### CLI Commands
```bash
gradience audit        # Spectral audit of a LoRA adapter
gradience audit-adapter
gradience merge-audit  # Merge compatibility analysis between two adapters
gradience summarize-inventory
gradience suggest-neighborhoods   # Advanced optional inventory workflow
gradience truncate     # SVD-based rank reduction of a LoRA adapter
gradience check        # Config validation and recommendations
gradience monitor      # Live training telemetry monitoring
gradience verify       # Verify installation and environment
gradience report       # Generate reports from bench artifacts
gradience explain      # Explain audit results in plain language
gradience merge-plan   # Generate a merge execution plan from audit
gradience merge        # Execute a merge plan
gradience-bench        # Run compression validation benchmark (separate entry point)
```

### Telemetry Schema
- **Schema Version**: `gradience.vnext.telemetry/v1`
- **Format**: JSONL with stable event types and field names
- **Events**: `run_start`, `train_step`, `eval`, `alert`, `metrics`, `run_end`

### HuggingFace Integration
```python
# Primary entry point for HF Trainer integration
from gradience.vnext.integrations.hf import GradienceCallback

trainer.add_callback(GradienceCallback())
```

### Rank Suggestion Functions (Pure)
```python
from gradience.vnext.rank_suggestion import (
    GlobalRankSuggestion,
    PerLayerRankSuggestion, 
    PerLayerRankSuggestionReport,
    suggest_global_ranks_from_audit,
    suggest_per_layer_ranks,
    DEFAULT_ALLOWED_RANKS,
)
```

### Python API (gradience.api)

Stable programmatic wrappers around Gradience's CLI surfaces. Use this module instead of importing internals.

```python
from gradience.api import (
    # Data containers
    BenchRunArtifacts,       # frozen dataclass: output_dir, bench_json, bench_md (all Path)
    BenchAggregateArtifacts, # frozen dataclass: output_dir, aggregate_json, aggregate_md (all Path)

    # Bench operations
    run_bench,               # Run a single benchmark protocol
    aggregate_bench_runs,    # Aggregate multiple bench runs

    # Audit & Monitor
    audit,                   # Run spectral audit on a LoRA adapter
    monitor,                 # Monitor training telemetry

    # Core preflight artifacts
    audit_adapter,           # Build AdapterQAArtifact
    merge_risk_report,       # Build MergeQAReport
    summarize_inventory,     # Build InventorySummary

    # Advanced optional wrappers
    compute_core_space_diagnostic,  # Return optional core_space block
    suggest_neighborhoods,          # Build MergeNeighborhoodReport

    # Artifact loaders
    load_bench_report,       # Load bench.json from an output directory
    load_bench_aggregate,    # Load bench_aggregate.json from an output directory
)
```

**Example: Run a benchmark and load results**
```python
from gradience.api import run_bench, load_bench_report

artifacts = run_bench(config="my_config.yaml", output="results/")
report = load_bench_report(artifacts.output_dir)
print(report["summary"])
```

**Example: Audit an adapter**
```python
from gradience.api import audit

result = audit(peft_dir="path/to/adapter", layers=True)
print(result.returncode)  # 0 on success
```

**Example: Aggregate multi-seed runs**
```python
from gradience.api import aggregate_bench_runs, load_bench_aggregate

agg = aggregate_bench_runs(
    runs=["results/seed42", "results/seed43", "results/seed45"],
    output="results/aggregate",
)
summary = load_bench_aggregate(agg.output_dir)
```

**Example: Advanced optional wrappers**
```python
from gradience.api import compute_core_space_diagnostic, suggest_neighborhoods

core_space = compute_core_space_diagnostic(
    adapter_a="./adapter_a",
    adapter_b="./adapter_b",
)

neighborhoods = suggest_neighborhoods(
    qa_dir="./qa",
    report_dir="./reports",
)
```

### Merge Audit (gradience.vnext.merge)

Spectral compatibility analysis between two LoRA adapters.

```python
from gradience.vnext.merge import merge_audit, VerdictThresholds

report = merge_audit(
    adapter_a_dir="./adapter_a",
    adapter_b_dir="./adapter_b",
    output_dir="./audit_output",       # optional: write JSON + Markdown reports
    energy_threshold=0.90,             # energy fraction for effective rank
    compute_dtype="float64",           # SVD precision
    verbose=True,                      # print progress
)

print(report.aggregate["overall_verdict"])   # "safe", "redundant", "conflicting"
print(report.aggregate["compatibility_score"])
```

### SVD Truncation (gradience.vnext.svd_truncate)

Rank reduction via SVD for LoRA adapters.

```python
from gradience.vnext.svd_truncate import svd_truncate_peft_dir

report = svd_truncate_peft_dir(
    peft_dir="./adapter",
    output_dir="./truncated",
    target_rank=4,
)
print(f"Reduced rank {report.original_rank} -> {report.target_rank}")
```

### Exception Hierarchy
```python
from gradience.exceptions import (
    GradienceError,          # Base — catch all Gradience errors
    ConfigError,             # Invalid configuration (YAML, missing fields, constraints)
    AuditError,              # Spectral audit failures (bad shapes, missing weights)
    MergeError,              # Merge audit failures (incompatible adapters)
    TelemetryError,          # Telemetry read/write errors
    TelemetrySchemaError,    # Schema version mismatch (subclass of TelemetryError)
    TelemetryFormatError,    # Malformed telemetry records (subclass of TelemetryError)
    DependencyError,         # Missing optional dependency
)

# Also available at top level:
from gradience import GradienceError
```

Each exception subclasses both `GradienceError` and the corresponding stdlib exception (`ValueError` or `RuntimeError`), so existing `except ValueError` handlers continue to work.

## Internal Implementation (May Change)

**Everything else is internal and may change.**

This includes:
- Internal helper functions
- Internal metric field names beyond the documented schema
- Experimental components (Guard, per-layer config generation)
- Implementation details of CLI commands
- Internal telemetry processing logic
- File format parsers and converters

## API Stability Policy

- **Public API**: Backward compatibility maintained within major versions
- **Telemetry Schema**: Additive changes only (new fields OK, removing fields requires major version bump)
- **CLI Commands**: Command names and basic usage patterns stable, detailed flags may evolve
- **Internal Components**: No stability guarantees, may refactor or remove

This approach prevents accidentally promising stability on experimental pieces while providing clear boundaries for users.
