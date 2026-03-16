# Module Map

Detailed reference for every package and module in Gradience.

## Package layout

```
gradience/
├── __init__.py              # Public re-exports, version
├── __main__.py              # python -m gradience entrypoint
├── api.py                   # Stable Python API wrappers
├── cli.py                   # CLI entrypoint (argparse)
├── exceptions.py            # Exception hierarchy
├── peft_utils.py            # PEFT/LoRA weight utilities
├── policy_analysis.py       # Rank policy analysis
│
├── vnext/                   # Current-generation core
│   ├── types.py             # Shared data model
│   ├── telemetry.py         # JSONL telemetry writer
│   ├── telemetry_reader.py  # JSONL telemetry reader
│   ├── rank_suggestion.py   # Rank compression suggestions
│   ├── svd_truncate.py      # SVD-based rank reduction
│   │
│   ├── audit/               # Spectral audit
│   │   ├── lora_audit.py    # Core audit logic
│   │   ├── rank_policies.py # Rank derivation policies
│   │   ├── gain_metrics.py  # UDR computation
│   │   └── qa_artifact.py   # QA eligibility artifacts
│   │
│   ├── merge/               # Merge compatibility analysis
│   │   ├── __init__.py      # Orchestrator (merge_audit)
│   │   ├── io.py            # Adapter loading
│   │   ├── spectral_compat.py # QR-based subspace metrics
│   │   ├── verdicts.py      # Decision tree
│   │   ├── report.py        # JSON/Markdown output
│   │   ├── strategies.py    # Merge strategies
│   │   ├── executor.py      # Merge execution
│   │   ├── plan.py          # Audit-aware merge planning
│   │   ├── recommend.py     # Strategy recommendations
│   │   ├── refactor.py      # Refactoring to LoRA form
│   │   └── eligibility.py   # QA eligibility screening
│   │
│   ├── integrations/        # Framework integrations
│   │   └── hf.py            # HuggingFace Trainer callback
│   │
│   ├── policy/              # Config validation
│   │   └── check.py         # Schema + constraint checking
│   │
│   └── experimental/        # Experimental features
│       └── guard.py         # LoRAGuard (experimental)
│
├── bench/                   # Benchmarking framework
│   ├── run_bench.py         # Bench entrypoint
│   ├── protocol.py          # Orchestration
│   ├── aggregate.py         # Multi-seed aggregation
│   ├── config_schema.py     # YAML config validation
│   ├── compression.py       # Compression validation
│   ├── reporting.py         # Report generation
│   ├── decision_trace.py    # Decision audit trail
│   ├── configs/             # Built-in YAML configs
│   ├── policies/            # Built-in rank policies
│   └── task_profiles/       # Task implementations
│
├── analysis/                # Time-series analysis
├── finetune/                # LoRA fine-tuning utilities
└── research/                # Research tools (Fisher, Hessian)
```

## Module details

### `gradience.api`

Thin wrappers that invoke CLI commands via `subprocess.run`. This is the recommended programmatic interface.

**Public classes**: `BenchRunArtifacts`, `BenchAggregateArtifacts`, `AuditResult`, `MonitorResult`

**Public functions**: `run_bench()`, `aggregate_bench_runs()`, `audit()`, `monitor()`, `load_bench_report()`, `load_bench_aggregate()`

### `gradience.vnext.audit`

SVD-based spectral analysis of PEFT LoRA adapters.

**Key function**: `audit_lora_peft_dir(peft_dir, ...)` — Takes a PEFT adapter directory and returns a `LoRAAuditResult` with per-layer metrics and summary statistics.

**Also exports**: `audit_lora_state_dict()`, `find_peft_files()`, `load_peft_adapter_config()`, `iter_lora_pairs()`, `orient_lora_factors()`, `infer_module_type()`

### `gradience.vnext.merge`

Geometric compatibility analysis and merge execution for adapter pairs.

**Key functions**:

- `merge_audit(adapter_a, adapter_b)` — Spectral compatibility analysis
- `plan_from_audit(strategy, report, ...)` — Generate merge plan from audit
- `execute_merge(plan, output_dir)` — Execute the plan

**Merge strategies**: `LinearMerge`, `TIESMerge`, `DARELinearMerge`, `DARETIESMerge`, `NormEqualizedMerge`

### `gradience.vnext.telemetry`

Structured JSONL telemetry for recording training dynamics.

**Classes**: `TelemetryWriter` (write events), `TelemetryReader` (read and summarize)

**Schema**: `gradience.vnext.telemetry/v1`

### `gradience.vnext.rank_suggestion`

Pure functions for converting audit results into compression suggestions.

**Functions**: `suggest_global_ranks_from_audit()`, `suggest_per_layer_ranks()`

### `gradience.vnext.types`

Core data model shared across all modules.

**Enums**: `TaskFamily`, `Severity`, `EventType`

**Config snapshots**: `ConfigSnapshot`, `LoRAConfigSnapshot`, `OptimizerConfigSnapshot`, `TrainingConfigSnapshot`

**Metric containers**: `EvalMetrics`, `SignalSnapshot`, `Recommendation`

### `gradience.exceptions`

Exception hierarchy rooted at `GradienceError`. See [Exceptions](../api/exceptions.md).

### `gradience.bench`

Complete benchmarking framework. Entry point: `gradience-bench` CLI or `python -m gradience.bench.run_bench`.

Orchestrates: model loading, LoRA fine-tuning, compression via rank policies, evaluation, statistical aggregation, and report generation.
