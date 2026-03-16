# Architecture Overview

Gradience is organized around a clear separation between **stable public API** and **internal implementation**, with a layered architecture that keeps core analysis logic independent of framework-specific integrations.

## High-level structure

```
┌─────────────────────────────────────────────────────────┐
│                      User Interfaces                     │
│   CLI (gradience)  │  Python API (gradience.api)  │  HF │
├─────────────────────────────────────────────────────────┤
│                    Core Analysis (vnext/)                 │
│   audit/  │  merge/  │  telemetry  │  rank_suggestion   │
├─────────────────────────────────────────────────────────┤
│                    Foundation                             │
│   types  │  exceptions  │  peft_utils  │  svd_truncate  │
├─────────────────────────────────────────────────────────┤
│                    Benchmarking (bench/)                  │
│   protocol  │  task_profiles  │  configs  │  policies   │
└─────────────────────────────────────────────────────────┘
```

## Key principles

### 1. Stability tiers

Gradience maintains two API stability tiers:

- **Stable (public API)** — `gradience.api`, CLI commands, `gradience.vnext.telemetry/v1` schema, HF callback, exception hierarchy. These are backward compatible across minor releases.
- **Internal** — Everything else. May change without notice between releases.

See [API Stability Policy](../api/stability.md) for the full contract.

### 2. Lazy imports for optional dependencies

Gradience's core (`torch`, `safetensors`, `scipy`) is always available. Optional dependencies (`transformers`, `peft`, `datasets`) are imported lazily so that:

- `gradience audit` works without HuggingFace installed
- Import errors are raised only when the missing dependency is actually needed
- `DependencyError` provides clear messages about which extra to install

### 3. CLI as the canonical entrypoint

The Python API (`gradience.api`) delegates to CLI commands via `subprocess.run`. This ensures:

- CLI and Python API always produce identical results
- The CLI is the single source of truth for argument parsing and validation
- Reproducibility: the exact command can be logged and replayed

### 4. Structured artifacts

All analysis produces structured output:

- **JSON** — Machine-readable data (`audit.json`, `bench.json`, `merge_audit.json`)
- **Markdown** — Human-readable reports (`bench.md`, `merge_audit.md`)
- **JSONL** — Streaming telemetry (`run.jsonl`)

### 5. Conservative defaults

Rank suggestions and compression recommendations are deliberately conservative:

- Ranks are bucketed to standard LoRA values (1, 2, 4, 8, 16, 32, 64)
- Suggestions never recommend increasing rank above the current value
- Multiple percentile targets (median and p90) let users choose their risk tolerance

## Data flow

### Spectral audit

```
PEFT adapter directory
    │
    ▼
find_peft_files() ──► load_adapter_state_dict()
                              │
                              ▼
                     iter_lora_pairs() ──► SVD per layer
                              │
                              ▼
                     LoRAAuditResult (per-layer + summary)
                              │
                     ┌────────┼────────┐
                     ▼        ▼        ▼
                  audit.json  CLI     rank_suggestion
                             table    (compression targets)
```

### Merge audit

```
Adapter A          Adapter B
    │                  │
    ▼                  ▼
load_adapter()    load_adapter()
    │                  │
    └───────┬──────────┘
            ▼
    match_layers() ──► extract_factors()
            │
            ▼
    compute_subspace_metrics() per layer
    (principal angles, directional agreement, magnitude balance)
            │
            ▼
    assess_layer() + assess_overall()
    (CompatibilityVerdict: safe / redundant / conflicting)
            │
            ▼
    MergeAuditReport ──► JSON + Markdown
```

### Training telemetry

```
HF Trainer ──► GradienceCallback ──► TelemetryWriter ──► run.jsonl
                                                             │
                                                             ▼
                              gradience monitor ◄── TelemetryReader
                                     │
                                     ▼
                              Alerts + Recommendations
```

## Next steps

- [Design Principles](design.md) — Restraint-first philosophy and measurement approach
- [Module Map](modules.md) — Detailed module-by-module reference
