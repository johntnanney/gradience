# Python API

::: gradience.api

The `gradience.api` module provides stable Python wrappers around Gradience's CLI commands. Use this module instead of importing internals.

```python
import gradience.api as gapi
```

## Functions

### `run_bench()`

Run the Bench protocol.

```python
def run_bench(
    *,
    config: str | Path,
    output: str | Path,
    smoke: bool = False,
    ci: bool = False,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> BenchRunArtifacts
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str \| Path` | Path to YAML bench configuration |
| `output` | `str \| Path` | Output directory for results |
| `smoke` | `bool` | Run in smoke test mode (faster, fewer steps) |
| `ci` | `bool` | Enable CI mode (stricter validation) |
| `python` | `str \| None` | Python executable to use (default: `sys.executable`) |
| `env` | `Mapping \| None` | Additional environment variables |
| `log_path` | `str \| Path \| None` | Write stdout/stderr to this file |
| `check` | `bool` | Raise on non-zero exit (default: `True`) |

**Returns:** `BenchRunArtifacts` with paths to `bench.json` and `bench.md`.

---

### `aggregate_bench_runs()`

Aggregate multiple Bench runs into statistical summaries.

```python
def aggregate_bench_runs(
    *,
    runs: Sequence[str | Path],
    output: str | Path,
    include_smoke: bool = False,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> BenchAggregateArtifacts
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `runs` | `Sequence[str \| Path]` | Directories of individual bench runs |
| `output` | `str \| Path` | Output directory for aggregate results |
| `include_smoke` | `bool` | Include smoke test runs in aggregation |

**Returns:** `BenchAggregateArtifacts` with paths to `bench_aggregate.json` and `bench_aggregate.md`.

---

### `audit()`

Run spectral audit on a PEFT LoRA adapter.

```python
def audit(
    *,
    peft_dir: str | Path,
    layers: bool = True,
    base_model: str | None = None,
    base_norms_cache: str | Path | None = None,
    no_udr: bool = False,
    extra_args: Sequence[str] | None = None,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> AuditResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `peft_dir` | `str \| Path` | Path to PEFT adapter directory |
| `layers` | `bool` | Include per-layer metrics (default: `True`) |
| `base_model` | `str \| None` | Base model name/path for UDR computation |
| `base_norms_cache` | `str \| Path \| None` | Cached base model norms |
| `no_udr` | `bool` | Skip UDR computation |
| `extra_args` | `Sequence[str] \| None` | Additional CLI arguments |

**Returns:** `AuditResult` with `success` property and optional `log_path`.

---

### `monitor()`

Run training telemetry analysis.

```python
def monitor(
    *,
    run_jsonl: str | Path,
    verbose: bool = False,
    extra_args: Sequence[str] | None = None,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> MonitorResult
```

**Returns:** `MonitorResult` with `success` property and optional `log_path`.

---

### `load_bench_report()`

Load a `bench.json` artifact.

```python
def load_bench_report(output_dir: str | Path) -> dict[str, Any]
```

---

### `load_bench_aggregate()`

Load a `bench_aggregate.json` artifact.

```python
def load_bench_aggregate(output_dir: str | Path) -> dict[str, Any]
```

## Data classes

### `BenchRunArtifacts`

```python
@dataclass(frozen=True)
class BenchRunArtifacts:
    output_dir: Path
    bench_json: Path
    bench_md: Path
```

### `BenchAggregateArtifacts`

```python
@dataclass(frozen=True)
class BenchAggregateArtifacts:
    output_dir: Path
    aggregate_json: Path
    aggregate_md: Path
```

### `AuditResult`

```python
@dataclass(frozen=True)
class AuditResult:
    returncode: int
    log_path: Path | None = None

    @property
    def success(self) -> bool: ...
```

### `MonitorResult`

```python
@dataclass(frozen=True)
class MonitorResult:
    returncode: int
    log_path: Path | None = None

    @property
    def success(self) -> bool: ...
```
