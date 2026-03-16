# Types & Data Model

::: gradience.vnext.types

Core data model shared across all Gradience modules. These types define the canonical representation for configurations, signals, and recommendations.

## Constants

### `TELEMETRY_SCHEMA_VERSION`

```python
TELEMETRY_SCHEMA_VERSION: str = "gradience.vnext.telemetry/v1"
```

Current telemetry schema version. Incremented when required fields or semantics change.

## Enums

### `TaskFamily`

Coarse task classification used for policy decisions.

```python
class TaskFamily(str, Enum):
    EASY_CLASSIFICATION = "easy_classification"
    HARD_REASONING = "hard_reasoning"
    GENERATION = "generation"
    UNKNOWN = "unknown"
```

### `Severity`

Alert and recommendation severity levels.

```python
class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### `EventType`

Telemetry event types.

```python
class EventType(str, Enum):
    AUDIT = "audit"
    TRAIN = "train"
    EVAL = "eval"
    ERROR = "error"
```

## Config snapshots

Immutable snapshots of training configuration, recorded in telemetry.

### `ConfigSnapshot`

Top-level run configuration.

```python
@dataclass(frozen=True)
class ConfigSnapshot:
    model_name: str | None = None
    dataset_name: str | None = None
    task_profile: TaskFamily = TaskFamily.UNKNOWN
    optimizer: OptimizerConfigSnapshot = ...
    lora: LoRAConfigSnapshot = ...
    training: TrainingConfigSnapshot = ...
    notes: str | None = None
    extras: dict[str, Any] = ...
```

All snapshots support `.to_dict()` and `.from_dict()` for serialization.

### `LoRAConfigSnapshot`

```python
@dataclass(frozen=True)
class LoRAConfigSnapshot:
    r: int | None = None
    alpha: float | None = None
    target_modules: list[str] = []
    dropout: float | None = None
    bias: str | None = None
    extras: dict[str, Any] = {}

    @property
    def alpha_over_r(self) -> float | None: ...
```

### `OptimizerConfigSnapshot`

```python
@dataclass(frozen=True)
class OptimizerConfigSnapshot:
    name: str | None = None
    lr: float | None = None
    weight_decay: float | None = None
    betas: tuple[float, float] | None = None
    eps: float | None = None
    extras: dict[str, Any] = {}
```

### `TrainingConfigSnapshot`

```python
@dataclass(frozen=True)
class TrainingConfigSnapshot:
    seed: int | None = None
    batch_size: int | None = None
    gradient_accumulation: int | None = None
    max_steps: int | None = None
    epochs: int | None = None
    dtype: str | None = None
    extras: dict[str, Any] = {}
```

## Metric containers

### `EvalMetrics`

Evaluation metrics for a single data split.

```python
@dataclass(frozen=True)
class EvalMetrics:
    loss: float | None = None
    ppl: float | None = None          # Perplexity
    accuracy: float | None = None
    n: int | None = None              # Number of examples
    extras: dict[str, Any] = {}
```

### `SignalSnapshot`

Cross-split signals used for regime detection.

```python
@dataclass(frozen=True)
class SignalSnapshot:
    train: EvalMetrics = ...
    test: EvalMetrics = ...
    gap: float | None = None              # e.g., test_ppl / train_ppl
    stable_rank_mean: float | None = None
    utilization_mean: float | None = None
    dominance_act_mean: float | None = None
    kappa_mean: float | None = None
    extras: dict[str, Any] = {}
```

### `Recommendation`

A human-readable recommendation emitted by policies or monitors.

```python
@dataclass(frozen=True)
class Recommendation:
    severity: Severity
    action: str
    message: str
    rationale: str | None = None
    confidence: float | None = None    # 0.0 to 1.0
    scope: str | None = None
    evidence: dict[str, Any] = {}
```

## Serialization

All data classes support round-trip serialization:

```python
config = ConfigSnapshot(model_name="mistral-7b")
d = config.to_dict()
config2 = ConfigSnapshot.from_dict(d)
assert config == config2
```
