# Exceptions

::: gradience.exceptions

All Gradience exceptions inherit from `GradienceError`, allowing callers to catch all Gradience-specific errors with a single `except GradienceError` clause.

## Hierarchy

```
GradienceError (Exception)
├── ConfigError (ValueError)
├── AuditError (ValueError)
├── MergeError (ValueError)
├── TelemetryError (ValueError)
│   ├── TelemetrySchemaError (ValueError)
│   └── TelemetryFormatError (ValueError)
└── DependencyError (RuntimeError)
```

Each exception also inherits from a standard Python exception type for compatibility.

## Exception types

### `GradienceError`

Base exception for all Gradience errors. Catch this to handle any Gradience-specific failure.

```python
from gradience.exceptions import GradienceError

try:
    result = audit_lora_peft_dir(path)
except GradienceError as e:
    print(f"Gradience error: {e}")
```

### `ConfigError`

Raised for invalid configuration: YAML parsing errors, missing required fields, constraint violations.

```python
from gradience.exceptions import ConfigError
```

### `AuditError`

Raised for spectral audit failures: missing weight files, incompatible shapes, SVD computation errors.

```python
from gradience.exceptions import AuditError
```

### `MergeError`

Raised for merge audit failures: incompatible adapters, shape mismatches between adapter pairs.

```python
from gradience.exceptions import MergeError
```

### `TelemetryError`

Base class for telemetry read/write errors.

```python
from gradience.exceptions import TelemetryError
```

### `TelemetrySchemaError`

Raised when a telemetry file's schema version does not match the expected version.

```python
from gradience.exceptions import TelemetrySchemaError
```

### `TelemetryFormatError`

Raised when a telemetry record is missing required envelope fields (`schema`, `ts`, `run_id`, `event`).

```python
from gradience.exceptions import TelemetryFormatError
```

### `DependencyError`

Raised when an optional dependency is missing. Includes guidance on which extra to install.

```python
from gradience.exceptions import DependencyError

# Example: using HF integration without transformers installed
# raises DependencyError("transformers is required. Install with: pip install gradience[hf]")
```

## Usage patterns

### Catch specific errors

```python
from gradience.exceptions import AuditError, ConfigError

try:
    result = audit_lora_peft_dir(path)
except AuditError as e:
    print(f"Audit failed: {e}")
except ConfigError as e:
    print(f"Bad config: {e}")
```

### Catch all Gradience errors

```python
from gradience.exceptions import GradienceError

try:
    result = audit_lora_peft_dir(path)
except GradienceError as e:
    print(f"Something went wrong: {e}")
```

### Catch via standard types

Since each exception also inherits from a standard type, existing error handling works:

```python
try:
    result = audit_lora_peft_dir(path)
except ValueError as e:
    # Catches ConfigError, AuditError, MergeError, TelemetryError
    pass
except RuntimeError as e:
    # Catches DependencyError
    pass
```
