# API Stability Policy

Gradience maintains a clear contract for what is stable and what may change.

## Stability tiers

### Stable (backward compatible)

These interfaces will not have breaking changes within a major version:

| Surface | Import / Access |
|---------|----------------|
| CLI commands | `gradience audit`, `gradience merge-audit`, `gradience monitor`, etc. |
| Python API | `import gradience.api` |
| Telemetry schema | `gradience.vnext.telemetry/v1` |
| HF callback | `from gradience.vnext.integrations.hf import GradienceCallback` |
| Rank suggestion | `from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit` |
| Merge analysis | `from gradience.vnext.merge import merge_audit` |
| Exception hierarchy | `from gradience.exceptions import GradienceError` |
| Core types | `from gradience.vnext.types import ConfigSnapshot, Severity, TaskFamily` |

### Internal (may change)

Everything not listed above is internal. Common internal modules:

- `gradience.bench.protocol` — Use `gradience.api.run_bench()` instead
- `gradience.vnext.audit.lora_audit` — Use `gradience.vnext.audit.audit_lora_peft_dir()` instead
- `gradience.vnext.merge.spectral_compat` — Use `gradience.vnext.merge.merge_audit()` instead

## Versioning

Gradience follows [Semantic Versioning](https://semver.org/):

- **Patch** (0.11.x): Bug fixes, documentation. No API changes.
- **Minor** (0.x.0): New features. Stable API remains backward compatible.
- **Major** (x.0.0): May include breaking changes to stable API (with migration guide).

## Deprecation process

When a stable API is being replaced:

1. The old API emits `DeprecationWarning` with migration guidance
2. The old API continues to work for at least one minor release
3. The old API is removed in the next major release

## Recommendations for users

```python
# Good: use stable imports
import gradience.api as gapi
from gradience.vnext.merge import merge_audit
from gradience.exceptions import GradienceError

# Bad: importing internals (may break)
from gradience.bench.protocol import run_protocol  # internal
from gradience.vnext.audit.lora_audit import _compute_stable_rank  # private
```

For reproducible results, pin to a specific version:

```bash
pip install "gradience==0.11.0"
```
