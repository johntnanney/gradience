# Gradience Public API

**Version**: v1 (gradience.vnext.telemetry/v1 schema)

This document defines the **public API surface** that Gradience commits to stability for. Everything not listed here is **internal** and may change without notice.

## Public API (Stability Guaranteed)

### CLI Commands
```bash
gradience check        # Config validation and recommendations
gradience monitor      # Live run monitoring and alerts  
gradience audit        # Post-hoc LoRA adapter analysis
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