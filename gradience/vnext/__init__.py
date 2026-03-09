"""
Gradience vNext

This subpackage defines the canonical data model and telemetry schema used by
the "restraint navigator" architecture (check / audit / monitor).

Design goals:
- Stable, versioned JSONL telemetry schema
- Typed config + metric snapshots that can be passed between components
- Backwards compatible: legacy gradience.telemetry remains unchanged

## Public API Components

Only the exports in __all__ are considered public API with stability guarantees.
Everything else is internal and may change without notice.
"""

# Audits (measurement -> summarized metrics)
from .audit import (
    LoRAAdapterConfig,
    LoRAAuditResult,
    LoRALayerAudit,
    audit_lora_peft_dir,
    audit_lora_state_dict,
)

# Policies (interpretation)
from .policy import check_config, check_run
from .telemetry import TelemetryReader, TelemetryWriter
from .types import (
    TELEMETRY_SCHEMA_VERSION,
    ConfigSnapshot,
    EvalMetrics,
    LoRAConfigSnapshot,
    OptimizerConfigSnapshot,
    Recommendation,
    Severity,
    SignalSnapshot,
    TaskFamily,
    TaskProfile,
    TrainingConfigSnapshot,
)

# Framework integrations (optional dependencies)
# Canonical import: from gradience.vnext.integrations.hf import GradienceCallback
# NOTE: GradienceCallback is NOT eagerly imported here to avoid pulling in
# transformers at module load time. Import directly from the submodule:
#   from gradience.vnext.integrations.hf import GradienceCallback

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "TaskFamily",
    "TaskProfile",
    "Severity",
    "LoRAConfigSnapshot",
    "OptimizerConfigSnapshot",
    "TrainingConfigSnapshot",
    "ConfigSnapshot",
    "EvalMetrics",
    "SignalSnapshot",
    "Recommendation",
    "TelemetryWriter",
    "TelemetryReader",
    "check_config",
    "check_run",
    # Audit
    "LoRAAdapterConfig",
    "LoRALayerAudit",
    "LoRAAuditResult",
    "audit_lora_peft_dir",
    "audit_lora_state_dict",
]
