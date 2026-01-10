"""
Gradience vNext

This subpackage defines the canonical data model and telemetry schema used by
the "restraint navigator" architecture (check / audit / monitor).

Design goals:
- Stable, versioned JSONL telemetry schema
- Typed config + metric snapshots that can be passed between components
- Backwards compatible: legacy gradience.telemetry remains unchanged
"""

from .types import (
    TELEMETRY_SCHEMA_VERSION,
    TaskProfile,
    Severity,
    LoRAConfigSnapshot,
    OptimizerConfigSnapshot,
    TrainingConfigSnapshot,
    ConfigSnapshot,
    EvalMetrics,
    SignalSnapshot,
    Recommendation,
)

from .telemetry import TelemetryWriter, TelemetryReader

# Policies (interpretation)
from .policy import check_config, check_run

# Audits (measurement -> summarized metrics)
from .audit import (
    LoRAAdapterConfig,
    LoRALayerAudit,
    LoRAAuditResult,
    audit_lora_peft_dir,
    audit_lora_state_dict,
)

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
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
