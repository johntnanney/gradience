"""gradience.vnext.audit

Auditing modules for vNext Gradience.

Primary entrypoints:
- audit_lora_peft_dir(peft_dir, ...)
- audit_lora_state_dict(state_dict, ...)

"""

from .lora_audit import (
    LoRAAdapterConfig,
    LoRALayerAudit,
    LoRAAuditResult,
    audit_lora_peft_dir,
    audit_lora_state_dict,
    find_peft_files,
    load_peft_adapter_config,
    load_adapter_state_dict,
    # LoRA weight utilities — promoted to public API for merge audit use.
    _iter_lora_pairs as iter_lora_pairs,
    _orient_lora_factors as orient_lora_factors,
    infer_module_type,
)
from .qa_artifact import AdapterQAArtifact, build_qa_artifact
