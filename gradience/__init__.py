"""
Gradience: Telemetry-first observability for LoRA / PEFT fine-tuning

Gradience is a flight recorder + mechanic for LoRA runs:
- Flight recorder: emits stable JSONL telemetry (gradience.vnext.telemetry/v1)
- Mechanic: audits adapters and provides conservative rank compression suggestions

Canonical API:
    # HuggingFace integration (recommended for most users)
    from gradience.vnext.integrations.hf import GradienceCallback
    trainer.add_callback(GradienceCallback())
    
    # CLI workflow
    gradience check --task seq_cls --peft-dir my_adapter
    gradience monitor run.jsonl  
    gradience audit --peft-dir my_adapter --layers --suggest-per-layer --json

Legacy components (DEPRECATED) have been moved to docs/legacy/
For current usage, see: README.md, QUICK_REFERENCE.md, USER_MANUAL.md
"""

__version__ = "0.2.0"

# Current API: vNext components
# For stable telemetry, use: gradience.vnext.telemetry
# For HF integration, use: gradience.vnext.integrations.hf

# Legacy components (maintained for backward compatibility, but deprecated)
import warnings

# Spectral analysis (legacy - for new code use gradience.vnext.audit)
from gradience.spectral import SpectralAnalyzer

# Structural analysis (legacy)  
from gradience.structural import (
    StructuralAnalyzer,
    StructuralMetrics,
    compute_muon_ratio,
    get_weight_decay_from_optimizer,
)

# Legacy telemetry (for new code use gradience.vnext.telemetry)
from gradience.telemetry import TelemetryWriter, TelemetryReader

# Deprecated Guard functionality
def _deprecated_guard_import():
    warnings.warn(
        "Guard functionality has been moved to docs/legacy/ and is no longer supported. "
        "Use gradience.vnext.integrations for framework integration instead.",
        DeprecationWarning,
        stacklevel=3
    )
    raise ImportError("Guard functionality is deprecated. See docs/legacy/ for archived code.")

# Create placeholder functions that raise deprecation warnings
def Guard(*args, **kwargs):
    _deprecated_guard_import()

def GuardConfig(*args, **kwargs):
    _deprecated_guard_import()
    
def create_guard(*args, **kwargs):
    _deprecated_guard_import()

__all__ = [
    # Current (but legacy) - use gradience.vnext for new code
    "SpectralAnalyzer",
    "StructuralAnalyzer", 
    "StructuralMetrics",
    "compute_muon_ratio",
    "get_weight_decay_from_optimizer",
    "TelemetryWriter",
    "TelemetryReader",
    
    # Deprecated (will raise ImportError with helpful message)
    "Guard",
    "GuardConfig",
    "create_guard",
]
