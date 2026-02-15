"""
Gradience: Telemetry-first observability for LoRA / PEFT fine-tuning

Gradience is a flight recorder + mechanic for LoRA runs:
- Flight recorder: emits stable JSONL telemetry (gradience.vnext.telemetry/v1)
- Mechanic: audits adapters and provides conservative rank compression suggestions
- Merge auditor: spectral compatibility analysis between LoRA adapter pairs

## Public API (Stability Guaranteed)

CLI Commands:
    gradience check        # Config validation and recommendations
    gradience monitor      # Live run monitoring and alerts
    gradience audit        # Post-hoc LoRA adapter analysis
    gradience merge-audit  # Spectral compatibility between two adapters

HuggingFace Integration:
    from gradience.vnext.integrations.hf import GradienceCallback
    trainer.add_callback(GradienceCallback())

Telemetry Schema:
    gradience.vnext.telemetry/v1 - Stable JSONL schema

Rank Suggestions:
    from gradience.vnext.rank_suggestion import (
        suggest_global_ranks_from_audit,
        suggest_per_layer_ranks,
        GlobalRankSuggestion,
        PerLayerRankSuggestionReport,
    )

Merge Compatibility Audit:
    from gradience.vnext.merge import (
        merge_audit,
        VerdictThresholds,
        MergeAuditReport,
        CompatibilityVerdict,
    )

## Internal Implementation

Everything else is internal and may change.

Legacy components (DEPRECATED) have been removed.
For current usage, see: README.md, QUICK_REFERENCE.md, USER_MANUAL.md, PUBLIC_API.md
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Should not happen on Python 3.10+, but be safe
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("gradience")
except Exception:  # Intentionally broad: outermost fallback for development installs
    __version__ = "0.9.8"

# Current API: vNext components re-exported for convenience
from gradience.vnext.telemetry import TelemetryWriter, TelemetryReader

# Stable public API (thin wrappers around CLI/module entrypoints)
from gradience import api

# Deprecated Guard functionality
import warnings

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
    # Stable public API
    "api",

    # vNext telemetry (canonical)
    "TelemetryWriter",
    "TelemetryReader",

    # Deprecated (will raise ImportError with helpful message)
    "Guard",
    "GuardConfig",
    "create_guard",
]
