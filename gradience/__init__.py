"""
Gradience: Spectral analysis of low-rank adaptation dynamics

Gradience is a research instrument for studying the geometry of LoRA fine-tuning:
- Spectral measurement: SVD-based analysis of rank structure, energy concentration,
  and utilization across adapter layers
- Training telemetry: structured JSONL recording (gradience.vnext.telemetry/v1)
  of spectral evolution during training
- Merge analysis: geometric compatibility between adapter pairs via principal
  angles, directional agreement, and magnitude balance

## Public API (Stability Guaranteed)

CLI Commands:
    gradience check        # Config validation
    gradience monitor      # Training telemetry analysis
    gradience audit        # Spectral measurement of LoRA adapters
    gradience merge-audit  # Geometric compatibility between adapter pairs

HuggingFace Integration:
    from gradience.vnext.integrations.hf import GradienceCallback
    trainer.add_callback(GradienceCallback())

Telemetry Schema:
    gradience.vnext.telemetry/v1 - Stable JSONL schema

Spectral Analysis:
    from gradience.vnext.rank_suggestion import (
        suggest_global_ranks_from_audit,
        suggest_per_layer_ranks,
        GlobalRankSuggestion,
        PerLayerRankSuggestionReport,
    )

Merge Compatibility Analysis:
    from gradience.vnext.merge import (
        merge_audit,
        VerdictThresholds,
        MergeAuditReport,
        CompatibilityVerdict,
    )

## Internal Implementation

Everything else is internal and may change.

Legacy components (DEPRECATED) have been removed.
For current usage, see: README.md, docs/QUICK_REFERENCE.md, docs/USER_MANUAL.md, PUBLIC_API.md
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Should not happen on Python 3.10+, but be safe
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("gradience")
except PackageNotFoundError:
    __version__ = "0.10.0"

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
