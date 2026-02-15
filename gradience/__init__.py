"""
Gradience: Telemetry-first observability for LoRA / PEFT fine-tuning

Gradience is a flight recorder + mechanic for LoRA runs:
- Flight recorder: emits stable JSONL telemetry (gradience.vnext.telemetry/v1)
- Mechanic: audits adapters and provides conservative rank compression suggestions

## Public API (Stability Guaranteed)

CLI Commands:
    gradience check        # Config validation and recommendations
    gradience monitor      # Live run monitoring and alerts  
    gradience audit        # Post-hoc LoRA adapter analysis

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

## Internal Implementation

Everything else is internal and may change.

Legacy components (DEPRECATED) have been moved to docs/legacy/
For current usage, see: README.md, QUICK_REFERENCE.md, USER_MANUAL.md, PUBLIC_API.md
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Should not happen on Python 3.10+, but be safe
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("gradience")
except PackageNotFoundError:
    # Fallback for development installs or editable installs
    __version__ = "0.9.1+local"

# Current API: vNext components
# For stable telemetry, use: gradience.vnext.telemetry
# For HF integration, use: gradience.vnext.integrations.hf

# Legacy components (maintained for backward compatibility, but deprecated)
import warnings

# Optional torch-dependent imports (legacy components)
try:
    # Spectral analysis (legacy - for new code use gradience.vnext.audit)
    from gradience.spectral import SpectralAnalyzer

    # Structural analysis (legacy)  
    from gradience.structural import (
        StructuralAnalyzer,
        StructuralMetrics,
        compute_muon_ratio,
        get_weight_decay_from_optimizer,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    # Create placeholder classes that provide helpful error messages
    class SpectralAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SpectralAnalyzer requires PyTorch. Install PyTorch from https://pytorch.org/get-started/ then: pip install gradience[bench]"
            )
    
    class StructuralAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "StructuralAnalyzer requires PyTorch. Install PyTorch from https://pytorch.org/get-started/ then: pip install gradience[bench]"
            )
    
    def StructuralMetrics(*args, **kwargs):
        raise ImportError(
            "StructuralMetrics requires PyTorch. Install PyTorch from https://pytorch.org/get-started/ then: pip install gradience[bench]"
        )
    
    def compute_muon_ratio(*args, **kwargs):
        raise ImportError(
            "compute_muon_ratio requires PyTorch. Install PyTorch from https://pytorch.org/get-started/ then: pip install gradience[bench]"
        )
    
    def get_weight_decay_from_optimizer(*args, **kwargs):
        raise ImportError(
            "get_weight_decay_from_optimizer requires PyTorch. Install PyTorch from https://pytorch.org/get-started/ then: pip install gradience[bench]"
        )
    
    _TORCH_AVAILABLE = False

# Stable public API (thin wrappers around CLI/module entrypoints)
from gradience import api


def __getattr__(name):
    """Lazy imports with deprecation warnings for legacy telemetry classes."""
    if name in ("TelemetryWriter", "TelemetryReader"):
        import warnings
        warnings.warn(
            f"gradience.{name} is deprecated. "
            f"Use gradience.vnext.telemetry.TelemetryWriter or "
            f"gradience.vnext.telemetry_reader.TelemetryReader instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from gradience import telemetry
        return getattr(telemetry, name)
    raise AttributeError(f"module 'gradience' has no attribute {name!r}")

__all__ = [
    # Stable public API
    "api",

    # Current (but legacy) - use gradience.vnext for new code
    "SpectralAnalyzer",
    "StructuralAnalyzer",
    "StructuralMetrics",
    "compute_muon_ratio",
    "get_weight_decay_from_optimizer",
    "TelemetryWriter",
    "TelemetryReader",
]
