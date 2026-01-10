"""
Gradience: Spectral Telemetry for Neural Network Training

Core metrics:
    - Spectral (κ, rank): Shape of learning - conditioning and expressivity
    - Structural (ρ): Balance of forces - expansion vs regularization

Components:
    - SpectralAnalyzer: Compute spectral metrics (κ, effective rank, σ_max)
    - StructuralAnalyzer: Compute muon ratio (ρ = λ × σ_max)
    - TelemetryWriter: Log metrics to JSONL
    - Guard: Checkpoint/rollback on training corruption (experimental)

Quick start:
    from gradience import SpectralAnalyzer, StructuralAnalyzer
    
    # Spectral analysis (shape of learning)
    spectral = SpectralAnalyzer()
    metrics = spectral.analyze(model)
    print(f"κ mean: {metrics['kappa_mean']:.1f}")
    
    # Structural analysis (balance of forces)
    structural = StructuralAnalyzer()
    metrics = structural.analyze(model, weight_decay=0.01)
    print(f"ρ (muon ratio): {metrics.muon_ratio:.2f}")

For HuggingFace:
    from gradience.integrations.huggingface import GradienceCallback
"""

__version__ = "0.2.0"

# Spectral (shape of learning)
from gradience.spectral import SpectralAnalyzer

# Structural (balance of forces)
from gradience.structural import (
    StructuralAnalyzer, 
    StructuralMetrics,
    compute_muon_ratio,
    get_weight_decay_from_optimizer,
)

# Telemetry
from gradience.telemetry import TelemetryWriter, TelemetryReader

# Guard (experimental)
from gradience.guard import Guard, GuardConfig, create_guard

__all__ = [
    # Spectral
    "SpectralAnalyzer",
    
    # Structural
    "StructuralAnalyzer",
    "StructuralMetrics",
    "compute_muon_ratio",
    "get_weight_decay_from_optimizer",
    
    # Telemetry
    "TelemetryWriter",
    "TelemetryReader",
    
    # Guard
    "Guard",
    "GuardConfig", 
    "create_guard",
]
