"""
Gradience Research Module

Extended instrumentation for studying training dynamics:

- spectral_extended: Full spectrum, effective rank, decay analysis
- hessian: Curvature estimation (top eigenvalues, trace)
- phase_transitions: Critical phenomena detection
- fisher: Information geometry (Fisher matrix, natural gradient)

Usage
-----
```python
from gradience.research import (
    compute_full_spectrum,
    compute_hessian_snapshot,
    PhaseTransitionTracker,
    FisherTracker,
)
```

Research Directions
-------------------
See docs/RESEARCH_AGENDA.md for the theoretical questions we're investigating:

1. Weight spectrum ↔ Hessian spectrum relationship
2. Phase transitions in training (grokking detection)
3. Information geometry (Fisher metric and natural gradient)
4. Implicit regularization (rank dynamics)
"""

from .fisher import (
    FisherSnapshot,
    FisherTracker,
    compute_effective_dimensionality,
    compute_empirical_fisher_diagonal,
    compute_fisher_spectral_properties,
    compute_natural_gradient_alignment,
)
from .hessian import (
    HessianSnapshot,
    HessianTracker,
    compute_hessian_snapshot,
    create_loss_fn_for_batch,
    hessian_vector_product,
    hutchinson_trace,
    power_iteration_hessian,
)
from .phase_transitions import (
    GrokDetector,
    PhaseTransitionMetrics,
    PhaseTransitionTracker,
    compute_autocorrelation,
    compute_integrated_autocorr_time,
)
from .spectral_extended import (
    FullSpectralSnapshot,
    RankTracker,
    aggregate_layerwise_spectra,
    compute_full_spectrum,
    compute_layerwise_spectra,
    fit_spectral_decay,
)

__all__ = [
    # Extended spectral
    "FullSpectralSnapshot",
    "compute_full_spectrum",
    "compute_layerwise_spectra",
    "aggregate_layerwise_spectra",
    "RankTracker",
    "fit_spectral_decay",
    # Hessian
    "HessianSnapshot",
    "HessianTracker",
    "compute_hessian_snapshot",
    "hessian_vector_product",
    "power_iteration_hessian",
    "hutchinson_trace",
    "create_loss_fn_for_batch",
    # Phase transitions
    "PhaseTransitionMetrics",
    "PhaseTransitionTracker",
    "GrokDetector",
    "compute_autocorrelation",
    "compute_integrated_autocorr_time",
    # Fisher
    "FisherSnapshot",
    "FisherTracker",
    "compute_empirical_fisher_diagonal",
    "compute_fisher_spectral_properties",
    "compute_natural_gradient_alignment",
    "compute_effective_dimensionality",
]
