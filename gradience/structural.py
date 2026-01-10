"""
Structural Integrity Metrics

The "balance of forces" view of training dynamics:
- Weight decay (λ) pulls weights toward zero
- Learning pushes weights outward (σ_max grows)
- The ratio ρ = λ × σ_max measures who's winning

When ρ ≈ 1: Equilibrium. Model learns but stays compact.
When ρ > 1: Expansion winning. Weights growing faster than decay can hold.
When ρ < 1: Decay winning. Model is shrinking/stable.

This is complementary to spectral metrics (κ, rank) which measure
the *shape* of learning. The Muon Ratio measures the *magnitude balance*.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class StructuralMetrics:
    """Structural integrity metrics for a model."""
    
    # The Muon Ratio: λ × σ_max
    # Named after the balance of forces (like gravity vs expansion in cosmology)
    muon_ratio: float
    
    # Components
    weight_decay: float      # λ from optimizer
    sigma_max_mean: float    # Average spectral norm across layers
    sigma_max_max: float     # Max spectral norm (worst case)
    
    # Per-layer breakdown
    per_layer_sigma_max: Dict[str, float]
    per_layer_muon: Dict[str, float]  # λ × σ_max for each layer
    
    # Derived
    layers_above_threshold: int  # Layers where ρ > 1
    expansion_pressure: float    # How far above equilibrium (ρ - 1)
    
    def is_stable(self, threshold: float = 1.0) -> bool:
        """Check if model is in stable regime."""
        return self.muon_ratio <= threshold
    
    def to_dict(self) -> Dict:
        return {
            'muon_ratio': self.muon_ratio,
            'weight_decay': self.weight_decay,
            'sigma_max_mean': self.sigma_max_mean,
            'sigma_max_max': self.sigma_max_max,
            'layers_above_threshold': self.layers_above_threshold,
            'expansion_pressure': self.expansion_pressure,
        }


class StructuralAnalyzer:
    """
    Analyze structural integrity of training.
    
    The key insight: training is a balance of forces.
    - Weight decay (λ) acts like "gravity" pulling weights to zero
    - Learning acts like "expansion" pushing weights outward
    
    The Muon Ratio (ρ = λ × σ_max) measures this balance:
    - ρ ≈ 1: Healthy equilibrium
    - ρ > 1: Runaway expansion (weights growing too fast)
    - ρ < 1: Stable regime (decay is winning)
    
    Usage:
        analyzer = StructuralAnalyzer()
        
        # Get weight decay from optimizer
        weight_decay = optimizer.defaults.get('weight_decay', 0.01)
        
        # Analyze
        metrics = analyzer.analyze(model, weight_decay)
        
        if metrics.muon_ratio > 1.0:
            print("Warning: Model expanding faster than decay can control")
            print(f"Consider reducing LR by {metrics.muon_ratio:.1f}x")
    """
    
    def __init__(self, min_layer_size: int = 10):
        """
        Args:
            min_layer_size: Minimum matrix dimension to analyze
        """
        self.min_layer_size = min_layer_size
        self.history: List[StructuralMetrics] = []
    
    def analyze(
        self, 
        model: nn.Module, 
        weight_decay: float,
        threshold: float = 1.0,
    ) -> StructuralMetrics:
        """
        Compute structural integrity metrics.
        
        Args:
            model: The model to analyze
            weight_decay: Weight decay value (λ) from optimizer
            threshold: Muon ratio threshold for stability
            
        Returns:
            StructuralMetrics with muon ratio and related metrics
        """
        per_layer_sigma = {}
        per_layer_muon = {}
        
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if min(param.shape) < self.min_layer_size:
                continue
            if 'weight' not in name:
                continue
            
            with torch.no_grad():
                W = param.float()
                if W.dim() > 2:
                    W = W.view(W.size(0), -1)
                
                try:
                    # Compute spectral norm (largest singular value)
                    # Using power iteration for efficiency on large matrices
                    if min(W.shape) > 1000:
                        # Approximate for very large matrices
                        sigma_max = self._approx_spectral_norm(W)
                    else:
                        S = torch.linalg.svdvals(W)
                        sigma_max = S[0].item()
                    
                    per_layer_sigma[name] = sigma_max
                    per_layer_muon[name] = weight_decay * sigma_max
                    
                except Exception:
                    continue
        
        if not per_layer_sigma:
            return StructuralMetrics(
                muon_ratio=0.0,
                weight_decay=weight_decay,
                sigma_max_mean=0.0,
                sigma_max_max=0.0,
                per_layer_sigma_max={},
                per_layer_muon={},
                layers_above_threshold=0,
                expansion_pressure=0.0,
            )
        
        sigma_values = list(per_layer_sigma.values())
        muon_values = list(per_layer_muon.values())
        
        sigma_max_mean = sum(sigma_values) / len(sigma_values)
        sigma_max_max = max(sigma_values)
        
        # Overall muon ratio (using mean)
        muon_ratio = weight_decay * sigma_max_mean
        
        # Count layers above threshold
        layers_above = sum(1 for m in muon_values if m > threshold)
        
        # Expansion pressure: how far above equilibrium
        expansion_pressure = max(0, muon_ratio - threshold)
        
        metrics = StructuralMetrics(
            muon_ratio=muon_ratio,
            weight_decay=weight_decay,
            sigma_max_mean=sigma_max_mean,
            sigma_max_max=sigma_max_max,
            per_layer_sigma_max=per_layer_sigma,
            per_layer_muon=per_layer_muon,
            layers_above_threshold=layers_above,
            expansion_pressure=expansion_pressure,
        )
        
        self.history.append(metrics)
        return metrics
    
    def _approx_spectral_norm(self, W: torch.Tensor, n_iters: int = 10) -> float:
        """Approximate spectral norm using power iteration."""
        with torch.no_grad():
            # Initialize random vector
            v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
            v = v / v.norm()
            
            for _ in range(n_iters):
                u = W @ v
                u = u / (u.norm() + 1e-10)
                v = W.t() @ u
                v = v / (v.norm() + 1e-10)
            
            # Spectral norm
            sigma = (u @ W @ v).item()
            return abs(sigma)
    
    def get_trajectory(self) -> List[float]:
        """Get muon ratio trajectory over time."""
        return [m.muon_ratio for m in self.history]
    
    def is_expanding(self, window: int = 5, threshold: float = 0.1) -> bool:
        """Check if muon ratio is trending upward (expansion accelerating)."""
        if len(self.history) < window * 2:
            return False
        
        recent = [m.muon_ratio for m in self.history[-window:]]
        earlier = [m.muon_ratio for m in self.history[-window*2:-window]]
        
        recent_mean = sum(recent) / len(recent)
        earlier_mean = sum(earlier) / len(earlier)
        
        growth = (recent_mean - earlier_mean) / (earlier_mean + 1e-10)
        return growth > threshold
    
    def suggest_lr_adjustment(self) -> Tuple[float, str]:
        """
        Suggest learning rate adjustment based on muon ratio.
        
        Returns:
            (multiplier, explanation)
            
        Example:
            mult, reason = analyzer.suggest_lr_adjustment()
            if mult < 1.0:
                new_lr = current_lr * mult
        """
        if not self.history:
            return 1.0, "No data yet"
        
        current = self.history[-1]
        
        if current.muon_ratio <= 0.8:
            return 1.2, f"ρ={current.muon_ratio:.2f} (stable). Could increase LR by 20%."
        
        elif current.muon_ratio <= 1.0:
            return 1.0, f"ρ={current.muon_ratio:.2f} (equilibrium). LR is appropriate."
        
        elif current.muon_ratio <= 1.5:
            factor = 1.0 / current.muon_ratio
            return factor, f"ρ={current.muon_ratio:.2f} (mild expansion). Reduce LR by {(1-factor)*100:.0f}%."
        
        else:
            factor = 0.5
            return factor, f"ρ={current.muon_ratio:.2f} (runaway). Reduce LR by 50% immediately."


def get_weight_decay_from_optimizer(optimizer: torch.optim.Optimizer) -> float:
    """Extract weight decay from optimizer."""
    # Check optimizer defaults
    if hasattr(optimizer, 'defaults'):
        wd = optimizer.defaults.get('weight_decay', 0.0)
        if wd > 0:
            return wd
    
    # Check param groups
    for group in optimizer.param_groups:
        wd = group.get('weight_decay', 0.0)
        if wd > 0:
            return wd
    
    return 0.0


# Convenience function
def compute_muon_ratio(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Quick computation of muon ratio.
    
    Usage:
        rho = compute_muon_ratio(model, optimizer)
        if rho > 1.0:
            print("Warning: expansion exceeding regularization")
    """
    weight_decay = get_weight_decay_from_optimizer(optimizer)
    analyzer = StructuralAnalyzer()
    metrics = analyzer.analyze(model, weight_decay)
    return metrics.muon_ratio
