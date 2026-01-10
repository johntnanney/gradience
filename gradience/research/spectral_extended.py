"""
Extended Spectral Analysis for Training Dynamics Research

This module provides comprehensive spectral measurements beyond κ̃:

- Full singular value spectrum
- Effective rank (entropy-based)
- Stable rank (Frobenius/operator norm ratio)
- Spectral decay rate (power-law fit)
- Nuclear norm (trace norm)
- Layer-wise analysis

These measurements support research on implicit regularization,
rank dynamics, and the geometry of training.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import time


@dataclass
class FullSpectralSnapshot:
    """Complete spectral characterization of a weight matrix."""
    
    step: int
    timestamp: float
    layer_name: str
    
    # Shape info
    rows: int
    cols: int
    
    # Full spectrum (sorted descending)
    singular_values: List[float]
    
    # Derived quantities
    sigma_max: float              # Largest singular value
    sigma_min: float              # Smallest singular value (> threshold)
    kappa: float                  # Condition number
    
    frobenius_norm: float         # ||W||_F = sqrt(sum σ_i²)
    nuclear_norm: float           # ||W||_* = sum σ_i
    operator_norm: float          # ||W||_2 = σ_max
    
    effective_rank: float         # exp(entropy of normalized spectrum)
    stable_rank: float            # ||W||_F² / ||W||_2²
    numerical_rank: float         # Count of σ_i > threshold
    
    spectral_decay_alpha: Optional[float]  # Power-law exponent if fitted
    spectral_decay_r2: Optional[float]     # Fit quality
    
    # Energy distribution
    top1_energy: float            # σ_1² / sum σ_i²
    top10_energy: float           # sum(σ_1:10)² / sum σ_i²
    tail_energy: float            # 1 - top10_energy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "layer_name": self.layer_name,
            "shape": [self.rows, self.cols],
            "sigma_max": self.sigma_max,
            "sigma_min": self.sigma_min,
            "kappa": self.kappa,
            "frobenius_norm": self.frobenius_norm,
            "nuclear_norm": self.nuclear_norm,
            "operator_norm": self.operator_norm,
            "effective_rank": self.effective_rank,
            "stable_rank": self.stable_rank,
            "numerical_rank": self.numerical_rank,
            "spectral_decay_alpha": self.spectral_decay_alpha,
            "spectral_decay_r2": self.spectral_decay_r2,
            "top1_energy": self.top1_energy,
            "top10_energy": self.top10_energy,
            "tail_energy": self.tail_energy,
            "n_singular_values": len(self.singular_values),
        }


def compute_full_spectrum(
    weight_tensor,
    layer_name: str = "unknown",
    step: int = 0,
    rank_threshold: float = 1e-6,
) -> FullSpectralSnapshot:
    """
    Compute complete spectral characterization of a weight matrix.
    
    Parameters
    ----------
    weight_tensor : torch.Tensor
        Weight matrix (will be reshaped to 2D if needed)
    layer_name : str
        Identifier for this layer
    step : int
        Training step
    rank_threshold : float
        Singular values below this are considered zero
    
    Returns
    -------
    FullSpectralSnapshot
        Complete spectral characterization
    """
    import torch
    
    with torch.no_grad():
        W = weight_tensor.float()
        
        # Reshape to 2D if needed
        original_shape = W.shape
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        elif W.dim() == 1:
            W = W.unsqueeze(0)
        
        rows, cols = W.shape
        
        # Full SVD
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            singular_values = S.cpu().tolist()
        except Exception:
            # Fallback for numerical issues
            singular_values = [1.0]
        
        # Filter and sort
        singular_values = sorted([s for s in singular_values if s > rank_threshold], reverse=True)
        
        if len(singular_values) == 0:
            singular_values = [rank_threshold]
        
        # Basic quantities
        sigma_max = singular_values[0]
        sigma_min = singular_values[-1] if singular_values[-1] > rank_threshold else rank_threshold
        kappa = sigma_max / sigma_min
        
        # Norms
        sq_sum = sum(s**2 for s in singular_values)
        frobenius_norm = math.sqrt(sq_sum)
        nuclear_norm = sum(singular_values)
        operator_norm = sigma_max
        
        # Rank measures
        # Effective rank: exp(entropy of normalized singular value distribution)
        total = sum(singular_values)
        if total > 0:
            probs = [s / total for s in singular_values]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            effective_rank = math.exp(entropy)
        else:
            effective_rank = 1.0
        
        # Stable rank: ||W||_F² / ||W||_2²
        stable_rank = sq_sum / (sigma_max**2 + 1e-10)
        
        # Numerical rank: count of significant singular values
        numerical_rank = sum(1 for s in singular_values if s > rank_threshold * sigma_max)
        
        # Energy distribution
        if sq_sum > 0:
            top1_energy = singular_values[0]**2 / sq_sum
            top10_sq = sum(s**2 for s in singular_values[:10])
            top10_energy = top10_sq / sq_sum
            tail_energy = 1.0 - top10_energy
        else:
            top1_energy = top10_energy = 1.0
            tail_energy = 0.0
        
        # Spectral decay: fit power law σ_i ~ i^(-α)
        alpha, r2 = fit_spectral_decay(singular_values)
        
        return FullSpectralSnapshot(
            step=step,
            timestamp=time.time(),
            layer_name=layer_name,
            rows=rows,
            cols=cols,
            singular_values=singular_values,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            kappa=kappa,
            frobenius_norm=frobenius_norm,
            nuclear_norm=nuclear_norm,
            operator_norm=operator_norm,
            effective_rank=effective_rank,
            stable_rank=stable_rank,
            numerical_rank=numerical_rank,
            spectral_decay_alpha=alpha,
            spectral_decay_r2=r2,
            top1_energy=top1_energy,
            top10_energy=top10_energy,
            tail_energy=tail_energy,
        )


def fit_spectral_decay(singular_values: List[float], min_points: int = 5) -> Tuple[Optional[float], Optional[float]]:
    """
    Fit power-law decay: σ_i ~ i^(-α)
    
    Uses log-log linear regression: log(σ_i) = -α log(i) + c
    
    Returns
    -------
    alpha : float or None
        Decay exponent (larger = faster decay = lower effective rank)
    r2 : float or None
        R² of the fit (quality measure)
    """
    if len(singular_values) < min_points:
        return None, None
    
    # Use log-log regression
    n = len(singular_values)
    log_i = [math.log(i + 1) for i in range(n)]  # +1 to avoid log(0)
    log_s = [math.log(s + 1e-10) for s in singular_values]
    
    # Linear regression: log_s = -alpha * log_i + c
    mean_x = sum(log_i) / n
    mean_y = sum(log_s) / n
    
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_i, log_s))
    denominator = sum((x - mean_x)**2 for x in log_i)
    
    if denominator < 1e-10:
        return None, None
    
    slope = numerator / denominator
    alpha = -slope  # Negative because we expect decay
    
    # Compute R²
    ss_res = sum((y - (slope * x + (mean_y - slope * mean_x)))**2 for x, y in zip(log_i, log_s))
    ss_tot = sum((y - mean_y)**2 for y in log_s)
    
    r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
    
    return alpha, r2


@dataclass
class RankTracker:
    """
    Tracks rank evolution across training.
    
    Maintains history for analyzing rank dynamics:
    - Smooth vs jumpy evolution
    - Compression vs expansion phases
    - Correlation with loss dynamics
    """
    
    window_size: int = 100
    _history: List[Dict[str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        self._history = []
    
    def add(self, step: int, effective_rank: float, stable_rank: float, 
            numerical_rank: float, layer_name: str = "aggregate") -> None:
        """Record rank observation."""
        self._history.append({
            "step": step,
            "effective_rank": effective_rank,
            "stable_rank": stable_rank,
            "numerical_rank": numerical_rank,
            "layer_name": layer_name,
            "timestamp": time.time(),
        })
        
        # Keep window
        if len(self._history) > self.window_size * 10:
            self._history = self._history[-self.window_size * 5:]
    
    def get_trajectory(self, layer_name: Optional[str] = None) -> List[Dict[str, float]]:
        """Get rank trajectory for a layer (or all if None)."""
        if layer_name is None:
            return self._history
        return [h for h in self._history if h["layer_name"] == layer_name]
    
    def compute_rank_velocity(self, layer_name: Optional[str] = None) -> Optional[float]:
        """Compute rate of change of effective rank."""
        traj = self.get_trajectory(layer_name)
        if len(traj) < 3:
            return None
        
        recent = traj[-self.window_size:] if len(traj) > self.window_size else traj
        
        steps = [h["step"] for h in recent]
        ranks = [h["effective_rank"] for h in recent]
        
        # Linear regression for slope
        n = len(steps)
        mean_s = sum(steps) / n
        mean_r = sum(ranks) / n
        
        num = sum((s - mean_s) * (r - mean_r) for s, r in zip(steps, ranks))
        den = sum((s - mean_s)**2 for s in steps)
        
        if den < 1e-10:
            return 0.0
        
        return num / den  # Δrank / Δstep
    
    def compute_rank_volatility(self, layer_name: Optional[str] = None) -> Optional[float]:
        """Compute volatility (std) of effective rank."""
        traj = self.get_trajectory(layer_name)
        if len(traj) < 3:
            return None
        
        recent = traj[-self.window_size:] if len(traj) > self.window_size else traj
        ranks = [h["effective_rank"] for h in recent]
        
        mean_r = sum(ranks) / len(ranks)
        variance = sum((r - mean_r)**2 for r in ranks) / len(ranks)
        
        return math.sqrt(variance)
    
    def detect_rank_phase(self) -> str:
        """
        Classify current phase of rank evolution.
        
        Returns one of:
        - "expanding": Rank increasing
        - "compressing": Rank decreasing
        - "stable": Rank roughly constant
        - "volatile": Rank fluctuating
        - "unknown": Insufficient data
        """
        velocity = self.compute_rank_velocity()
        volatility = self.compute_rank_volatility()
        
        if velocity is None or volatility is None:
            return "unknown"
        
        # Thresholds (may need tuning)
        if volatility > 0.5:
            return "volatile"
        elif velocity > 0.01:
            return "expanding"
        elif velocity < -0.01:
            return "compressing"
        else:
            return "stable"


def compute_layerwise_spectra(
    model,
    step: int = 0,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[FullSpectralSnapshot]:
    """
    Compute full spectral characterization for all weight matrices.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to analyze
    step : int
        Current training step
    include_patterns : list of str, optional
        Only include layers matching these patterns
    exclude_patterns : list of str, optional
        Exclude layers matching these patterns
    
    Returns
    -------
    list of FullSpectralSnapshot
        One per weight matrix
    """
    import re
    
    results = []
    
    for name, param in model.named_parameters():
        # Skip non-weight parameters
        if "weight" not in name:
            continue
        
        # Skip 1D parameters (biases, LayerNorm)
        if param.dim() < 2:
            continue
        
        # Apply filters
        if include_patterns:
            if not any(re.search(p, name) for p in include_patterns):
                continue
        
        if exclude_patterns:
            if any(re.search(p, name) for p in exclude_patterns):
                continue
        
        snapshot = compute_full_spectrum(
            param.data,
            layer_name=name,
            step=step,
        )
        results.append(snapshot)
    
    return results


def aggregate_layerwise_spectra(snapshots: List[FullSpectralSnapshot]) -> Dict[str, float]:
    """
    Aggregate layer-wise spectral statistics.
    
    Returns summary statistics across all layers.
    """
    if not snapshots:
        return {}
    
    n = len(snapshots)
    
    return {
        "n_layers": n,
        "mean_effective_rank": sum(s.effective_rank for s in snapshots) / n,
        "mean_stable_rank": sum(s.stable_rank for s in snapshots) / n,
        "mean_kappa": sum(s.kappa for s in snapshots) / n,
        "max_kappa": max(s.kappa for s in snapshots),
        "mean_spectral_decay": sum(s.spectral_decay_alpha or 0 for s in snapshots) / n,
        "mean_top1_energy": sum(s.top1_energy for s in snapshots) / n,
        "total_nuclear_norm": sum(s.nuclear_norm for s in snapshots),
        "total_frobenius_norm": math.sqrt(sum(s.frobenius_norm**2 for s in snapshots)),
    }
