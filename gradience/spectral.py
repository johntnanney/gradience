"""
Spectral Analysis Module

Tracks condition number proxy to monitor training geometry.

The condition number κ(W) = σ_max / σ_min measures sensitivity of matrix
operations to perturbations. High κ → ill-conditioned → unstable gradients.

Key insight: condition number slope often leads loss degradation, providing early warning.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from enum import Enum
import time


class RiskLevel(Enum):
    """Training risk classification based on condition number dynamics."""
    STABLE = "STABLE"           # slope < 0.0005: quiescent
    LEARNING = "LEARNING"       # 0.0005-0.002: healthy training
    WARNING = "WARNING"         # 0.002-0.004: elevated risk
    DANGER = "DANGER"           # > 0.004: high risk, intervene
    INSTABILITY = "INSTABILITY" # volatility > threshold: chaotic


# Threshold table (from recommendations)
RISK_THRESHOLDS = {
    "slope_stable": 0.0005,
    "slope_learning": 0.002,
    "slope_warning": 0.004,
    "volatility_instability": 0.003,
}

# Policy recommendations per risk level
RISK_POLICY = {
    RiskLevel.STABLE: {
        "telemetry": "sparse",
        "snapshot": "normal",
        "action": None,
    },
    RiskLevel.LEARNING: {
        "telemetry": "normal",
        "snapshot": "normal", 
        "action": None,
    },
    RiskLevel.WARNING: {
        "telemetry": "burst",
        "snapshot": "accelerated",
        "action": "alert",
    },
    RiskLevel.DANGER: {
        "telemetry": "burst",
        "snapshot": "immediate",
        "action": "lr_backoff",
    },
    RiskLevel.INSTABILITY: {
        "telemetry": "burst",
        "snapshot": "immediate",
        "action": "rollback_or_halt",
    },
}


@dataclass
class SpectralSnapshot:
    """Single observation of spectral metrics."""
    step: int
    timestamp: float
    sigma_max: float
    sigma_min: Optional[float]
    kappa_tilde: float
    layer_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "spectral_sigma_max": self.sigma_max,
            "spectral_sigma_min": self.sigma_min,
            "spectral_kappa": self.kappa_tilde,
            "layer_name": self.layer_name,
        }


@dataclass
class SpectralAnalyzer:
    """
    Tracks spectral metrics over time and computes derived statistics.
    
    Maintains rolling window for:
    - Condition number slope (trend indicator via linear regression)
    - Volatility (stability indicator via rolling std)
    - Risk assessment (actionable classification)
    
    Parameters
    ----------
    window_size : int
        Observations for rolling statistics (default: 50)
    """
    window_size: int = 50
    
    # Configurable thresholds
    slope_threshold_stable: float = RISK_THRESHOLDS["slope_stable"]
    slope_threshold_learning: float = RISK_THRESHOLDS["slope_learning"]
    slope_threshold_warning: float = RISK_THRESHOLDS["slope_warning"]
    volatility_threshold: float = RISK_THRESHOLDS["volatility_instability"]
    
    # Internal state
    _history: deque = field(default_factory=deque)
    _kappa_values: deque = field(default_factory=deque)
    
    def __post_init__(self):
        self._history = deque(maxlen=self.window_size)
        self._kappa_values = deque(maxlen=self.window_size)
    
    def add_observation(self, snapshot: SpectralSnapshot) -> None:
        """Record a new spectral observation."""
        self._history.append(snapshot)
        self._kappa_values.append(snapshot.kappa_tilde)
    
    def compute_slope(self) -> Optional[float]:
        """
        Compute condition number slope via OLS over the window.
        
        Returns
        -------
        slope : float or None
            Δκ/Δstep, or None if < 3 observations
        """
        if len(self._history) < 3:
            return None
        
        n = len(self._history)
        steps = [s.step for s in self._history]
        kappas = list(self._kappa_values)
        
        mean_step = sum(steps) / n
        mean_kappa = sum(kappas) / n
        
        numerator = sum((s - mean_step) * (k - mean_kappa) 
                       for s, k in zip(steps, kappas))
        denominator = sum((s - mean_step) ** 2 for s in steps)
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def compute_volatility(self) -> Optional[float]:
        """
        Compute condition number volatility as rolling standard deviation.
        
        Returns
        -------
        volatility : float or None
            std(κ) over window, or None if < 3 observations
        """
        if len(self._kappa_values) < 3:
            return None
        
        kappas = list(self._kappa_values)
        mean_k = sum(kappas) / len(kappas)
        variance = sum((k - mean_k) ** 2 for k in kappas) / len(kappas)
        
        return math.sqrt(variance)
    
    def assess_risk(self) -> Tuple[RiskLevel, Dict[str, Any]]:
        """
        Assess current training risk.
        
        Returns
        -------
        level : RiskLevel
            Current classification
        metrics : dict
            slope, volatility, window_size
        """
        slope = self.compute_slope()
        volatility = self.compute_volatility()
        
        metrics = {
            "spectral_slope": slope,
            "spectral_volatility": volatility,
            "spectral_window_size": len(self._history),
        }
        
        if slope is None or volatility is None:
            return RiskLevel.STABLE, metrics
        
        # Volatility trumps slope
        if volatility > self.volatility_threshold:
            return RiskLevel.INSTABILITY, metrics
        
        # Assess by slope magnitude
        abs_slope = abs(slope)
        if abs_slope > self.slope_threshold_warning:
            return RiskLevel.DANGER, metrics
        elif abs_slope > self.slope_threshold_learning:
            return RiskLevel.WARNING, metrics
        elif abs_slope > self.slope_threshold_stable:
            return RiskLevel.LEARNING, metrics
        else:
            return RiskLevel.STABLE, metrics
    
    def get_current_kappa(self) -> Optional[float]:
        """Most recent condition number value."""
        return self._kappa_values[-1] if self._kappa_values else None
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current spectral telemetry."""
        risk_level, metrics = self.assess_risk()
        return {
            "spectral_kappa": self.get_current_kappa(),
            "risk_level": risk_level.value,
            **metrics,
        }


def estimate_spectral_norm(weight_tensor, n_iterations: int = 10) -> float:
    """
    Estimate largest singular value via power iteration.
    
    O(n²) per iteration vs O(n³) for full SVD.
    
    Parameters
    ----------
    weight_tensor : torch.Tensor
        2D weight matrix
    n_iterations : int
        Power iteration steps (10 usually sufficient)
    
    Returns
    -------
    sigma_max : float
        Estimated largest singular value
    """
    import torch
    
    with torch.no_grad():
        W = weight_tensor.float()
        if W.dim() != 2:
            # Reshape to 2D if needed
            W = W.view(W.size(0), -1)
        
        # Initialize random vector
        v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-8)
        
        # Power iteration
        for _ in range(n_iterations):
            u = W @ v
            u = u / (u.norm() + 1e-8)
            v = W.T @ u
            v = v / (v.norm() + 1e-8)
        
        # σ_max = ||Wv||
        sigma_max = (W @ v).norm().item()
        
    return sigma_max


def estimate_condition_proxy(
    weight_tensor,
    n_iterations: int = 10,
) -> Tuple[float, float, float]:
    """
    Estimate condition number proxy.
    
    Uses power iteration for σ_max and Frobenius heuristic for σ_min.
    
    Returns
    -------
    sigma_max, sigma_min, kappa_tilde
    """
    import torch
    
    sigma_max = estimate_spectral_norm(weight_tensor, n_iterations)
    
    with torch.no_grad():
        W = weight_tensor.float()
        if W.dim() != 2:
            W = W.view(W.size(0), -1)
        
        # Heuristic: σ_min ≈ ||W||_F / (sqrt(rank) * scale_factor)
        fro_norm = W.norm().item()
        approx_rank = min(W.shape)
        sigma_min = fro_norm / (math.sqrt(approx_rank) * 10)
    
    kappa_tilde = sigma_max / (sigma_min + 1e-8)
    
    return sigma_max, sigma_min, kappa_tilde
