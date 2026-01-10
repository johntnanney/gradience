"""
Hessian Estimation for Training Dynamics Research

This module provides tractable Hessian measurements:

- Top-k eigenvalues via power iteration / Lanczos
- Trace estimation via Hutchinson's method
- Hessian-vector products (the primitive operation)

These enable studying the loss landscape curvature and its
relationship to weight matrix spectra.

Computational Notes
-------------------
Full Hessian is O(p²) storage and O(p³) to compute, where p = #parameters.
For a 100M parameter model, that's 10^16 elements—completely infeasible.

Instead, we use:
- Hessian-vector products: O(p) via autodiff
- Power iteration: O(k * n_iters) Hessian-vector products for top-k eigenvalues
- Hutchinson trace: O(n_samples) Hessian-vector products

These are tractable even for large models.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
import time


@dataclass
class HessianSnapshot:
    """Snapshot of Hessian spectral properties."""
    
    step: int
    timestamp: float
    
    # Top eigenvalues (from power iteration or Lanczos)
    top_eigenvalues: List[float]
    
    # Derived quantities
    lambda_max: float             # Largest eigenvalue
    lambda_min_approx: Optional[float]  # Smallest (if computed)
    hessian_kappa: Optional[float]      # Condition number of Hessian
    
    # Trace estimates
    trace: Optional[float]        # tr(H) via Hutchinson
    trace_variance: Optional[float]  # Variance of trace estimate
    
    # Curvature statistics
    mean_curvature: Optional[float]   # trace / n_params
    spectral_norm: float          # ||H||_2 = |λ_max|
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "top_eigenvalues": self.top_eigenvalues,
            "lambda_max": self.lambda_max,
            "lambda_min_approx": self.lambda_min_approx,
            "hessian_kappa": self.hessian_kappa,
            "trace": self.trace,
            "trace_variance": self.trace_variance,
            "mean_curvature": self.mean_curvature,
            "spectral_norm": self.spectral_norm,
        }


def hessian_vector_product(
    loss_fn: Callable,
    params: List,
    vector: List,
) -> List:
    """
    Compute Hessian-vector product Hv using autodiff.
    
    Uses the identity: Hv = ∂/∂θ (∇L · v)
    
    Parameters
    ----------
    loss_fn : callable
        Function that computes the loss (must be differentiable)
    params : list of Tensor
        Model parameters
    vector : list of Tensor
        Vector to multiply (same structure as params)
    
    Returns
    -------
    Hv : list of Tensor
        Hessian-vector product (same structure as params)
    """
    import torch
    
    # First: compute gradient
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Second: compute gradient of (grad · vector)
    grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vector))
    
    Hv = torch.autograd.grad(grad_dot_v, params)
    
    return list(Hv)


def power_iteration_hessian(
    loss_fn: Callable,
    params: List,
    n_iterations: int = 20,
    n_eigenvalues: int = 1,
    tolerance: float = 1e-5,
) -> List[float]:
    """
    Estimate top eigenvalues of Hessian via power iteration.
    
    For multiple eigenvalues, uses deflation (orthogonalize against
    previously found eigenvectors).
    
    Parameters
    ----------
    loss_fn : callable
        Function computing the loss
    params : list of Tensor
        Model parameters
    n_iterations : int
        Power iterations per eigenvalue
    n_eigenvalues : int
        Number of top eigenvalues to compute
    tolerance : float
        Convergence threshold
    
    Returns
    -------
    eigenvalues : list of float
        Top eigenvalues in descending order
    """
    import torch
    
    eigenvalues = []
    eigenvectors = []  # For deflation
    
    for k in range(n_eigenvalues):
        # Initialize random vector
        v = [torch.randn_like(p) for p in params]
        v = normalize_vector(v)
        
        # Orthogonalize against previous eigenvectors
        for ev in eigenvectors:
            proj = sum((vi * evi).sum() for vi, evi in zip(v, ev))
            v = [vi - proj * evi for vi, evi in zip(v, ev)]
        v = normalize_vector(v)
        
        # Power iteration
        prev_eigenvalue = None
        for i in range(n_iterations):
            # Hv
            Hv = hessian_vector_product(loss_fn, params, v)
            
            # Orthogonalize against previous eigenvectors
            for ev in eigenvectors:
                proj = sum((hvi * evi).sum() for hvi, evi in zip(Hv, ev))
                Hv = [hvi - proj * evi for hvi, evi in zip(Hv, ev)]
            
            # Rayleigh quotient: λ = v^T H v / v^T v = v^T H v (since ||v|| = 1)
            eigenvalue = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv)).item()
            
            # Normalize
            v = normalize_vector(Hv)
            
            # Check convergence
            if prev_eigenvalue is not None:
                if abs(eigenvalue - prev_eigenvalue) < tolerance:
                    break
            prev_eigenvalue = eigenvalue
        
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
    
    return eigenvalues


def hutchinson_trace(
    loss_fn: Callable,
    params: List,
    n_samples: int = 100,
) -> Tuple[float, float]:
    """
    Estimate trace of Hessian using Hutchinson's method.
    
    Uses the identity: E[z^T H z] = tr(H) where z is Rademacher random.
    
    Parameters
    ----------
    loss_fn : callable
        Function computing the loss
    params : list of Tensor
        Model parameters
    n_samples : int
        Number of random vectors for estimation
    
    Returns
    -------
    trace : float
        Estimated trace
    variance : float
        Variance of estimate
    """
    import torch
    
    estimates = []
    
    for _ in range(n_samples):
        # Rademacher random vector: ±1 with equal probability
        z = [torch.sign(torch.randn_like(p)) for p in params]
        
        # Hz
        Hz = hessian_vector_product(loss_fn, params, z)
        
        # z^T H z
        trace_est = sum((zi * Hzi).sum() for zi, Hzi in zip(z, Hz)).item()
        estimates.append(trace_est)
    
    trace = sum(estimates) / len(estimates)
    variance = sum((e - trace)**2 for e in estimates) / len(estimates)
    
    return trace, variance


def normalize_vector(v: List) -> List:
    """Normalize a parameter-structured vector to unit norm."""
    import torch
    
    norm = math.sqrt(sum((vi**2).sum().item() for vi in v))
    if norm < 1e-10:
        return v
    return [vi / norm for vi in v]


def compute_hessian_snapshot(
    model,
    loss_fn: Callable,
    step: int = 0,
    n_top_eigenvalues: int = 5,
    n_trace_samples: int = 50,
    power_iterations: int = 20,
) -> HessianSnapshot:
    """
    Compute Hessian spectral snapshot.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model
    loss_fn : callable
        Function that computes the loss (should create computation graph)
    step : int
        Training step
    n_top_eigenvalues : int
        Number of top eigenvalues to compute
    n_trace_samples : int
        Hutchinson samples for trace
    power_iterations : int
        Iterations for power method
    
    Returns
    -------
    HessianSnapshot
        Spectral properties of the Hessian
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    # Top eigenvalues
    top_eigs = power_iteration_hessian(
        loss_fn, params,
        n_iterations=power_iterations,
        n_eigenvalues=n_top_eigenvalues,
    )
    
    lambda_max = top_eigs[0] if top_eigs else 0.0
    
    # Trace
    trace, trace_var = hutchinson_trace(loss_fn, params, n_samples=n_trace_samples)
    
    # Derived quantities
    mean_curvature = trace / n_params if n_params > 0 else 0.0
    
    # We don't have λ_min easily, but we can estimate from trace
    # If spectrum were uniform: λ_min ≈ trace/n - (n-1) * spread
    # This is a rough heuristic
    lambda_min_approx = None
    hessian_kappa = None
    
    return HessianSnapshot(
        step=step,
        timestamp=time.time(),
        top_eigenvalues=top_eigs,
        lambda_max=lambda_max,
        lambda_min_approx=lambda_min_approx,
        hessian_kappa=hessian_kappa,
        trace=trace,
        trace_variance=trace_var,
        mean_curvature=mean_curvature,
        spectral_norm=abs(lambda_max),
    )


@dataclass
class HessianTracker:
    """
    Track Hessian properties over training.
    
    Maintains history for analyzing curvature dynamics
    and correlation with weight spectra.
    """
    
    _history: List[HessianSnapshot] = field(default_factory=list)
    
    def __post_init__(self):
        self._history = []
    
    def add(self, snapshot: HessianSnapshot) -> None:
        """Record Hessian observation."""
        self._history.append(snapshot)
    
    def get_trajectory(self, field: str = "lambda_max") -> List[Tuple[int, float]]:
        """Get trajectory of a specific field."""
        return [(h.step, getattr(h, field)) for h in self._history]
    
    def compute_curvature_velocity(self, window: int = 20) -> Optional[float]:
        """Compute rate of change of max eigenvalue."""
        if len(self._history) < 3:
            return None
        
        recent = self._history[-window:]
        steps = [h.step for h in recent]
        values = [h.lambda_max for h in recent]
        
        n = len(steps)
        mean_s = sum(steps) / n
        mean_v = sum(values) / n
        
        num = sum((s - mean_s) * (v - mean_v) for s, v in zip(steps, values))
        den = sum((s - mean_s)**2 for s in steps)
        
        if den < 1e-10:
            return 0.0
        
        return num / den


def create_loss_fn_for_batch(model, batch, loss_criterion):
    """
    Create a closure that computes loss for Hessian estimation.
    
    Usage:
        loss_fn = create_loss_fn_for_batch(model, batch, nn.CrossEntropyLoss())
        snapshot = compute_hessian_snapshot(model, loss_fn, step=100)
    """
    def loss_fn():
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        if hasattr(outputs, "loss"):
            return outputs.loss
        elif hasattr(outputs, "logits"):
            return loss_criterion(outputs.logits, batch["labels"])
        else:
            return loss_criterion(outputs, batch["labels"])
    
    return loss_fn
