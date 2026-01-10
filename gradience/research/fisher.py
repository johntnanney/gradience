"""
Fisher Information for Training Dynamics Research

Implements Fisher Information estimation and related information-geometric quantities:

- Empirical Fisher matrix (outer product of gradients)
- Fisher spectral properties
- Natural gradient alignment
- Effective dimensionality

Information Geometry Background
-------------------------------
The Fisher Information Matrix F defines a Riemannian metric on parameter space:

    F_ij = E[∂log p(y|x,θ)/∂θ_i · ∂log p(y|x,θ)/∂θ_j]

This is the *expected* outer product of score functions. Key properties:

1. F is positive semi-definite
2. F = H (Hessian) for exponential family with natural parameters
3. F ≈ H under certain regularity conditions

The natural gradient is: ∇̃L = F⁻¹∇L
- Steepest descent in Fisher metric (not Euclidean)
- Invariant to parameterization
- Related to second-order methods

High κ(F) means parameter space is anisotropic—some directions carry much more
information about the output distribution than others. This geometric distortion
makes first-order optimization difficult.

Relation to Training Stability
------------------------------
Hypothesis: Training instability may correspond to extreme anisotropy in Fisher metric.
- High κ(F) → SGD step sizes are wrong in most directions
- Mismatch between Euclidean LR and Fisher geometry

We can test this by tracking F spectra alongside weight spectra.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
import time


@dataclass
class FisherSnapshot:
    """Snapshot of Fisher Information properties."""
    
    step: int
    timestamp: float
    n_samples: int  # Samples used for estimation
    
    # Spectral properties
    top_eigenvalues: List[float]
    lambda_max: float
    lambda_min_approx: Optional[float]
    fisher_kappa: Optional[float]  # Condition number
    
    # Trace and norms
    trace: float                   # tr(F)
    frobenius_norm: float          # ||F||_F
    
    # Effective dimensionality
    effective_dim: float           # tr(F)² / tr(F²) or similar
    participation_ratio: float     # Another effective dim measure
    
    # Natural gradient alignment
    natural_grad_alignment: Optional[float]  # cos(∇L, F⁻¹∇L)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "top_eigenvalues": self.top_eigenvalues,
            "fisher_lambda_max": self.lambda_max,
            "fisher_lambda_min": self.lambda_min_approx,
            "fisher_kappa": self.fisher_kappa,
            "fisher_trace": self.trace,
            "fisher_frobenius": self.frobenius_norm,
            "effective_dim": self.effective_dim,
            "participation_ratio": self.participation_ratio,
            "natural_grad_alignment": self.natural_grad_alignment,
        }


def compute_empirical_fisher_diagonal(
    model,
    data_loader,
    n_samples: int = 100,
) -> List:
    """
    Compute diagonal of empirical Fisher matrix.
    
    F_ii ≈ (1/n) Σ (∂L/∂θ_i)²
    
    Much cheaper than full Fisher—O(p) vs O(p²) storage.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model
    data_loader : iterable
        Data for Fisher estimation
    n_samples : int
        Number of samples to use
    
    Returns
    -------
    fisher_diag : list of Tensor
        Diagonal elements, same structure as parameters
    """
    import torch
    
    # Initialize accumulator
    fisher_diag = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
    
    model.eval()
    count = 0
    
    for batch in data_loader:
        if count >= n_samples:
            break
        
        # Move to device
        if isinstance(batch, dict):
            batch = {k: v.to(next(model.parameters()).device) 
                    for k, v in batch.items() if hasattr(v, 'to')}
        
        # Forward pass
        model.zero_grad()
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            # Assume classification
            loss = torch.nn.functional.cross_entropy(
                outputs.logits if hasattr(outputs, 'logits') else outputs,
                batch['labels']
            )
        
        # Backward pass
        loss.backward()
        
        # Accumulate squared gradients
        for i, p in enumerate(model.parameters()):
            if p.requires_grad and p.grad is not None:
                fisher_diag[i] += p.grad.data ** 2
        
        count += 1
    
    # Average
    for i in range(len(fisher_diag)):
        fisher_diag[i] /= count
    
    model.train()
    return fisher_diag


def compute_fisher_spectral_properties(
    model,
    data_loader,
    n_samples: int = 50,
    n_eigenvalues: int = 5,
    power_iterations: int = 20,
) -> Tuple[List[float], float, float]:
    """
    Compute top eigenvalues and trace of empirical Fisher.
    
    Uses power iteration with Fisher-vector products:
    Fv ≈ (1/n) Σ g_i (g_i · v) where g_i = ∂L_i/∂θ
    
    Parameters
    ----------
    model : torch.nn.Module
        The model
    data_loader : iterable
        Data for Fisher estimation
    n_samples : int
        Samples for Fisher estimation
    n_eigenvalues : int
        Number of top eigenvalues
    power_iterations : int
        Iterations per eigenvalue
    
    Returns
    -------
    eigenvalues : list of float
        Top eigenvalues
    trace : float
        Trace estimate
    frobenius : float
        Frobenius norm estimate
    """
    import torch
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Collect gradients for Fisher estimation
    gradients = []
    
    model.eval()
    for i, batch in enumerate(data_loader):
        if i >= n_samples:
            break
        
        if isinstance(batch, dict):
            batch = {k: v.to(next(model.parameters()).device)
                    for k, v in batch.items() if hasattr(v, 'to')}
        
        model.zero_grad()
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            loss = torch.nn.functional.cross_entropy(
                outputs.logits if hasattr(outputs, 'logits') else outputs,
                batch['labels']
            )
        
        loss.backward()
        
        grad = [p.grad.data.clone().flatten() for p in params if p.grad is not None]
        gradients.append(torch.cat(grad))
    
    model.train()
    
    if not gradients:
        return [], 0.0, 0.0
    
    # Stack gradients: (n_samples, n_params)
    G = torch.stack(gradients)
    n, d = G.shape
    
    # Trace: tr(F) = E[||g||²] = (1/n) Σ ||g_i||²
    trace = (G ** 2).sum().item() / n
    
    # Frobenius: ||F||_F² = E[||g||²]² + Var(g·g') ≈ trace² / d (rough)
    frobenius = math.sqrt(trace)  # Approximation
    
    # Power iteration for top eigenvalues
    # F = (1/n) G^T G, so Fv = (1/n) G^T (G v)
    eigenvalues = []
    V = []  # Eigenvectors for deflation
    
    for k in range(n_eigenvalues):
        v = torch.randn(d, device=G.device)
        v = v / (v.norm() + 1e-8)
        
        # Orthogonalize against previous eigenvectors
        for u in V:
            v = v - (v @ u) * u
        v = v / (v.norm() + 1e-8)
        
        for _ in range(power_iterations):
            # Fv = (1/n) G^T (G v)
            Gv = G @ v
            Fv = G.T @ Gv / n
            
            # Deflation
            for u in V:
                Fv = Fv - (Fv @ u) * u
            
            eigenvalue = (v @ Fv).item()
            v = Fv / (Fv.norm() + 1e-8)
        
        eigenvalues.append(eigenvalue)
        V.append(v)
    
    return eigenvalues, trace, frobenius


def compute_natural_gradient_alignment(
    model,
    batch,
    fisher_diag: List,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute alignment between gradient and approximate natural gradient.
    
    cos(∇L, F̂⁻¹∇L) where F̂ is diagonal Fisher
    
    High alignment → SGD is acting like natural gradient
    Low alignment → geometry is highly distorted
    
    Parameters
    ----------
    model : torch.nn.Module
        The model
    batch : dict or Tensor
        Input batch
    fisher_diag : list of Tensor
        Diagonal Fisher (from compute_empirical_fisher_diagonal)
    epsilon : float
        Regularization for Fisher inversion
    
    Returns
    -------
    alignment : float
        Cosine similarity in [0, 1]
    """
    import torch
    
    # Compute current gradient
    model.zero_grad()
    outputs = model(**batch) if isinstance(batch, dict) else model(batch)
    
    if hasattr(outputs, 'loss'):
        loss = outputs.loss
    else:
        loss = torch.nn.functional.cross_entropy(
            outputs.logits if hasattr(outputs, 'logits') else outputs,
            batch['labels']
        )
    
    loss.backward()
    
    # Extract gradient
    grad = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grad.append(p.grad.data.clone())
    
    # Compute natural gradient: F^{-1} g (diagonal approximation)
    natural_grad = []
    for g, f in zip(grad, fisher_diag):
        ng = g / (f + epsilon)
        natural_grad.append(ng)
    
    # Flatten for dot product
    g_flat = torch.cat([g.flatten() for g in grad])
    ng_flat = torch.cat([ng.flatten() for ng in natural_grad])
    
    # Cosine similarity
    dot = (g_flat * ng_flat).sum()
    norm_g = g_flat.norm()
    norm_ng = ng_flat.norm()
    
    if norm_g < 1e-8 or norm_ng < 1e-8:
        return 0.0
    
    alignment = (dot / (norm_g * norm_ng)).item()
    return abs(alignment)  # Take absolute value


def compute_effective_dimensionality(fisher_diag: List) -> Tuple[float, float]:
    """
    Compute effective dimensionality from diagonal Fisher.
    
    Two measures:
    1. Participation ratio: (Σ λ_i)² / Σ λ_i² = tr(F)² / tr(F²)
    2. Entropy-based: exp(entropy of normalized spectrum)
    
    Parameters
    ----------
    fisher_diag : list of Tensor
        Diagonal Fisher elements
    
    Returns
    -------
    participation_ratio : float
    entropy_dim : float
    """
    import torch
    
    # Flatten to single vector
    flat = torch.cat([f.flatten() for f in fisher_diag])
    
    # Ensure non-negative (should be, but numerics)
    flat = torch.clamp(flat, min=0)
    
    trace = flat.sum().item()
    trace_sq = (flat ** 2).sum().item()
    
    if trace_sq < 1e-10:
        return 1.0, 1.0
    
    # Participation ratio
    pr = trace ** 2 / trace_sq
    
    # Entropy-based
    probs = flat / (trace + 1e-10)
    probs = probs[probs > 1e-10]  # Filter zeros
    entropy = -(probs * torch.log(probs)).sum().item()
    entropy_dim = math.exp(entropy)
    
    return pr, entropy_dim


@dataclass
class FisherTracker:
    """
    Track Fisher Information properties over training.
    
    Computes and stores Fisher snapshots for analyzing
    information geometry dynamics.
    """
    
    _history: List[FisherSnapshot] = field(default_factory=list)
    
    def __post_init__(self):
        self._history = []
    
    def compute_and_record(
        self,
        model,
        data_loader,
        step: int,
        n_samples: int = 50,
        n_eigenvalues: int = 5,
    ) -> FisherSnapshot:
        """Compute Fisher snapshot and add to history."""
        
        # Compute spectral properties
        eigenvalues, trace, frobenius = compute_fisher_spectral_properties(
            model, data_loader, n_samples, n_eigenvalues
        )
        
        lambda_max = eigenvalues[0] if eigenvalues else 0.0
        
        # Compute diagonal Fisher for effective dim
        fisher_diag = compute_empirical_fisher_diagonal(model, data_loader, n_samples)
        pr, entropy_dim = compute_effective_dimensionality(fisher_diag)
        
        # Condition number estimate (rough, using trace and max)
        lambda_min_approx = trace / len(fisher_diag) if fisher_diag else None
        fisher_kappa = lambda_max / (lambda_min_approx + 1e-8) if lambda_min_approx else None
        
        snapshot = FisherSnapshot(
            step=step,
            timestamp=time.time(),
            n_samples=n_samples,
            top_eigenvalues=eigenvalues,
            lambda_max=lambda_max,
            lambda_min_approx=lambda_min_approx,
            fisher_kappa=fisher_kappa,
            trace=trace,
            frobenius_norm=frobenius,
            effective_dim=entropy_dim,
            participation_ratio=pr,
            natural_grad_alignment=None,  # Computed separately if needed
        )
        
        self._history.append(snapshot)
        return snapshot
    
    def get_trajectory(self, field: str = "fisher_kappa") -> List[Tuple[int, float]]:
        """Get trajectory of a specific field."""
        return [(h.step, getattr(h, field)) for h in self._history]
