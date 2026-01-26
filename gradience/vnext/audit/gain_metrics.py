"""
gradience.vnext.audit.gain_metrics

Low-rank LoRA gain/magnitude computation utilities.

This module provides efficient computation of LoRA update norms without materializing
the full update matrix ΔW = s · (B @ A).

Key idea: For ΔW = s · (B @ A) where A: (r × in), B: (out × r), s: scalar
- Frobenius norm: ||ΔW||_F^2 = s^2 · trace(B^T B @ A A^T)  
- Spectral norm: ||ΔW||_2^2 = s^2 · λ_max(sqrt(A A^T) @ B^T B @ sqrt(A A^T))

All computations operate in r×r space for efficiency, using float64 for numerical stability.

Configuration:
- The composition analysis can be disabled in bench configs with:
  audit:
    enable_composition_analysis: false
  
- Default is true (composition analysis enabled) for maximum insight into 
  energy concentration patterns across transformer depth.
"""

from typing import Tuple, Dict, List, Optional
import torch
import math
import re


def sqrt_psd(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute matrix square root of positive semi-definite matrix via eigendecomposition.
    
    Args:
        M: (n × n) positive semi-definite matrix
        eps: Minimum eigenvalue threshold (clamp negatives to 0)
        
    Returns:
        sqrt_M: Matrix square root such that sqrt_M @ sqrt_M ≈ M
    """
    # Use eigh for symmetric/Hermitian matrices (more stable than eig)
    eigenvals, eigenvecs = torch.linalg.eigh(M)
    
    # Clamp negative eigenvalues to 0 (numerical noise)
    eigenvals = torch.clamp(eigenvals, min=0.0)
    
    # sqrt_M = U @ diag(sqrt(λ)) @ U^T
    sqrt_eigenvals = torch.sqrt(eigenvals)
    sqrt_M = eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T
    
    return sqrt_M


def compute_lora_norms(
    A: torch.Tensor,
    B: torch.Tensor, 
    scaling: float = 1.0,
    *,
    compute_dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    eps: float = 1e-12
) -> Tuple[float, float]:
    """Compute LoRA update norms without materializing ΔW.
    
    For ΔW = scaling · (B @ A):
    - Computes ||ΔW||_F (Frobenius norm)
    - Computes ||ΔW||_2 (spectral/operator norm)
    
    Args:
        A: (r × d_in) LoRA factor A  
        B: (d_out × r) LoRA factor B
        scaling: Scalar multiplier (e.g., lora_alpha / r)
        compute_dtype: Dtype for internal computation (float64 recommended)
        device: Device for computation ("cpu" recommended)
        eps: Numerical stability threshold
        
    Returns:
        (frobenius_norm, spectral_norm): Both as float scalars
        
    Raises:
        ValueError: If A, B shapes are incompatible or not 2D
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors. Got A.shape={A.shape}, B.shape={B.shape}")
    
    r_A, d_in = A.shape
    d_out, r_B = B.shape
    
    if r_A != r_B:
        raise ValueError(f"Incompatible LoRA factors: A.shape[0]={r_A} != B.shape[1]={r_B}")
    
    r = r_A
    
    # Move to computation device/dtype
    A_compute = A.detach().to(dtype=compute_dtype, device=device)
    B_compute = B.detach().to(dtype=compute_dtype, device=device)
    
    # Compute Gram matrices (r × r)
    AAt = A_compute @ A_compute.T  # (r × r)
    BtB = B_compute.T @ B_compute  # (r × r)
    
    # Frobenius norm: ||ΔW||_F^2 = scaling^2 · trace(BtB @ AAt)
    # trace(AB) = sum(A * B) for element-wise multiplication
    frobenius_sq = torch.sum(BtB * AAt).item()
    frobenius_norm = abs(scaling) * math.sqrt(max(frobenius_sq, 0.0))
    
    # Spectral norm: ||ΔW||_2 = largest singular value of B @ A
    # Use existing optimized low-rank singular value computation
    from gradience.vnext.audit.lora_audit import low_rank_singular_values
    
    # Compute singular values without materializing ΔW
    singular_values = low_rank_singular_values(
        A_compute, B_compute, 
        compute_dtype=compute_dtype, 
        eps=eps
    )
    
    # Largest singular value (unscaled)
    max_singular_value = singular_values[0].item() if singular_values.numel() > 0 else 0.0
    spectral_norm = abs(scaling) * max_singular_value
    
    return frobenius_norm, spectral_norm


def compute_lora_stable_rank(
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float = 1.0,
    *,
    compute_dtype: torch.dtype = torch.float64,
    device: str = "cpu", 
    eps: float = 1e-12
) -> Tuple[float, float, float]:
    """Compute stable rank and utilization for LoRA update ΔW.
    
    Args:
        A, B, scaling: LoRA factors and scaling
        compute_dtype, device, eps: Computation parameters
        
    Returns:
        (stable_rank, utilization, rank): 
        - stable_rank: ||ΔW||_F^2 / ||ΔW||_2^2
        - utilization: stable_rank / r 
        - rank: LoRA rank r
    """
    frobenius_norm, spectral_norm = compute_lora_norms(
        A, B, scaling, 
        compute_dtype=compute_dtype, 
        device=device, 
        eps=eps
    )
    
    r = A.shape[0]
    
    # Stable rank = ||ΔW||_F^2 / ||ΔW||_2^2
    spectral_norm_sq = max(spectral_norm * spectral_norm, eps)
    stable_rank = (frobenius_norm * frobenius_norm) / spectral_norm_sq
    
    # Utilization = stable_rank / r
    utilization = stable_rank / max(float(r), 1.0)
    
    return stable_rank, utilization, float(r)


def extract_layer_index(module_name: str) -> Optional[int]:
    """Extract transformer layer index from module name.
    
    Handles common patterns:
    - model.layers.17.self_attn.q_proj -> 17
    - encoder.layer.11.attention.self.query -> 11
    - transformer.layer.5.attention.q_lin -> 5
    
    Args:
        module_name: Full module name (e.g., "model.layers.17.self_attn.q_proj")
        
    Returns:
        Layer index if found, None otherwise
    """
    # Common patterns for transformer layers
    patterns = [
        r'\.layers\.(\d+)\.',           # model.layers.N.
        r'\.layer\.(\d+)\.',            # encoder.layer.N. or transformer.layer.N.
        r'layer_(\d+)\.',               # layer_N.
        r'h\.(\d+)\.',                  # h.N. (GPT-style)
        r'decoder\.layers\.(\d+)\.',    # decoder.layers.N.
        r'encoder\.layer\.(\d+)\.',     # BERT encoder.layer.N.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, module_name)
        if match:
            return int(match.group(1))
    
    return None


def compute_layer_energy_concentration(
    module_energies: Dict[str, float],
    *,
    top_k: int = 5,
    eps: float = 1e-12
) -> Dict:
    """Compute energy concentration across transformer layers.
    
    Args:
        module_energies: Dict mapping module names to energy values (||ΔW||_F^2)
        top_k: Number of top layers to analyze (default: 5)
        eps: Numerical stability threshold
        
    Returns:
        Dict with concentration analysis:
        {
            "energy_total_fro2": float,
            "layers": [{"layer": int, "energy_fro2": float, "share": float}, ...],
            "top_k": {
                "k": int,
                "share": float, 
                "layers": [{"layer": int, "energy_fro2": float, "share": float}, ...]
            },
            "top_10pct": {
                "n": int,
                "share": float,
                "layers": [int, ...]
            }
        }
    """
    if not module_energies:
        return {
            "energy_total_fro2": 0.0,
            "layers": [],
            "top_k": {"k": top_k, "share": None, "layers": []},
            "top_10pct": {"n": 0, "share": None, "layers": []},
            "concentration_index": None
        }
    
    # Group modules by layer
    layer_energies = {}
    unknown_energy = 0.0
    
    for module_name, energy in module_energies.items():
        layer_idx = extract_layer_index(module_name)
        if layer_idx is not None:
            layer_energies[layer_idx] = layer_energies.get(layer_idx, 0.0) + energy
        else:
            unknown_energy += energy
    
    # Include unknown layer if significant
    if unknown_energy > eps:
        layer_energies[-1] = unknown_energy  # Use -1 for unknown layer
    
    total_energy = sum(layer_energies.values())
    
    if total_energy <= eps:
        # Zero energy case: set all shares to null for hygiene
        return {
            "energy_total_fro2": total_energy,
            "layers": [],
            "top_k": {"k": top_k, "share": None, "layers": []},
            "top_10pct": {"n": 0, "share": None, "layers": []},
            "concentration_index": None  # HHI undefined for zero energy
        }
    
    # Build layers list with shares
    layers_list = []
    for layer_idx in sorted(layer_energies.keys()):
        energy = layer_energies[layer_idx]
        share = energy / total_energy
        layers_list.append({
            "layer": layer_idx,
            "energy_fro2": energy,
            "share": share
        })
    
    # Sort by energy descending for top-k analysis
    sorted_layers = sorted(layers_list, key=lambda x: x["energy_fro2"], reverse=True)
    
    # Top-k analysis
    k = min(top_k, len(sorted_layers))
    top_k_layers = sorted_layers[:k]
    top_k_share = sum(layer["share"] for layer in top_k_layers)
    
    # Top-10% analysis
    n_layers = len(sorted_layers)
    top_10pct_count = max(1, int(0.1 * n_layers)) if n_layers > 0 else 0
    top_10pct_layers = sorted_layers[:top_10pct_count]
    top_10pct_share = sum(layer["share"] for layer in top_10pct_layers)
    top_10pct_indices = [layer["layer"] for layer in top_10pct_layers]
    
    # Concentration index (HHI/Simpson index): sum of squared shares
    # Higher values indicate more concentration
    concentration_index = sum(layer["share"] ** 2 for layer in layers_list)
    
    return {
        "energy_total_fro2": total_energy,
        "layers": layers_list,  # Sorted by layer index
        "top_k": {
            "k": k,
            "share": top_k_share,
            "layers": top_k_layers  # Sorted by energy descending
        },
        "top_10pct": {
            "n": top_10pct_count,
            "share": top_10pct_share,
            "layers": top_10pct_indices  # Sorted by energy descending
        },
        "concentration_index": concentration_index  # HHI: higher = more concentrated
    }


def compute_lora_energy_concentration(
    layer_energies: torch.Tensor,
    *,
    top_k_fraction: float = 0.1,
    eps: float = 1e-12
) -> Tuple[float, float]:
    """Compute energy concentration metrics for LoRA layers.
    
    Args:
        layer_energies: (n_layers,) tensor of per-layer energy values (e.g., ||ΔW||_F^2)
        top_k_fraction: Fraction of top layers to analyze (default: 0.1 = 10%)
        eps: Numerical stability threshold
        
    Returns:
        (top_k_share, top_10pct_share):
        - top_k_share: Fraction of total energy in top k layers
        - top_10pct_share: Fraction of total energy in top 10% of layers
    """
    if layer_energies.numel() == 0:
        return 0.0, 0.0
    
    total_energy = torch.sum(layer_energies).item()
    
    if total_energy <= eps:
        return 0.0, 0.0
    
    # Sort energies in descending order
    sorted_energies, _ = torch.sort(layer_energies, descending=True)
    
    n_layers = len(layer_energies)
    
    # Top k layers (custom fraction)
    k = max(1, int(top_k_fraction * n_layers))
    top_k_energy = torch.sum(sorted_energies[:k]).item()
    top_k_share = top_k_energy / total_energy
    
    # Top 10% layers  
    top_10pct_count = max(1, int(0.1 * n_layers))
    top_10pct_energy = torch.sum(sorted_energies[:top_10pct_count]).item()
    top_10pct_share = top_10pct_energy / total_energy
    
    return top_k_share, top_10pct_share


def verify_lora_factors_orientation(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Verify and correct LoRA factor orientation.
    
    Standard orientation: A (r × d_in), B (d_out × r)
    Common alternative: A (d_in × r), B (r × d_out)
    
    Args:
        A, B: LoRA factor tensors (any orientation)
        
    Returns:
        (A_oriented, B_oriented, r): Correctly oriented factors and rank
        
    Raises:
        ValueError: If factors cannot be oriented consistently
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"LoRA factors must be 2D. Got A.shape={A.shape}, B.shape={B.shape}")
    
    # Standard orientation: A (r × d_in), B (d_out × r) where A.shape[0] == B.shape[1] 
    if A.shape[0] == B.shape[1]:
        return A, B, int(A.shape[0])
    
    # Alternative orientation: A (d_in × r), B (r × d_out) where A.shape[1] == B.shape[0]
    if A.shape[1] == B.shape[0]:
        A_oriented = A.T.contiguous()  # (d_in × r) -> (r × d_in)
        B_oriented = B.T.contiguous()  # (r × d_out) -> (d_out × r)
        return A_oriented, B_oriented, int(A.shape[1])
    
    raise ValueError(
        f"Cannot orient LoRA factors consistently. "
        f"A.shape={A.shape}, B.shape={B.shape}. "
        f"Expected either A.shape[0]==B.shape[1] or A.shape[1]==B.shape[0]"
    )