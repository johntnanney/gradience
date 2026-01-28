"""
gradience.vnext.audit.rank_policies

Pure rank selection policies for LoRA compression.

Small, testable module that converts singular values → rank suggestions.
No dependencies on torch models, PEFT directories, or JSON I/O.
Just math.

Core API:
- RankPolicySpec: Policy name + parameters  
- RankSuggestion: Suggested rank + metadata
- apply_rank_policy(): Pure function singular_values → suggestion

Policies:
- energy_threshold: k@X% energy capture (original Gradience approach)
- entropy_effective: exp(entropy) of normalized singular value distribution
- optimal_hard_threshold: OHT from random matrix theory  
- knee_elbow: Elbow detection in singular value scree plot
- stable_rank_ceil: Conservative stable rank ceiling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Data Structures  
# =============================================================================

@dataclass(frozen=True)
class RankPolicySpec:
    """Specification for a rank selection policy."""
    name: str
    params: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate policy name."""
        valid_policies = {
            'energy_threshold', 'entropy_effective', 'optimal_hard_threshold',
            'knee_elbow', 'stable_rank_ceil'
        }
        if self.name not in valid_policies:
            available = ', '.join(sorted(valid_policies))
            raise ValueError(f"Unknown policy '{self.name}'. Available: {available}")


@dataclass(frozen=True)
class RankSuggestion:
    """Result from applying a rank selection policy."""
    k: int                              # Suggested rank
    confidence: float                   # 0.0 to 1.0 confidence score
    details: Dict[str, Union[float, int, str]]  # Policy-specific metadata
    
    def __post_init__(self):
        """Validate suggestion."""
        if self.k < 0:
            raise ValueError(f"Suggested rank must be non-negative, got {self.k}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")


# =============================================================================
# Core API
# =============================================================================

def apply_rank_policy(
    policy_spec: RankPolicySpec,
    s: np.ndarray,
    shape: Tuple[int, int],
    r_alloc: int,
    eps: float = 1e-12
) -> RankSuggestion:
    """
    Apply rank selection policy to singular values.
    
    Args:
        policy_spec: Policy name and parameters
        s: Singular values in descending order, shape (r,) where r <= r_alloc
        shape: Shape (out_dim, in_dim) of effective ΔW = B @ A  
        r_alloc: Allocated LoRA rank (usually len(s), but provided for validation)
        eps: Numerical stability threshold
        
    Returns:
        RankSuggestion with suggested rank k and metadata
        
    Note: This is a pure function with no side effects or external dependencies.
    """
    # Input validation
    if not isinstance(s, np.ndarray) or s.ndim != 1:
        raise ValueError(f"Expected 1D numpy array for singular values, got shape {s.shape}")
    if len(s) > r_alloc:
        raise ValueError(f"len(s)={len(s)} > r_alloc={r_alloc}")
    if not all(s[i] >= s[i+1] for i in range(len(s)-1)):
        raise ValueError("Singular values must be in descending order")
    
    # Route to specific policy implementation
    policy_name = policy_spec.name
    params = policy_spec.params
    
    if policy_name == 'energy_threshold':
        return _energy_threshold_policy(s, shape, r_alloc, params, eps)
    elif policy_name == 'entropy_effective':
        return _entropy_effective_policy(s, shape, r_alloc, params, eps)
    elif policy_name == 'optimal_hard_threshold':
        return _optimal_hard_threshold_policy(s, shape, r_alloc, params, eps)
    elif policy_name == 'knee_elbow':
        return _knee_elbow_policy(s, shape, r_alloc, params, eps)
    elif policy_name == 'stable_rank_ceil':
        return _stable_rank_ceil_policy(s, shape, r_alloc, params, eps)
    else:
        # Should not reach here due to RankPolicySpec validation
        raise ValueError(f"Unhandled policy: {policy_name}")


# =============================================================================
# Policy Implementations (Pure Math)
# =============================================================================

def _energy_threshold_policy(
    s: np.ndarray, 
    shape: Tuple[int, int],
    r_alloc: int,
    params: Dict[str, Union[float, int, str]],
    eps: float
) -> RankSuggestion:
    """Energy@X% threshold policy (original Gradience approach)."""
    threshold = float(params.get('threshold', 0.90))
    
    if len(s) == 0 or np.sum(s > eps) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'threshold': threshold, 'reason': 'no_significant_singular_values'}
        )
    
    # Filter out negligible singular values
    s_clean = s[s > eps]
    
    # Compute cumulative energy
    energy = s_clean ** 2
    total_energy = np.sum(energy)
    
    if total_energy <= eps:
        return RankSuggestion(
            k=0,
            confidence=0.0, 
            details={'threshold': threshold, 'total_energy': total_energy}
        )
    
    cumulative = np.cumsum(energy) / total_energy
    
    # Find first index where cumulative energy >= threshold
    indices = np.where(cumulative >= threshold)[0]
    if len(indices) > 0:
        k = int(indices[0] + 1)  # +1 because index is 0-based
    else:
        k = len(s_clean)  # Use all non-negligible singular values
    
    # Confidence = actual energy captured
    actual_energy = cumulative[k-1] if k > 0 else 0.0
    confidence = min(actual_energy, 1.0)
    
    return RankSuggestion(
        k=k,
        confidence=confidence,
        details={
            'threshold': threshold,
            'actual_energy_captured': float(actual_energy),
            'total_energy': float(total_energy),
            'max_singular_value': float(s_clean[0]),
            'num_significant_sv': len(s_clean)
        }
    )


def _entropy_effective_policy(
    s: np.ndarray,
    shape: Tuple[int, int], 
    r_alloc: int,
    params: Dict[str, Union[float, int, str]],
    eps: float
) -> RankSuggestion:
    """Entropy effective rank (eRank) policy.
    
    Implements Roy & Vetterli's effective rank via spectral entropy:
    - Normalize: p_i = σ_i / Σ_j σ_j (canonical normalization)
    - Shannon entropy: H(p) = -Σ p_k log(p_k) (ignore p=0 terms)
    - Effective rank: erank = exp(H)
    - Convert with rounding rule: k = ceil(erank) (slightly conservative by default)
    """
    
    if len(s) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'reason': 'empty_singular_values'}
        )
    
    s_clean = s[s > eps]
    if len(s_clean) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'reason': 'no_significant_singular_values'}
        )
    
    # Roy & Vetterli canonical normalization: p_i = σ_i / Σ_j σ_j
    total_sv = np.sum(s_clean)
    p = s_clean / total_sv
    
    # Shannon entropy H(p) = -Σ p_k log(p_k), ignoring p=0 terms
    # GUARDRAIL: Explicitly handle numerical precision issues with p ≈ 0
    p_safe = p[p > eps]  # Extra safety: remove near-zero probabilities
    if len(p_safe) == 0:
        return RankSuggestion(
            k=1,
            confidence=0.0,
            details={
                'reason': 'numerical_precision_issue_in_entropy',
                'note': 'All normalized probabilities too close to zero'
            }
        )
    
    log_p = np.log(p_safe)  # Safe since p_safe > eps > 0
    entropy = -np.sum(p_safe * log_p)
    
    # Effective rank: erank = exp(H)
    erank_float = float(np.exp(entropy))
    
    # Convert to integer rank suggestion with rounding rule
    rounding_mode = params.get('rounding', 'ceil')  # Default: ceil (conservative)
    if rounding_mode == 'ceil':
        k_suggested = math.ceil(erank_float)
    elif rounding_mode == 'round':
        k_suggested = round(erank_float)
    elif rounding_mode == 'floor':
        k_suggested = math.floor(erank_float)
    else:
        k_suggested = math.ceil(erank_float)  # Default to ceil
    
    # Clamp to valid range [1, r_alloc]
    k = max(1, min(k_suggested, r_alloc))
    
    # Record multiple rounding options for analysis
    k_ceil = math.ceil(erank_float)
    k_round = round(erank_float)
    k_floor = math.floor(erank_float)
    
    # Confidence: high when effective rank is well-defined (not near uniform)
    # Use concentration: entropy closer to 0 = high confidence, entropy closer to max = low confidence
    max_possible_entropy = math.log(len(p_safe))
    normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0
    confidence = max(0.0, min(1.0, 1.0 - normalized_entropy))  # Double clamp for numerical safety
    
    return RankSuggestion(
        k=k,
        confidence=confidence,
        details={
            'erank_float': erank_float,
            'k_ceil': k_ceil,
            'k_round': k_round, 
            'k_floor': k_floor,
            'entropy': float(entropy),
            'max_possible_entropy': max_possible_entropy,
            'normalized_entropy': float(normalized_entropy),
            'rounding_mode': rounding_mode,
            'num_significant_sv': len(s_clean)
        }
    )


def _optimal_hard_threshold_policy(
    s: np.ndarray,
    shape: Tuple[int, int],
    r_alloc: int, 
    params: Dict[str, Union[float, int, str]],
    eps: float
) -> RankSuggestion:
    """OHT (Gavish-Donoho optimal hard threshold) policy.
    
    Implements Gavish & Donoho's optimal singular value threshold adapted for LoRA:
    - Compute β = min(out,in)/max(out,in) (aspect ratio)
    - ω(β) ≈ 0.56β³ - 0.95β² + 1.82β + 1.43 (cubic approximation)
    - τ = ω(β) × median(s)
    - k = count(s_i > τ) clamped to [1, r_alloc]
    
    EXPERIMENTAL: Based on random matrix theory; adapted from full-rank to LoRA use case.
    """
    
    if len(s) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'reason': 'empty_singular_values'}
        )
    
    # GUARDRAIL: OHT needs sufficient data for meaningful threshold computation
    if r_alloc < 3:
        return RankSuggestion(
            k=1,
            confidence=0.0,
            details={
                'reason': 'insufficient_rank_for_oht',
                'r_alloc': r_alloc,
                'fallback_note': 'OHT requires r_alloc >= 3; consider using energy or knee policies'
            }
        )
    
    # Convert to float32 for stability as recommended
    s_f32 = s.astype(np.float32)
    s_clean = s_f32[s_f32 > eps]
    
    if len(s_clean) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'reason': 'no_significant_singular_values'}
        )
    
    # Compute aspect ratio β = min(out,in)/max(out,in) ≤ 1
    out_dim, in_dim = shape
    beta = float(min(out_dim, in_dim)) / float(max(out_dim, in_dim))
    
    # Gavish-Donoho cubic approximation for ω(β)
    # ω(β) ≈ 0.56β³ - 0.95β² + 1.82β + 1.43
    omega = 0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * beta + 1.43
    
    # Compute median singular value
    median_sv = float(np.median(s_clean))
    
    # Handle degenerate case: median = 0
    if median_sv <= eps:
        return RankSuggestion(
            k=1,  # Return k=1 as suggested (could be k=0 for "drop layer" semantics)
            confidence=0.0,
            details={
                'reason': 'median_sv_zero',
                'beta': beta,
                'omega': omega,
                'tau': 0.0,
                'median_sv': median_sv
            }
        )
    
    # Compute threshold τ = ω(β) × median(s)
    tau = omega * median_sv
    
    # Count singular values above threshold
    k_raw = int(np.sum(s_clean > tau))
    
    # Clamp to [1, r_alloc] as specified
    k = max(1, min(k_raw, r_alloc))
    
    # Confidence: high when threshold is clearly separating signal from noise
    # Use ratio of largest SV above threshold to threshold
    if k_raw > 0:
        max_above_threshold = float(np.max(s_clean[s_clean > tau]))
        confidence = min(1.0, max_above_threshold / (tau + eps))
    else:
        confidence = 0.0
    
    return RankSuggestion(
        k=k,
        confidence=confidence,
        details={
            'beta': beta,
            'omega': omega, 
            'tau': tau,
            'median_sv': median_sv,
            'k_raw': k_raw,
            'num_above_threshold': k_raw,
            'num_significant_sv': len(s_clean)
        }
    )


def _knee_elbow_policy(
    s: np.ndarray,
    shape: Tuple[int, int],
    r_alloc: int,
    params: Dict[str, Union[float, int, str]], 
    eps: float
) -> RankSuggestion:
    """Knee detection (Kneedle-style elbow) policy.
    
    Implements Kneedle's core idea using cumulative energy curve:
    - Compute energy weights: e_i = σ_i²
    - Cumulative energy fraction: y[i] = Σ_{j≤i} e_j / Σ_{all} e_j  
    - Normalized indices: x[i] = i/(r-1) for i=0..r-1
    - Difference curve: diff[i] = y[i] - x[i] (curve vs straight line)
    - Knee: k = argmax(diff) + 1
    - Guardrails: smooth with moving average, clamp k to [1, r_alloc]
    """
    
    if len(s) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'reason': 'empty_singular_values'}
        )
    
    s_clean = s[s > eps]
    if len(s_clean) < 2:
        return RankSuggestion(
            k=len(s_clean) if len(s_clean) > 0 else 1,
            confidence=0.0,
            details={'reason': 'insufficient_data', 'num_significant_sv': len(s_clean)}
        )
    
    r = len(s_clean)
    
    # Compute energy weights e_i = σ_i²
    energy = s_clean ** 2
    total_energy = np.sum(energy)
    
    if total_energy <= eps:
        return RankSuggestion(
            k=1,
            confidence=0.0,
            details={'reason': 'negligible_total_energy'}
        )
    
    # Cumulative energy fraction y[i] = Σ_{j≤i} e_j / Σ_{all} e_j
    cumulative_energy = np.cumsum(energy) / total_energy
    y = cumulative_energy
    
    # Normalized indices x[i] = i/(r-1) for i=0..r-1  
    # Both x and y live in [0,1]
    if r == 1:
        x = np.array([0.0])
    else:
        x = np.arange(r, dtype=np.float32) / (r - 1)
    
    # Apply tiny moving average smoothing (window 3-5) as guardrail
    window_size = min(5, max(3, r // 3))  # Adaptive window size
    if r >= window_size:
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        # Use 'same' mode to keep same length, pad edges appropriately
        y_smooth = np.convolve(y, kernel, mode='same')
    else:
        y_smooth = y  # No smoothing for very short arrays
    
    # Compute difference curve: diff[i] = y[i] - x[i] 
    # This is Kneedle's "(x, y - x)" trick
    diff_curve = y_smooth - x
    
    # Find knee: k = argmax(diff) + 1
    knee_idx = int(np.argmax(diff_curve))
    knee_diff_max = float(diff_curve[knee_idx])
    
    # Convert to rank (1-indexed)
    k_raw = knee_idx + 1
    
    # GUARDRAIL: Detect flat spectrum using "last ~10-15%" rule
    # If knee lands in the last ~15% of ranks, treat as "no knee detected"
    flat_threshold = params.get('flat_threshold', 0.1)  # Configurable threshold
    last_15_percent_cutoff = max(1, int(0.85 * r))  # Last ~15% of spectrum
    
    if knee_idx >= last_15_percent_cutoff or knee_diff_max < flat_threshold:
        # Spectrum appears flat or knee too close to end → don't compress
        k_suggested = r_alloc
        flat_spectrum = True
        flat_reason = 'knee_in_tail' if knee_idx >= last_15_percent_cutoff else 'weak_knee_signal'
    else:
        k_suggested = k_raw 
        flat_spectrum = False
        flat_reason = None
    
    # Clamp k to [1, r_alloc] as specified
    k = max(1, min(k_suggested, r_alloc))
    
    # Confidence: high when knee is pronounced and not at edges
    if flat_spectrum:
        confidence = 0.0  # No confidence in flat spectrum
    else:
        # High confidence when diff_max is large and knee not at edges
        edge_penalty = 1.0
        if knee_idx <= 1 or knee_idx >= r - 2:
            edge_penalty = 0.5  # Reduce confidence for edge knees
        
        # Scale confidence by how pronounced the knee is
        max_possible_diff = 1.0  # Theoretical max of y[i] - x[i]
        confidence = min(1.0, (knee_diff_max / max_possible_diff) * edge_penalty)
    
    return RankSuggestion(
        k=k,
        confidence=confidence,
        details={
            'knee_diff_max': knee_diff_max,
            'knee_index': knee_idx,
            'k_raw': k_raw,
            'flat_spectrum': flat_spectrum,
            'flat_reason': flat_reason,
            'last_15_percent_cutoff': last_15_percent_cutoff,
            'edge_penalty': edge_penalty if not flat_spectrum else 0.0,
            'window_size': window_size,
            'num_significant_sv': len(s_clean),
            'total_energy': float(total_energy)
        }
    )


def _stable_rank_ceil_policy(
    s: np.ndarray,
    shape: Tuple[int, int],
    r_alloc: int,
    params: Dict[str, Union[float, int, str]],
    eps: float
) -> RankSuggestion:
    """Conservative stable rank ceiling policy."""
    
    if len(s) == 0 or np.sum(s > eps) == 0:
        return RankSuggestion(
            k=0,
            confidence=0.0,
            details={'reason': 'no_significant_singular_values'}
        )
    
    s_clean = s[s > eps]
    
    # Stable rank = ||s||_F^2 / ||s||_2^2
    frob_sq = np.sum(s_clean ** 2)
    max_sv_sq = s_clean[0] ** 2
    stable_rank = frob_sq / (max_sv_sq + eps)
    
    # Conservative: ceiling of stable rank, capped at actual rank
    k = min(max(1, math.ceil(stable_rank)), len(s_clean))
    
    # Confidence: higher when stable rank is close to an integer
    fractional_part = abs(stable_rank - round(stable_rank))
    confidence = 1.0 - fractional_part  # More confident when stable_rank ≈ integer
    
    return RankSuggestion(
        k=k,
        confidence=confidence,
        details={
            'stable_rank': float(stable_rank),
            'frobenius_norm_sq': float(frob_sq),
            'max_singular_value': float(s_clean[0]),
            'fractional_part': fractional_part,
            'num_significant_sv': len(s_clean)
        }
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_energy_policy(threshold: float = 0.90) -> RankPolicySpec:
    """Create energy@X% threshold policy."""
    return RankPolicySpec('energy_threshold', {'threshold': threshold})


def create_entropy_policy(rounding: str = 'ceil') -> RankPolicySpec:
    """Create entropy effective rank policy.
    
    Args:
        rounding: Rounding mode for erank→k conversion ('ceil', 'round', 'floor')
                 'ceil' is slightly conservative (default)
    """
    return RankPolicySpec('entropy_effective', {'rounding': rounding})


def create_oht_policy() -> RankPolicySpec:
    """Create optimal hard threshold (Gavish-Donoho) policy.
    
    Uses the exact Gavish-Donoho cubic approximation for ω(β).
    No parameters needed - computes β from matrix shape and uses median(s).
    """
    return RankPolicySpec('optimal_hard_threshold')


def create_knee_policy(flat_threshold: float = 0.1) -> RankPolicySpec:
    """Create knee/elbow detection (Kneedle-style) policy.
    
    Args:
        flat_threshold: Threshold for detecting flat spectra (default 0.1)
                       Lower values = more aggressive flat detection
    """
    return RankPolicySpec('knee_elbow', {'flat_threshold': flat_threshold})


def create_stable_ceil_policy() -> RankPolicySpec:
    """Create stable rank ceiling policy."""
    return RankPolicySpec('stable_rank_ceil')


def get_standard_policies() -> List[RankPolicySpec]:
    """Get list of standard rank selection policies for comparison."""
    return [
        create_energy_policy(0.90),
        create_energy_policy(0.95), 
        create_entropy_policy(),
        create_oht_policy(),
        create_knee_policy(),
        create_stable_ceil_policy(),
    ]


# =============================================================================
# Multi-Policy Analysis
# =============================================================================

@dataclass(frozen=True)
class PolicyConsensus:
    """Consensus analysis from multiple rank policies."""
    suggestions: List[RankSuggestion]
    median_k: int
    k_range: Tuple[int, int] 
    high_confidence_policies: List[str]  # Policies with confidence > 0.8
    disagreement_score: float  # Standard deviation / mean of suggested ranks


def analyze_policy_consensus(
    policies: List[RankPolicySpec],
    s: np.ndarray,
    shape: Tuple[int, int],
    r_alloc: int,
    eps: float = 1e-12
) -> PolicyConsensus:
    """
    Apply multiple policies and analyze consensus.
    
    Args:
        policies: List of rank policy specifications
        s: Singular values in descending order
        shape: Shape (out_dim, in_dim) of effective ΔW 
        r_alloc: Allocated LoRA rank
        eps: Numerical stability threshold
        
    Returns:
        PolicyConsensus with aggregate analysis
    """
    suggestions = []
    
    for policy_spec in policies:
        try:
            suggestion = apply_rank_policy(policy_spec, s, shape, r_alloc, eps)
            suggestions.append(suggestion)
        except Exception:
            # Skip failed policies in consensus (could log this)
            continue
    
    if not suggestions:
        return PolicyConsensus(
            suggestions=[],
            median_k=0,
            k_range=(0, 0),
            high_confidence_policies=[],
            disagreement_score=float('inf')
        )
    
    # Compute consensus metrics
    ranks = [sugg.k for sugg in suggestions]
    median_k = int(np.median(ranks))
    k_range = (min(ranks), max(ranks))
    
    # High confidence policies (threshold could be parameterized)
    high_conf = [policy_spec.name for policy_spec, sugg in zip(policies, suggestions) 
                 if sugg.confidence > 0.8]
    
    # Disagreement: coefficient of variation
    if len(ranks) > 1 and np.mean(ranks) > 0:
        disagreement = np.std(ranks) / np.mean(ranks)
    else:
        disagreement = 0.0
    
    return PolicyConsensus(
        suggestions=suggestions,
        median_k=median_k,
        k_range=k_range,
        high_confidence_policies=high_conf,
        disagreement_score=disagreement
    )