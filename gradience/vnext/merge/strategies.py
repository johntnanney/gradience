"""
Merge strategy implementations for combining two LoRA ΔW matrices.

Provides pluggable strategies that take two dense delta-weight matrices and
produce a single merged delta-weight matrix.  Each strategy encodes a different
assumption about how task-specific updates interact:

- **LinearMerge**: weighted sum, appropriate when subspaces are orthogonal.
- **TIESMerge**: Trim → Elect Sign → Disjoint Mean (Yadav et al., 2023),
  appropriate when subspaces overlap and redundancy needs deduplication.

Usage::

    from gradience.vnext.merge.strategies import get_strategy, LayerMergeConfig

    strategy = get_strategy("linear")
    config = LayerMergeConfig(
        module_prefix="model.layers.0.self_attn.q_proj",
        strategy="linear",
        coefficients=(0.5, 0.5),
    )
    merged_dW = strategy.merge(dW_a, dW_b, config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from gradience.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Per-layer merge configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerMergeConfig:
    """Per-layer merge parameters from the plan.

    Attributes
    ----------
    module_prefix : canonical layer name (e.g. ``model.layers.0.self_attn.q_proj``)
    strategy : merge method name (``"linear"`` or ``"ties"``)
    coefficients : (coeff_a, coeff_b) weighting for the two adapters
    target_rank : target rank for the refactored LoRA output (informational,
        not used by the strategy itself but carried for the executor)
    trim_fraction : fraction of smallest-magnitude parameters to zero out
        before merging.  Only used by TIES.  0.0 = no trimming.
    """

    module_prefix: str
    strategy: str
    coefficients: tuple[float, float] = (0.5, 0.5)
    target_rank: int | None = None
    trim_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "module_prefix": self.module_prefix,
            "strategy": self.strategy,
            "coefficients": list(self.coefficients),
            "target_rank": self.target_rank,
            "trim_fraction": self.trim_fraction,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LayerMergeConfig:
        return cls(
            module_prefix=d["module_prefix"],
            strategy=d["strategy"],
            coefficients=tuple(d.get("coefficients", (0.5, 0.5))),
            target_rank=d.get("target_rank"),
            trim_fraction=d.get("trim_fraction", 0.0),
        )


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class MergeStrategy(ABC):
    """Base class for merge strategies.

    All strategies operate on dense ΔW matrices (d_out × d_in) and return a
    merged ΔW of the same shape.
    """

    @abstractmethod
    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        """Merge two delta-weight matrices.

        Parameters
        ----------
        dW_a : (d_out, d_in) — scaled ΔW from adapter A
        dW_b : (d_out, d_in) — scaled ΔW from adapter B
        config : per-layer merge configuration

        Returns
        -------
        merged_dW : (d_out, d_in)
        """
        ...


# ---------------------------------------------------------------------------
# Linear merge
# ---------------------------------------------------------------------------


class LinearMerge(MergeStrategy):
    """Weighted linear combination of two task vectors.

    ``dW_merged = coeff_a * dW_a + coeff_b * dW_b``

    Best when the two adapters' subspaces are approximately orthogonal,
    so their contributions don't interfere.
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        coeff_a, coeff_b = config.coefficients
        return coeff_a * dW_a + coeff_b * dW_b


# ---------------------------------------------------------------------------
# TIES merge
# ---------------------------------------------------------------------------


def _trim(task_vector: Tensor, fraction: float) -> Tensor:
    """Zero out the bottom ``fraction`` of entries by magnitude.

    Parameters
    ----------
    task_vector : flat or 2-D tensor
    fraction : in [0, 1]; 0.0 means no trimming

    Returns
    -------
    Trimmed tensor with the same shape, where the smallest-magnitude
    entries are zeroed out.
    """
    if fraction <= 0.0:
        return task_vector.clone()
    if fraction >= 1.0:
        return torch.zeros_like(task_vector)

    flat = task_vector.reshape(-1)
    n_trim = int(flat.numel() * fraction)
    if n_trim == 0:
        return task_vector.clone()

    # Find the magnitude threshold
    abs_vals = flat.abs()
    threshold = abs_vals.kthvalue(n_trim).values
    mask = abs_vals > threshold
    return (task_vector * mask.reshape(task_vector.shape)).clone()


def _elect_sign(task_vectors: list[Tensor]) -> Tensor:
    """Majority-vote sign election across task vectors.

    For each parameter position, the elected sign is the sign that appears
    most frequently (weighted by magnitude) across the task vectors.

    Parameters
    ----------
    task_vectors : list of tensors with the same shape

    Returns
    -------
    Sign tensor (+1 or -1) with the same shape, representing the
    majority-vote sign at each position.
    """
    # Stack and compute sign-weighted sum
    stacked = torch.stack(task_vectors, dim=0)  # (n_tasks, *shape)
    # Magnitude-weighted sign vote
    sign_sum = stacked.sign().sum(dim=0)  # (*shape)
    # Resolve ties toward positive
    elected = torch.where(sign_sum >= 0, torch.ones_like(sign_sum), -torch.ones_like(sign_sum))
    return elected


def _disjoint_mean(task_vectors: list[Tensor], elected_sign: Tensor) -> Tensor:
    """Average values that agree with the elected sign.

    For each parameter position, only task vectors whose value has the same
    sign as the elected sign contribute to the mean.  Positions where no
    task vector agrees default to zero.

    Parameters
    ----------
    task_vectors : list of tensors (same shape)
    elected_sign : sign tensor from ``_elect_sign``

    Returns
    -------
    Merged tensor with the same shape.
    """
    stacked = torch.stack(task_vectors, dim=0)  # (n_tasks, *shape)
    # Mask: only keep values that agree with elected sign
    agrees = stacked.sign() == elected_sign.unsqueeze(0)  # (n_tasks, *shape)
    # Zero out disagreeing values
    filtered = stacked * agrees.float()
    # Count agreeing values per position
    count = agrees.float().sum(dim=0).clamp(min=1)  # avoid div-by-zero
    return filtered.sum(dim=0) / count


class TIESMerge(MergeStrategy):
    """TIES: Trim → Elect Sign → Disjoint Mean (Yadav et al., 2023).

    Three-step merge that handles redundancy and sign conflicts:

    1. **Trim**: Zero out the bottom ``trim_fraction`` of each task vector
       by magnitude, keeping only the most influential parameters.
    2. **Elect Sign**: For each parameter position, determine the majority
       sign across (trimmed) task vectors.
    3. **Disjoint Mean**: Average only the values whose sign agrees with
       the elected sign.

    When ``trim_fraction=0.0``, this reduces to sign-elected disjoint mean
    (no magnitude trimming).
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        coeff_a, coeff_b = config.coefficients
        trim_fraction = config.trim_fraction

        # Scale by coefficients before TIES processing
        tv_a = coeff_a * dW_a
        tv_b = coeff_b * dW_b

        # Step 1: Trim
        tv_a_trimmed = _trim(tv_a, trim_fraction)
        tv_b_trimmed = _trim(tv_b, trim_fraction)

        # Step 2: Elect sign
        elected = _elect_sign([tv_a_trimmed, tv_b_trimmed])

        # Step 3: Disjoint mean
        merged = _disjoint_mean([tv_a_trimmed, tv_b_trimmed], elected)

        return merged


# ---------------------------------------------------------------------------
# DARE helpers
# ---------------------------------------------------------------------------


def _dare_dropout(task_vector: Tensor, drop_fraction: float) -> Tensor:
    """Randomly drop parameters and rescale survivors by 1/(1-p).

    Implements the DARE (Drop And REscale) sparsification from
    Yu et al., 2023 ("Language Models are Super Mario").

    Parameters
    ----------
    task_vector : tensor of any shape
    drop_fraction : probability of dropping each parameter (0 = no drop, 1 = all dropped)

    Returns
    -------
    Sparsified and rescaled tensor with same shape.
    """
    if drop_fraction <= 0.0:
        return task_vector.clone()
    if drop_fraction >= 1.0:
        return torch.zeros_like(task_vector)

    # Bernoulli mask: 1 = keep, 0 = drop
    mask = torch.bernoulli(torch.full_like(task_vector, 1.0 - drop_fraction))
    # Rescale kept values so E[output] = input
    rescale = 1.0 / (1.0 - drop_fraction)
    return task_vector * mask * rescale


# ---------------------------------------------------------------------------
# DARE-Linear merge
# ---------------------------------------------------------------------------


class DARELinearMerge(MergeStrategy):
    """DARE + Linear: random dropout with rescaling, then weighted average.

    For each task vector:
    1. Randomly drop parameters with probability ``trim_fraction``
    2. Rescale surviving parameters by ``1 / (1 - trim_fraction)``
    3. Weighted linear combination of the sparsified task vectors

    This reduces parameter interference while preserving the expected
    value of the merged output (Yu et al., 2023).

    When ``trim_fraction=0.0``, this is identical to ``LinearMerge``.
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        coeff_a, coeff_b = config.coefficients
        drop_fraction = config.trim_fraction

        # DARE sparsification
        sparse_a = _dare_dropout(dW_a, drop_fraction)
        sparse_b = _dare_dropout(dW_b, drop_fraction)

        # Linear combination
        return coeff_a * sparse_a + coeff_b * sparse_b


# ---------------------------------------------------------------------------
# DARE-TIES merge
# ---------------------------------------------------------------------------


class DARETIESMerge(MergeStrategy):
    """DARE + TIES: random dropout with rescaling, then TIES pipeline.

    For each task vector:
    1. DARE: randomly drop with probability ``trim_fraction``, rescale by ``1/(1-p)``
    2. TIES: elect majority sign across sparsified task vectors
    3. TIES: disjoint mean of values agreeing with elected sign

    Note: ``trim_fraction`` controls the DARE dropout probability.  The TIES
    magnitude trim step is *not* applied separately -- DARE sparsification
    replaces it.  This follows the DARE-TIES formulation from Yu et al. (2023).

    When ``trim_fraction=0.0``, this is identical to ``TIESMerge`` with no trim.
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        coeff_a, coeff_b = config.coefficients
        drop_fraction = config.trim_fraction

        # Scale by coefficients
        tv_a = coeff_a * dW_a
        tv_b = coeff_b * dW_b

        # DARE sparsification (replaces TIES magnitude trim)
        tv_a_sparse = _dare_dropout(tv_a, drop_fraction)
        tv_b_sparse = _dare_dropout(tv_b, drop_fraction)

        # TIES: elect sign + disjoint mean
        elected = _elect_sign([tv_a_sparse, tv_b_sparse])
        return _disjoint_mean([tv_a_sparse, tv_b_sparse], elected)


# ---------------------------------------------------------------------------
# Norm-equalized merge
# ---------------------------------------------------------------------------


class NormEqualizedMerge(MergeStrategy):
    """Scale both adapters to equal Frobenius norm before linear averaging.

    Removes scale imbalance as a confounding factor: both ΔW matrices are
    rescaled to their geometric-mean Frobenius norm, then combined with
    the configured coefficients.

    This is one of the strongest validated single interventions for merge
    quality.  It is simpler than the full audit-aware tree and often
    competitive, making it a valuable baseline.

    ``coefficients`` control the weighted average *after* norm equalization.
    ``trim_fraction`` is unused (ignored).
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        from gradience.vnext.merge.norm_equalized import norm_equalized_merge

        coeff_a, _ = config.coefficients
        result = norm_equalized_merge(dW_a, dW_b, weight_A=coeff_a)
        return result["merged"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type[MergeStrategy]] = {
    "linear": LinearMerge,
    "ties": TIESMerge,
    "dare_linear": DARELinearMerge,
    "dare_ties": DARETIESMerge,
    "norm_equalized": NormEqualizedMerge,
}


def get_strategy(name: str) -> MergeStrategy:
    """Create a merge strategy instance by name.

    Parameters
    ----------
    name : ``"linear"``, ``"ties"``, ``"dare_linear"``, ``"dare_ties"``,
        or ``"norm_equalized"``

    Returns
    -------
    MergeStrategy instance

    Raises
    ------
    ValueError
        If ``name`` is not a recognized strategy.
    """
    cls = _STRATEGIES.get(name)
    if cls is None:
        raise ConfigError(f"Unknown merge strategy '{name}'. Available: {sorted(_STRATEGIES.keys())}")
    return cls()
