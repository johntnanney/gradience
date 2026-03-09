"""
Fixture generators for merge compatibility audit tests.

Generates synthetic PEFT adapter directories with controlled spectral
properties (orthogonal, redundant, conflicting, imbalanced) for
deterministic testing of subspace metrics and verdict logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Adapter directory builder
# ---------------------------------------------------------------------------


def _make_adapter_dir(
    base_dir: Path,
    name: str,
    rank: int,
    alpha: float,
    target_modules: list[str],
    weights: dict[str, torch.Tensor],
    base_model: str = "test-model/tiny",
) -> Path:
    """Create a minimal PEFT adapter directory with config + weights."""
    adapter_dir = base_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    torch.save(weights, adapter_dir / "adapter_model.bin")
    return adapter_dir


def _make_lora_weights_from_svd(
    module_prefixes: list[str],
    U: torch.Tensor,
    S: torch.Tensor,
    Vt: torch.Tensor,
    rank: int,
    alpha: float,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Build LoRA A/B weight tensors from a desired SVD decomposition.

    Given ΔW = U @ diag(S) @ Vt, factors into B @ A such that
    scaling * B @ A ≈ U @ diag(S) @ Vt where scaling = alpha / rank.

    We distribute S evenly: B = U @ diag(sqrt(S)), A = diag(sqrt(S)) @ Vt,
    then divide by scaling to get the raw weights.
    """
    scaling = alpha / max(rank, 1)

    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank] * scale
    Vt_r = Vt[:rank, :]

    # Factor: ΔW = B_raw * scaling @ A_raw
    # ΔW = U_r @ diag(S_r) @ Vt_r
    # B_raw @ A_raw = (1/scaling) * U_r @ diag(S_r) @ Vt_r

    sqrt_S = torch.sqrt(S_r.clamp(min=0)).unsqueeze(1)  # (rank, 1)

    A = (sqrt_S * Vt_r) / (scaling**0.5)  # (rank, d_in)
    B = (U_r * sqrt_S.T) / (scaling**0.5)  # (d_out, rank)

    weights = {}
    for i, prefix in enumerate(module_prefixes):
        # Add small per-layer perturbation so layers are not identical.
        # This exercises per-layer variance paths in assess_overall().
        noise_scale = 1.0 + 0.1 * i
        weights[f"{prefix}.lora_A.weight"] = (A * noise_scale).float()
        weights[f"{prefix}.lora_B.weight"] = (B * noise_scale).float()

    return weights


# ---------------------------------------------------------------------------
# Standard dimensions
# ---------------------------------------------------------------------------

D_OUT = 64  # output dimension for test matrices
D_IN = 48  # input dimension for test matrices
RANK = 8  # default rank
ALPHA = 16.0  # default alpha

MODULE_PREFIXES = [
    "base_model.model.layers.0.self_attn.q_proj",
    "base_model.model.layers.0.self_attn.v_proj",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orthogonal_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two adapters with orthogonal dominant subspaces.

    Expected: verdict=SAFE, mean_overlap ≈ 0.0
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate a random orthogonal basis
    full_U, _, full_Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)

    # Adapter A uses first RANK singular directions
    U_a = full_U[:, :RANK]
    S_a = torch.linspace(5.0, 1.0, RANK)
    Vt_a = full_Vt[:RANK, :]

    # Adapter B uses next RANK singular directions (orthogonal to A)
    U_b = full_U[:, RANK : 2 * RANK]
    S_b = torch.linspace(4.0, 0.8, RANK)
    Vt_b = full_Vt[RANK : 2 * RANK, :]

    weights_a = _make_lora_weights_from_svd(MODULE_PREFIXES, U_a, S_a, Vt_a, RANK, ALPHA)
    weights_b = _make_lora_weights_from_svd(MODULE_PREFIXES, U_b, S_b, Vt_b, RANK, ALPHA)

    dir_a = _make_adapter_dir(tmp_path, "adapter_a", RANK, ALPHA, ["q_proj", "v_proj"], weights_a)
    dir_b = _make_adapter_dir(tmp_path, "adapter_b", RANK, ALPHA, ["q_proj", "v_proj"], weights_b)

    return dir_a, dir_b


@pytest.fixture
def redundant_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two adapters with identical subspaces, aligned directions.

    Expected: verdict=REDUNDANT, mean_overlap ≈ 1.0, agreement ≈ 1.0
    """
    torch.manual_seed(123)

    U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)
    U_r = U[:, :RANK]
    Vt_r = Vt[:RANK, :]

    S_a = torch.linspace(5.0, 1.0, RANK)
    S_b = torch.linspace(4.5, 0.9, RANK)  # Same directions, slightly different magnitudes

    weights_a = _make_lora_weights_from_svd(MODULE_PREFIXES, U_r, S_a, Vt_r, RANK, ALPHA)
    weights_b = _make_lora_weights_from_svd(MODULE_PREFIXES, U_r, S_b, Vt_r, RANK, ALPHA)

    dir_a = _make_adapter_dir(tmp_path, "adapter_a", RANK, ALPHA, ["q_proj", "v_proj"], weights_a)
    dir_b = _make_adapter_dir(tmp_path, "adapter_b", RANK, ALPHA, ["q_proj", "v_proj"], weights_b)

    return dir_a, dir_b


@pytest.fixture
def conflicting_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two adapters with overlapping subspaces, opposing directions.

    Expected: verdict=CONFLICTING, mean_overlap ≈ 1.0, agreement < -0.3
    """
    torch.manual_seed(456)

    U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)
    U_r = U[:, :RANK]
    Vt_r = Vt[:RANK, :]

    S_a = torch.linspace(5.0, 1.0, RANK)
    # Negate to flip direction (opposing contributions)
    S_b = torch.linspace(5.0, 1.0, RANK)

    weights_a = _make_lora_weights_from_svd(MODULE_PREFIXES, U_r, S_a, Vt_r, RANK, ALPHA)
    # For B, negate U to get opposing directions
    weights_b = _make_lora_weights_from_svd(MODULE_PREFIXES, -U_r, S_b, Vt_r, RANK, ALPHA)

    dir_a = _make_adapter_dir(tmp_path, "adapter_a", RANK, ALPHA, ["q_proj", "v_proj"], weights_a)
    dir_b = _make_adapter_dir(tmp_path, "adapter_b", RANK, ALPHA, ["q_proj", "v_proj"], weights_b)

    return dir_a, dir_b


@pytest.fixture
def imbalanced_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two adapters where one has ~10x the magnitude.

    Expected: verdict=IMBALANCED, magnitude_ratio ≈ 10.0
    """
    torch.manual_seed(789)

    U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)
    U_r = U[:, :RANK]
    Vt_r = Vt[:RANK, :]

    S_a = torch.linspace(50.0, 10.0, RANK)
    S_b = torch.linspace(5.0, 1.0, RANK)  # 10x smaller

    weights_a = _make_lora_weights_from_svd(MODULE_PREFIXES, U_r, S_a, Vt_r, RANK, ALPHA)
    weights_b = _make_lora_weights_from_svd(MODULE_PREFIXES, U_r, S_b, Vt_r, RANK, ALPHA)

    dir_a = _make_adapter_dir(tmp_path, "adapter_a", RANK, ALPHA, ["q_proj", "v_proj"], weights_a)
    dir_b = _make_adapter_dir(tmp_path, "adapter_b", RANK, ALPHA, ["q_proj", "v_proj"], weights_b)

    return dir_a, dir_b


@pytest.fixture
def different_rank_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two adapters with different ranks (r_a=4, r_b=16).

    Expected: analysis completes, metrics reflect rank asymmetry.
    """
    torch.manual_seed(101)

    rank_a = 4
    rank_b = 16

    U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)

    S_a = torch.linspace(5.0, 1.0, rank_a)
    S_b = torch.linspace(3.0, 0.5, rank_b)

    weights_a = _make_lora_weights_from_svd(MODULE_PREFIXES, U[:, :rank_a], S_a, Vt[:rank_a, :], rank_a, ALPHA)
    weights_b = _make_lora_weights_from_svd(MODULE_PREFIXES, U[:, :rank_b], S_b, Vt[:rank_b, :], rank_b, ALPHA)

    dir_a = _make_adapter_dir(tmp_path, "adapter_a", rank_a, ALPHA, ["q_proj", "v_proj"], weights_a)
    dir_b = _make_adapter_dir(tmp_path, "adapter_b", rank_b, ALPHA, ["q_proj", "v_proj"], weights_b)

    return dir_a, dir_b


@pytest.fixture
def partial_overlap_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two adapters with partially overlapping target modules.

    A targets: q_proj, v_proj
    B targets: v_proj, o_proj

    Expected: shared=[v_proj], only_a=[q_proj], only_b=[o_proj]
    """
    torch.manual_seed(202)

    U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)
    U_r = U[:, :RANK]
    Vt_r = Vt[:RANK, :]
    S = torch.linspace(5.0, 1.0, RANK)

    prefixes_a = [
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.0.self_attn.v_proj",
    ]
    prefixes_b = [
        "base_model.model.layers.0.self_attn.v_proj",
        "base_model.model.layers.0.self_attn.o_proj",
    ]

    weights_a = _make_lora_weights_from_svd(prefixes_a, U_r, S, Vt_r, RANK, ALPHA)
    weights_b = _make_lora_weights_from_svd(prefixes_b, U_r, S, Vt_r, RANK, ALPHA)

    dir_a = _make_adapter_dir(tmp_path, "adapter_a", RANK, ALPHA, ["q_proj", "v_proj"], weights_a)
    dir_b = _make_adapter_dir(tmp_path, "adapter_b", RANK, ALPHA, ["v_proj", "o_proj"], weights_b)

    return dir_a, dir_b
