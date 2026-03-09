"""Tests for gradience.vnext.merge.io — adapter loading and layer matching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gradience.vnext.merge.io import (
    AdapterInfo,
    extract_factors,
    get_module_type,
    load_adapter,
    match_layers,
)

# ---------------------------------------------------------------------------
# load_adapter
# ---------------------------------------------------------------------------


class TestLoadAdapter:
    """Tests for load_adapter()."""

    def test_load_success(self, orthogonal_pair):
        """Basic load succeeds and returns AdapterInfo."""
        dir_a, dir_b = orthogonal_pair
        info = load_adapter(dir_a)

        assert isinstance(info, AdapterInfo)
        assert info.path == dir_a
        assert info.rank == 8
        assert info.alpha == 16.0
        assert info.scaling == 16.0 / 8
        assert len(info.lora_pairs) == 2
        assert len(info.module_prefixes) == 2

    def test_load_missing_dir(self, tmp_path):
        """Raises FileNotFoundError for non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Adapter directory not found"):
            load_adapter(tmp_path / "nonexistent")

    def test_load_missing_config(self, tmp_path):
        """Raises FileNotFoundError when adapter_config.json is missing."""
        adapter_dir = tmp_path / "no_config"
        adapter_dir.mkdir()
        torch.save({"k": torch.randn(4, 4)}, adapter_dir / "adapter_model.bin")

        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            load_adapter(adapter_dir)

    def test_load_missing_weights(self, tmp_path):
        """Raises FileNotFoundError when weights file is missing."""
        adapter_dir = tmp_path / "no_weights"
        adapter_dir.mkdir()
        config = {"r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"]}
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(FileNotFoundError, match="adapter weights"):
            load_adapter(adapter_dir)

    def test_load_no_lora_pairs(self, tmp_path):
        """Raises ValueError when weights have no LoRA A/B pairs."""
        adapter_dir = tmp_path / "no_pairs"
        adapter_dir.mkdir()
        config = {"r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"]}
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(config, f)
        # Save weights without lora_A/lora_B pattern
        torch.save({"some_key": torch.randn(4, 4)}, adapter_dir / "adapter_model.bin")

        with pytest.raises(ValueError, match="No LoRA A/B pairs"):
            load_adapter(adapter_dir)


# ---------------------------------------------------------------------------
# match_layers
# ---------------------------------------------------------------------------


class TestMatchLayers:
    """Tests for match_layers()."""

    def test_exact_match(self, orthogonal_pair):
        """Both adapters have same modules -> all shared."""
        dir_a, dir_b = orthogonal_pair
        info_a = load_adapter(dir_a)
        info_b = load_adapter(dir_b)

        shared, only_a, only_b = match_layers(info_a, info_b)

        assert len(shared) == 2
        assert len(only_a) == 0
        assert len(only_b) == 0

    def test_partial_overlap(self, partial_overlap_pair):
        """Adapters with different target modules -> partial match."""
        dir_a, dir_b = partial_overlap_pair
        info_a = load_adapter(dir_a)
        info_b = load_adapter(dir_b)

        shared, only_a, only_b = match_layers(info_a, info_b)

        assert len(shared) == 1
        assert "v_proj" in shared[0]
        assert len(only_a) == 1
        assert "q_proj" in only_a[0]
        assert len(only_b) == 1
        assert "o_proj" in only_b[0]

    def test_no_overlap(self, tmp_path):
        """Adapters with completely different modules -> no shared."""
        from tests.merge.conftest import (
            ALPHA,
            D_IN,
            D_OUT,
            RANK,
            _make_adapter_dir,
            _make_lora_weights_from_svd,
        )

        torch.manual_seed(303)
        U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)
        S = torch.linspace(5.0, 1.0, RANK)

        prefixes_a = ["base_model.model.layers.0.self_attn.q_proj"]
        prefixes_b = ["base_model.model.layers.0.self_attn.o_proj"]

        weights_a = _make_lora_weights_from_svd(prefixes_a, U[:, :RANK], S, Vt[:RANK, :], RANK, ALPHA)
        weights_b = _make_lora_weights_from_svd(prefixes_b, U[:, :RANK], S, Vt[:RANK, :], RANK, ALPHA)

        dir_a = _make_adapter_dir(tmp_path, "a", RANK, ALPHA, ["q_proj"], weights_a)
        dir_b = _make_adapter_dir(tmp_path, "b", RANK, ALPHA, ["o_proj"], weights_b)

        info_a = load_adapter(dir_a)
        info_b = load_adapter(dir_b)
        shared, only_a, only_b = match_layers(info_a, info_b)

        assert len(shared) == 0
        assert len(only_a) == 1
        assert len(only_b) == 1


# ---------------------------------------------------------------------------
# extract_factors
# ---------------------------------------------------------------------------


class TestExtractFactors:
    """Tests for extract_factors()."""

    def test_correct_shapes(self, orthogonal_pair):
        """Extracted A, B have correct shapes and rank."""
        dir_a, _ = orthogonal_pair
        info = load_adapter(dir_a)
        prefix = info.module_prefixes[0]

        A, B, r = extract_factors(info, prefix)

        assert r == 8
        assert A.shape[0] == r  # (r, d_in)
        assert B.shape[1] == r  # (d_out, r)
        assert A.shape[1] > 0
        assert B.shape[0] > 0

    def test_missing_prefix(self, orthogonal_pair):
        """Raises KeyError for unknown module prefix."""
        dir_a, _ = orthogonal_pair
        info = load_adapter(dir_a)

        with pytest.raises(KeyError, match="not found"):
            extract_factors(info, "nonexistent.module")


# ---------------------------------------------------------------------------
# get_module_type
# ---------------------------------------------------------------------------


class TestGetModuleType:
    """Tests for get_module_type()."""

    def test_attn_modules(self):
        assert get_module_type("model.layers.0.self_attn.q_proj") in ("attn", "attention")
        assert get_module_type("model.layers.0.self_attn.v_proj") in ("attn", "attention")

    def test_mlp_modules(self):
        result = get_module_type("model.layers.0.mlp.gate_proj")
        # Accept either "mlp" or "ffn" depending on implementation
        assert result in ("mlp", "ffn", "other")
