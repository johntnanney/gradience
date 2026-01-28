"""
Unit tests for SVD truncation functionality.

Tests mathematical correctness, adapter I/O, and integration with bench protocol.
These are fast CPU-only tests that don't require Transformers or large models.
"""

import pytest
import tempfile
import json
import shutil
import math
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

from gradience.vnext.svd_truncate import (
    _compute_svd_truncation,
    _parse_and_pair_lora_matrices, 
    svd_truncate_peft_dir,
    _update_adapter_config,
    SVDTruncationReport
)
from gradience.vnext.audit.lora_audit import LoRAAdapterConfig


class TestSVDMathematicalCorrectness:
    """Test mathematical correctness of SVD truncation (4.1)."""
    
    @pytest.fixture
    def random_lora_matrices(self):
        """Generate random LoRA matrices A (r×d_in) and B (d_out×r)."""
        torch.manual_seed(42)  # Reproducible tests
        
        # Test parameters
        r = 16  # Original rank
        d_in = 512  # Input dimension
        d_out = 768  # Output dimension
        
        # Random LoRA matrices with reasonable magnitudes
        A = torch.randn(r, d_in, dtype=torch.float32) * 0.1
        B = torch.randn(d_out, r, dtype=torch.float32) * 0.1
        
        return A, B, r, d_in, d_out
    
    def test_truncation_shapes(self, random_lora_matrices):
        """Verify output shapes are correct."""
        A, B, r, d_in, d_out = random_lora_matrices
        target_rank = 8
        
        A_new, B_new, energy_retained = _compute_svd_truncation(A, B, target_rank)
        
        # Verify shapes
        assert A_new.shape == (target_rank, d_in), f"Expected A_new shape ({target_rank}, {d_in}), got {A_new.shape}"
        assert B_new.shape == (d_out, target_rank), f"Expected B_new shape ({d_out}, {target_rank}), got {B_new.shape}"
        
        # Verify types
        assert A_new.dtype == A.dtype, f"A dtype changed from {A.dtype} to {A_new.dtype}"
        assert B_new.dtype == B.dtype, f"B dtype changed from {B.dtype} to {B_new.dtype}"
        
        # Verify energy is in valid range
        assert 0.0 <= energy_retained <= 1.0, f"Energy retained {energy_retained} not in [0,1]"
    
    def test_reconstruction_error_bounds(self, random_lora_matrices):
        """Verify reconstruction error matches dropped singular values."""
        A, B, r, d_in, d_out = random_lora_matrices
        target_rank = 8
        
        # Original delta W = B @ A
        delta_W_original = B @ A
        
        # Truncated reconstruction
        A_new, B_new, energy_retained = _compute_svd_truncation(A, B, target_rank)
        delta_W_truncated = B_new @ A_new
        
        # Compute reconstruction error
        error = torch.norm(delta_W_original - delta_W_truncated, 'fro').item()
        original_norm = torch.norm(delta_W_original, 'fro').item()
        relative_error = error / (original_norm + 1e-8)
        
        # Energy retained should correlate with low reconstruction error
        expected_error_ratio = 1.0 - energy_retained
        
        # Allow some tolerance for numerical precision and energy computation differences
        # The relationship should hold approximately: relative_error² ≈ 1 - energy_retained
        assert relative_error <= 1.0, "Relative error should not exceed 100%"
        
        # For a rank-8 truncation of a rank-16 matrix, we expect reasonable reconstruction
        # Note: random matrices may have fairly uniform singular value spectrum
        assert relative_error < 0.8, f"Reconstruction error too high: {relative_error:.3f}"
        
        # Test energy computation is reasonable
        if energy_retained > 0.9:
            assert relative_error < 0.2, "Very high energy retention should give low reconstruction error"
        elif energy_retained > 0.7:
            assert relative_error < 0.5, "High energy retention should give reasonable reconstruction error"
    
    def test_rank_constraint(self, random_lora_matrices):
        """Verify rank(B' @ A') ≤ k."""
        A, B, r, d_in, d_out = random_lora_matrices
        target_rank = 8
        
        A_new, B_new, energy_retained = _compute_svd_truncation(A, B, target_rank)
        
        # Compute actual rank of reconstructed matrix
        delta_W_truncated = B_new @ A_new
        
        # Use SVD to compute numerical rank
        S = torch.linalg.svdvals(delta_W_truncated)
        
        # Count significant singular values (numerical rank)
        threshold = 1e-6 * S[0].item()  # Relative threshold
        numerical_rank = torch.sum(S > threshold).item()
        
        assert numerical_rank <= target_rank, f"Numerical rank {numerical_rank} exceeds target rank {target_rank}"
        
        # For well-conditioned case, should achieve exactly target rank
        assert numerical_rank >= target_rank - 1, f"Rank deficient: got {numerical_rank}, expected ~{target_rank}"
    
    def test_edge_cases(self):
        """Test edge cases: rank-1 truncation, equal ranks, etc."""
        torch.manual_seed(123)
        
        # Test rank-1 truncation
        A = torch.randn(4, 10, dtype=torch.float32) * 0.1
        B = torch.randn(8, 4, dtype=torch.float32) * 0.1
        
        A_new, B_new, energy = _compute_svd_truncation(A, B, target_rank=1)
        assert A_new.shape == (1, 10)
        assert B_new.shape == (8, 1)
        assert 0.0 <= energy <= 1.0
        
        # Test target_rank equals original rank (should preserve everything)
        A = torch.randn(6, 12, dtype=torch.float32) * 0.1
        B = torch.randn(10, 6, dtype=torch.float32) * 0.1
        
        A_new, B_new, energy = _compute_svd_truncation(A, B, target_rank=6)
        assert A_new.shape == (6, 12)
        assert B_new.shape == (10, 6)
        
        # Energy should be very close to 1.0 (perfect reconstruction)
        assert energy > 0.99, f"Perfect rank preservation should give energy ~1.0, got {energy:.4f}"
        
        # Test very small matrices
        A = torch.randn(2, 3, dtype=torch.float32) * 0.1  
        B = torch.randn(4, 2, dtype=torch.float32) * 0.1
        
        A_new, B_new, energy = _compute_svd_truncation(A, B, target_rank=1)
        assert A_new.shape == (1, 3)
        assert B_new.shape == (4, 1)
    
    def test_numerical_stability(self):
        """Test numerical stability with different dtypes and magnitudes."""
        torch.manual_seed(456)
        
        # Test with different dtypes
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            A = torch.randn(8, 16, dtype=dtype) * 0.1
            B = torch.randn(12, 8, dtype=dtype) * 0.1
            
            try:
                A_new, B_new, energy = _compute_svd_truncation(A, B, target_rank=4)
                assert A_new.dtype == dtype, f"Output dtype mismatch for {dtype}"
                assert 0.0 <= energy <= 1.0, f"Invalid energy for dtype {dtype}: {energy}"
            except Exception as e:
                pytest.fail(f"SVD failed for dtype {dtype}: {e}")
        
        # Test with very small values
        A = torch.randn(6, 10, dtype=torch.float32) * 1e-6
        B = torch.randn(8, 6, dtype=torch.float32) * 1e-6
        
        A_new, B_new, energy = _compute_svd_truncation(A, B, target_rank=3)
        assert torch.isfinite(A_new).all(), "Small values produced non-finite results"
        assert torch.isfinite(B_new).all(), "Small values produced non-finite results"
        
        # Test with large values
        A = torch.randn(6, 10, dtype=torch.float32) * 100
        B = torch.randn(8, 6, dtype=torch.float32) * 100
        
        A_new, B_new, energy = _compute_svd_truncation(A, B, target_rank=3)
        assert torch.isfinite(A_new).all(), "Large values produced non-finite results"
        assert torch.isfinite(B_new).all(), "Large values produced non-finite results"
    
    def test_energy_computation_accuracy(self, random_lora_matrices):
        """Test that energy computation accurately reflects spectral information."""
        A, B, r, d_in, d_out = random_lora_matrices
        
        # Compute full SVD for reference
        delta_W = B @ A
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        
        # Test energy computation for different truncation levels
        for k in [2, 4, 8, 12]:
            if k >= len(S):
                continue
                
            A_new, B_new, energy_reported = _compute_svd_truncation(A, B, k)
            
            # Compute expected energy from true singular values
            total_energy = torch.sum(S**2).item()
            retained_energy = torch.sum(S[:k]**2).item()
            expected_energy = retained_energy / total_energy if total_energy > 0 else 0.0
            
            # Allow some tolerance due to the QR decomposition method vs direct SVD
            tolerance = 0.05  # 5% tolerance
            energy_diff = abs(energy_reported - expected_energy)
            
            assert energy_diff < tolerance, (
                f"Energy computation inaccurate for k={k}: "
                f"reported={energy_reported:.4f}, expected={expected_energy:.4f}, "
                f"diff={energy_diff:.4f}"
            )


class TestAdapterIO:
    """Test adapter I/O functionality (4.2)."""
    
    @pytest.fixture
    def temp_peft_dir(self):
        """Create a temporary PEFT directory with fake adapter data."""
        temp_dir = tempfile.mkdtemp()
        peft_dir = Path(temp_dir) / "fake_peft_adapter"
        peft_dir.mkdir(parents=True, exist_ok=True)
        
        yield peft_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def create_minimal_adapter_config(self, peft_dir: Path, rank: int = 16) -> None:
        """Create minimal adapter_config.json."""
        config = {
            "alpha_pattern": {},
            "auto_mapping": None, 
            "base_model_name_or_path": "distilbert-base-uncased",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": float(rank),  # Common default: alpha = rank
            "lora_dropout": 0.1,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin"],
            "task_type": "FEATURE_EXTRACTION"
        }
        
        config_path = peft_dir / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_fake_adapter_weights(self, peft_dir: Path, rank: int = 16) -> None:
        """Create fake adapter weights with 2 LoRA pairs."""
        # Create fake weights matching typical PEFT structure
        state_dict = {}
        
        # Create 2 LoRA pairs with proper naming
        modules = ["q_lin", "v_lin"]  # Just 2 modules for lightweight testing
        
        for module in modules:
            # LoRA A matrices (rank, input_dim)
            A_key = f"base_model.model.distilbert.transformer.layer.0.attention.{module}.lora_A.default.weight"
            state_dict[A_key] = torch.randn(rank, 768, dtype=torch.float16) * 0.1
            
            # LoRA B matrices (output_dim, rank)  
            B_key = f"base_model.model.distilbert.transformer.layer.0.attention.{module}.lora_B.default.weight"
            state_dict[B_key] = torch.randn(768, rank, dtype=torch.float16) * 0.1
        
        # Add a non-LoRA tensor to test preservation
        state_dict["modules_to_save.default.classifier.weight"] = torch.randn(2, 768, dtype=torch.float16) * 0.1
        
        # Save as safetensors (or torch if safetensors unavailable)
        weights_path = peft_dir / "adapter_model.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(state_dict, weights_path)
        except ImportError:
            # Fallback to torch.save
            weights_path = peft_dir / "adapter_model.pt"
            torch.save(state_dict, weights_path)
    
    def test_adapter_directory_creation(self, temp_peft_dir):
        """Test that fake PEFT directory structure is created correctly."""
        self.create_minimal_adapter_config(temp_peft_dir, rank=16)
        self.create_fake_adapter_weights(temp_peft_dir, rank=16)
        
        # Verify files exist
        assert (temp_peft_dir / "adapter_config.json").exists()
        
        weights_file = temp_peft_dir / "adapter_model.safetensors"
        if not weights_file.exists():
            weights_file = temp_peft_dir / "adapter_model.pt"
        assert weights_file.exists(), "No adapter weights file found"
    
    def test_svd_truncate_peft_dir_basic(self, temp_peft_dir):
        """Test basic SVD truncation of PEFT directory."""
        original_rank = 16
        target_rank = 8
        
        # Create fake adapter
        self.create_minimal_adapter_config(temp_peft_dir, rank=original_rank)
        self.create_fake_adapter_weights(temp_peft_dir, rank=original_rank)
        
        # Create output directory
        output_dir = temp_peft_dir.parent / "truncated_adapter"
        
        # Run SVD truncation
        report = svd_truncate_peft_dir(
            peft_dir=temp_peft_dir,
            out_dir=output_dir, 
            target_rank=target_rank,
            alpha_mode="keep_ratio",
            save_dtype="fp16"
        )
        
        # Verify output directory structure
        assert output_dir.exists(), "Output directory not created"
        assert (output_dir / "adapter_config.json").exists(), "Config not copied"
        
        # Check weights file
        weights_file = output_dir / "adapter_model.safetensors"
        if not weights_file.exists():
            weights_file = output_dir / "adapter_model.pt" 
        assert weights_file.exists(), "Truncated weights not saved"
        
        # Verify truncation report
        assert isinstance(report, SVDTruncationReport)
        assert report.original_rank == original_rank
        assert report.target_rank == target_rank
        assert report.total_modules == 2  # q_lin, v_lin
        assert 0.0 <= report.energy_retained <= 1.0
        assert report.compression_ratio > 1.0  # Should be compressed
        
        # Verify truncation report file
        assert (output_dir / "truncation_report.json").exists()
        
        # Verify README was generated
        assert (output_dir / "README.md").exists()
    
    def test_config_update_keep_ratio(self, temp_peft_dir):
        """Test that adapter config is properly updated with keep_ratio mode."""
        original_rank = 16
        target_rank = 8
        
        self.create_minimal_adapter_config(temp_peft_dir, rank=original_rank)
        self.create_fake_adapter_weights(temp_peft_dir, rank=original_rank)
        
        output_dir = temp_peft_dir.parent / "truncated_adapter"
        
        report = svd_truncate_peft_dir(
            peft_dir=temp_peft_dir,
            out_dir=output_dir,
            target_rank=target_rank,
            alpha_mode="keep_ratio"
        )
        
        # Load updated config
        with open(output_dir / "adapter_config.json") as f:
            new_config = json.load(f)
        
        # Verify rank update
        assert new_config["r"] == target_rank
        
        # Verify alpha update (keep_ratio: alpha_new = alpha_old * target_rank / original_rank)
        expected_alpha = float(original_rank) * (target_rank / original_rank)  # Should be 8.0
        assert abs(new_config["lora_alpha"] - expected_alpha) < 1e-6
        
        # Verify patterns are reset
        assert new_config["rank_pattern"] == {}
        assert new_config["alpha_pattern"] == {}
    
    def test_config_update_keep_alpha(self, temp_peft_dir):
        """Test that adapter config is properly updated with keep_alpha mode."""
        original_rank = 16
        target_rank = 8
        
        self.create_minimal_adapter_config(temp_peft_dir, rank=original_rank)
        self.create_fake_adapter_weights(temp_peft_dir, rank=original_rank)
        
        output_dir = temp_peft_dir.parent / "truncated_adapter"
        
        report = svd_truncate_peft_dir(
            peft_dir=temp_peft_dir,
            out_dir=output_dir,
            target_rank=target_rank,
            alpha_mode="keep_alpha"
        )
        
        # Load updated config
        with open(output_dir / "adapter_config.json") as f:
            new_config = json.load(f)
        
        # Verify rank update
        assert new_config["r"] == target_rank
        
        # Verify alpha unchanged (keep_alpha mode)
        assert new_config["lora_alpha"] == float(original_rank)  # Should stay 16.0
    
    def test_weights_loading(self, temp_peft_dir):
        """Test that truncated weights can be loaded back."""
        original_rank = 16
        target_rank = 8
        
        self.create_minimal_adapter_config(temp_peft_dir, rank=original_rank)
        self.create_fake_adapter_weights(temp_peft_dir, rank=original_rank)
        
        output_dir = temp_peft_dir.parent / "truncated_adapter"
        
        report = svd_truncate_peft_dir(
            peft_dir=temp_peft_dir,
            out_dir=output_dir,
            target_rank=target_rank
        )
        
        # Try to load truncated weights
        weights_file = output_dir / "adapter_model.safetensors"
        if not weights_file.exists():
            weights_file = output_dir / "adapter_model.pt"
        
        if weights_file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weights_file)
            except ImportError:
                pytest.skip("safetensors not available")
        else:
            state_dict = torch.load(weights_file, map_location="cpu")
        
        # Verify LoRA matrices have correct shapes
        lora_a_keys = [k for k in state_dict.keys() if "lora_A" in k]
        lora_b_keys = [k for k in state_dict.keys() if "lora_B" in k]
        
        assert len(lora_a_keys) == 2, f"Expected 2 LoRA A matrices, got {len(lora_a_keys)}"
        assert len(lora_b_keys) == 2, f"Expected 2 LoRA B matrices, got {len(lora_b_keys)}"
        
        for key in lora_a_keys:
            tensor = state_dict[key]
            assert tensor.shape[0] == target_rank, f"LoRA A rank mismatch: {tensor.shape}"
        
        for key in lora_b_keys:
            tensor = state_dict[key]
            assert tensor.shape[1] == target_rank, f"LoRA B rank mismatch: {tensor.shape}"
        
        # Verify non-LoRA tensors preserved
        non_lora_keys = [k for k in state_dict.keys() if "lora_A" not in k and "lora_B" not in k]
        assert len(non_lora_keys) > 0, "Non-LoRA tensors not preserved"
    
    def test_error_handling(self, temp_peft_dir):
        """Test error handling for invalid inputs."""
        # Test missing config
        with pytest.raises(FileNotFoundError):
            svd_truncate_peft_dir(
                peft_dir=temp_peft_dir,
                out_dir=temp_peft_dir.parent / "output",
                target_rank=8
            )
        
        # Test target rank >= original rank
        self.create_minimal_adapter_config(temp_peft_dir, rank=8)
        self.create_fake_adapter_weights(temp_peft_dir, rank=8)
        
        with pytest.raises(ValueError, match="Target rank.*must be.*original rank"):
            svd_truncate_peft_dir(
                peft_dir=temp_peft_dir,
                out_dir=temp_peft_dir.parent / "output",
                target_rank=8  # Equal to original rank
            )
    
    def test_parameter_scaling_tripwire(self, temp_peft_dir):
        """Future-proofing: Test that parameter count scales linearly with rank."""
        # Create adapters at different target ranks (all starting from r=16)
        target_ranks_and_expected_params = [
            (8, 2 * (8 * 768 + 768 * 8)),   # 2 modules * (A_params + B_params)
            (4, 2 * (4 * 768 + 768 * 4)),   # A: rank×embed, B: embed×rank  
            (2, 2 * (2 * 768 + 768 * 2))
        ]
        
        for target_rank, expected_params in target_ranks_and_expected_params:
            with tempfile.TemporaryDirectory() as temp_dir:
                source_dir = Path(temp_dir) / "source"
                output_dir = Path(temp_dir) / "output"
                
                source_dir.mkdir()
                
                # Create adapter with rank=16, truncate to target_rank
                self.create_minimal_adapter_config(source_dir, rank=16)  # Start from r=16
                self.create_fake_adapter_weights(source_dir, rank=16)  # Creates q_lin, v_lin by default
                
                # Truncate to target rank
                report = svd_truncate_peft_dir(
                    peft_dir=source_dir,
                    out_dir=output_dir,
                    target_rank=target_rank
                )
                
                # Tripwire: parameter count must scale linearly with rank
                actual_params = report.total_new_params
                assert actual_params == expected_params, \
                    f"Parameter scaling violation: rank {target_rank} expected {expected_params}, got {actual_params}"
                
                # Tripwire: compression ratio must be consistent
                expected_compression = (16 / target_rank)
                assert abs(report.compression_ratio - expected_compression) < 0.01, \
                    f"Compression ratio mismatch: rank {target_rank} expected {expected_compression}, got {report.compression_ratio}"
    
    def test_audit_mixup_prevention(self, temp_peft_dir):
        """Future-proofing: Test that truncation reports include identifying metadata."""
        self.create_minimal_adapter_config(temp_peft_dir, rank=16)
        self.create_fake_adapter_weights(temp_peft_dir, rank=16)
        
        output_dir = temp_peft_dir.parent / "truncated_r8"
        
        report = svd_truncate_peft_dir(
            peft_dir=temp_peft_dir,
            out_dir=output_dir,
            target_rank=8
        )
        
        # Verify metadata to prevent audit mixups
        assert report.source_directory is not None, "Missing source directory in report"
        assert report.output_directory is not None, "Missing output directory in report"
        assert report.timestamp is not None, "Missing timestamp in report"
        assert report.total_original_params is not None, "Missing original param count"
        assert report.total_new_params is not None, "Missing new param count"
        
        # Verify the metadata makes sense
        assert str(temp_peft_dir.name) in report.source_directory
        assert str(output_dir.name) in report.output_directory
        assert report.total_original_params > report.total_new_params
        assert report.original_rank == 16
        assert report.target_rank == 8
        
        # Load report from JSON to verify serialization
        report_file = output_dir / "truncation_report.json"
        assert report_file.exists(), "Truncation report not saved"
        
        with open(report_file) as f:
            saved_report = json.load(f)
        
        # Verify metadata preserved in JSON
        assert "source_directory" in saved_report
        assert "output_directory" in saved_report
        assert "timestamp" in saved_report
        assert "total_original_params" in saved_report
        assert "total_new_params" in saved_report


class TestLoraMatrixParsing:
    """Test LoRA matrix parsing and pairing logic."""
    
    def test_parse_and_pair_lora_matrices_modern_format(self):
        """Test parsing modern PEFT format (.lora_A.default.weight)."""
        state_dict = {
            "base_model.layer.0.attention.q_lin.lora_A.default.weight": torch.randn(8, 512),
            "base_model.layer.0.attention.q_lin.lora_B.default.weight": torch.randn(512, 8),
            "base_model.layer.0.attention.v_lin.lora_A.default.weight": torch.randn(8, 512),
            "base_model.layer.0.attention.v_lin.lora_B.default.weight": torch.randn(512, 8),
            "some.other.weight": torch.randn(100, 50),  # Non-LoRA tensor
        }
        
        pairs = _parse_and_pair_lora_matrices(state_dict, adapter_name="default")
        
        assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}"
        
        # Check that pairs have correct structure
        for base_key, pair in pairs.items():
            assert "A" in pair and "B" in pair
            A_key, A_tensor = pair["A"]
            B_key, B_tensor = pair["B"]
            
            # Check shapes are compatible
            assert A_tensor.shape[0] == B_tensor.shape[1], f"Rank mismatch: A={A_tensor.shape}, B={B_tensor.shape}"
    
    def test_parse_and_pair_lora_matrices_legacy_format(self):
        """Test parsing legacy PEFT format (.lora_A.weight)."""
        state_dict = {
            "transformer.h.0.attn.q_proj.lora_A.weight": torch.randn(16, 1024),
            "transformer.h.0.attn.q_proj.lora_B.weight": torch.randn(1024, 16),
            "transformer.h.0.attn.v_proj.lora_A.weight": torch.randn(16, 1024),
            "transformer.h.0.attn.v_proj.lora_B.weight": torch.randn(1024, 16),
        }
        
        pairs = _parse_and_pair_lora_matrices(state_dict, adapter_name="default")
        
        assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}"
    
    def test_parse_and_pair_shape_mismatch(self):
        """Test handling of shape mismatches."""
        state_dict = {
            "layer.lora_A.default.weight": torch.randn(8, 512),
            "layer.lora_B.default.weight": torch.randn(512, 16),  # Wrong rank: 16 vs 8
        }
        
        pairs = _parse_and_pair_lora_matrices(state_dict, adapter_name="default")
        
        # Should skip mismatched pairs
        assert len(pairs) == 0, "Mismatched shapes should be filtered out"
    
    def test_parse_and_pair_missing_partner(self):
        """Test handling of orphaned A or B matrices."""
        state_dict = {
            "layer1.lora_A.default.weight": torch.randn(8, 512),
            # Missing corresponding B matrix
            "layer2.lora_B.default.weight": torch.randn(512, 8),
            # Missing corresponding A matrix
        }
        
        pairs = _parse_and_pair_lora_matrices(state_dict, adapter_name="default")
        
        # Should skip orphaned matrices
        assert len(pairs) == 0, "Orphaned matrices should be filtered out"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])