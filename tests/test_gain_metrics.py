"""
Test suite for gain_metrics module.

This test suite verifies the mathematical correctness of efficient LoRA norm computation
without materializing the full ΔW matrix. It compares the r×r method against explicit
ΔW computation to ensure accuracy within numerical tolerance.

"These tests are the reason you'll sleep at night." - User wisdom
"""

import pytest
import torch
import math
import numpy as np
from typing import Tuple, Dict

from gradience.vnext.audit.gain_metrics import (
    compute_lora_norms,
    compute_lora_stable_rank,
    extract_layer_index,
    compute_layer_energy_concentration,
    sqrt_psd,
    verify_lora_factors_orientation
)


class TestLoRANormComputation:
    """Test the core r×r norm computation against explicit ΔW materialization."""
    
    @pytest.fixture
    def random_seed(self):
        """Fixed seed for reproducible tests."""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def generate_random_lora_factors(
        self, 
        out_dim: int, 
        in_dim: int, 
        rank: int, 
        scaling: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Generate random LoRA factors A, B and scaling factor."""
        # A: (r × in_dim), B: (out_dim × r)
        A = torch.randn(rank, in_dim, dtype=torch.float64)
        B = torch.randn(out_dim, rank, dtype=torch.float64)
        return A, B, scaling
    
    def compute_explicit_delta_w_norms(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        scaling: float
    ) -> Tuple[float, float]:
        """Compute norms by explicitly materializing ΔW = scaling * (B @ A)."""
        delta_W = scaling * (B @ A)
        
        # Frobenius norm
        fro_norm = torch.linalg.norm(delta_W, ord='fro').item()
        
        # Spectral norm (largest singular value)
        singular_values = torch.linalg.svdvals(delta_W)
        spectral_norm = singular_values[0].item() if singular_values.numel() > 0 else 0.0
        
        return fro_norm, spectral_norm
    
    def test_small_random_matrices_accuracy(self, random_seed):
        """Test accuracy against explicit computation for small random matrices."""
        test_cases = [
            (13, 11, 4, 1.0),     # User's suggested dimensions
            (8, 12, 3, 2.5),      # Different scaling
            (20, 15, 6, 0.5),     # Larger matrices
            (5, 7, 2, 1.0),       # Small matrices
        ]
        
        tolerance = 1e-10  # Very strict tolerance for float64
        
        for out_dim, in_dim, rank, scaling in test_cases:
            # Test each case  
            A, B, _ = self.generate_random_lora_factors(out_dim, in_dim, rank, scaling)
            
            # Compute using our efficient r×r method
            fro_efficient, spectral_efficient = compute_lora_norms(
                A, B, scaling, 
                compute_dtype=torch.float64, 
                device="cpu"
            )
            
            # Compute using explicit ΔW materialization
            fro_explicit, spectral_explicit = self.compute_explicit_delta_w_norms(A, B, scaling)
            
            # Verify Frobenius norm accuracy
            fro_error = abs(fro_efficient - fro_explicit)
            assert fro_error < tolerance, (
                f"Frobenius norm error {fro_error:.2e} exceeds tolerance {tolerance:.2e} "
                f"for dims ({out_dim}, {in_dim}, {rank}), scaling={scaling}. "
                f"Efficient: {fro_efficient:.10f}, Explicit: {fro_explicit:.10f}"
            )
            
            # Verify spectral norm accuracy
            spectral_error = abs(spectral_efficient - spectral_explicit)
            assert spectral_error < tolerance, (
                f"Spectral norm error {spectral_error:.2e} exceeds tolerance {tolerance:.2e} "
                f"for dims ({out_dim}, {in_dim}, {rank}), scaling={scaling}. "
                f"Efficient: {spectral_efficient:.10f}, Explicit: {spectral_explicit:.10f}"
            )
    
    def test_scaling_correctness(self, random_seed):
        """Test that scaling behaves correctly: if s doubles, norms should double."""
        A, B, _ = self.generate_random_lora_factors(10, 8, 3, 1.0)
        
        scaling_1 = 1.0
        scaling_2 = 2.0
        
        fro_1, spectral_1 = compute_lora_norms(A, B, scaling_1)
        fro_2, spectral_2 = compute_lora_norms(A, B, scaling_2)
        
        # Norms should scale linearly with scaling factor
        assert abs(fro_2 - 2 * fro_1) < 1e-10, f"Frobenius norm scaling incorrect: {fro_2} vs 2*{fro_1}"
        assert abs(spectral_2 - 2 * spectral_1) < 1e-10, f"Spectral norm scaling incorrect: {spectral_2} vs 2*{spectral_1}"
        
        # Test negative scaling (should take absolute value)
        scaling_neg = -1.5
        fro_neg, spectral_neg = compute_lora_norms(A, B, scaling_neg)
        fro_pos, spectral_pos = compute_lora_norms(A, B, 1.5)
        
        assert abs(fro_neg - fro_pos) < 1e-10, "Negative scaling should use absolute value"
        assert abs(spectral_neg - spectral_pos) < 1e-10, "Negative scaling should use absolute value"
    
    def test_near_zero_eigenvalues_stability(self, random_seed):
        """Test numerical stability when eigenvalues are near zero."""
        # Create rank-deficient case: A has some very small singular values
        rank = 4
        A = torch.randn(rank, 10, dtype=torch.float64)
        B = torch.randn(12, rank, dtype=torch.float64)
        
        # Make A nearly rank-deficient by zeroing out some components
        A[2:, :] *= 1e-12  # Very small but not exactly zero
        
        # Should not crash and should produce reasonable results
        fro_norm, spectral_norm = compute_lora_norms(A, B, 1.0, eps=1e-15)
        
        assert fro_norm >= 0, "Frobenius norm should be non-negative"
        assert spectral_norm >= 0, "Spectral norm should be non-negative"
        assert math.isfinite(fro_norm), "Frobenius norm should be finite"
        assert math.isfinite(spectral_norm), "Spectral norm should be finite"
        
        # Verify against explicit computation
        fro_explicit, spectral_explicit = self.compute_explicit_delta_w_norms(A, B, 1.0)
        assert abs(fro_norm - fro_explicit) < 1e-8, "Should match explicit computation even with small eigenvalues"
        assert abs(spectral_norm - spectral_explicit) < 1e-8, "Should match explicit computation even with small eigenvalues"
    
    def test_zero_matrices(self):
        """Test behavior with zero matrices."""
        A = torch.zeros(3, 5, dtype=torch.float64)
        B = torch.zeros(7, 3, dtype=torch.float64)
        
        fro_norm, spectral_norm = compute_lora_norms(A, B, 1.0)
        
        assert fro_norm == 0.0, "Zero matrices should produce zero norms"
        assert spectral_norm == 0.0, "Zero matrices should produce zero norms"
    
    def test_matrix_orientation_handling(self, random_seed):
        """Test that factor orientation is handled correctly."""
        # Standard orientation: A (r × d_in), B (d_out × r)
        A_std = torch.randn(4, 10, dtype=torch.float64)
        B_std = torch.randn(12, 4, dtype=torch.float64)
        
        # Alternative orientation: A (d_in × r), B (r × d_out) 
        A_alt = A_std.T.contiguous()  # (10 × 4)
        B_alt = B_std.T.contiguous()  # (4 × 12)
        
        # verify_lora_factors_orientation should fix alternative orientation
        A_fixed, B_fixed, r = verify_lora_factors_orientation(A_alt, B_alt)
        
        assert A_fixed.shape == A_std.shape, "Fixed A should match standard orientation"
        assert B_fixed.shape == B_std.shape, "Fixed B should match standard orientation"
        assert r == 4, "Rank should be extracted correctly"
        
        # Norms should be same regardless of input orientation
        fro_std, spectral_std = compute_lora_norms(A_std, B_std, 1.0)
        fro_alt, spectral_alt = compute_lora_norms(A_fixed, B_fixed, 1.0)
        
        assert abs(fro_std - fro_alt) < 1e-10, "Norms should be orientation-invariant"
        assert abs(spectral_std - spectral_alt) < 1e-10, "Norms should be orientation-invariant"


class TestStableRankComputation:
    """Test stable rank computation and utilization metrics."""
    
    def test_stable_rank_properties(self):
        """Test mathematical properties of stable rank."""
        # Create test matrices
        A = torch.randn(3, 8, dtype=torch.float64)
        B = torch.randn(10, 3, dtype=torch.float64) 
        scaling = 1.5
        
        stable_rank, utilization, rank = compute_lora_stable_rank(A, B, scaling)
        
        # Basic properties
        assert stable_rank > 0, "Stable rank should be positive"
        assert utilization > 0, "Utilization should be positive"
        assert utilization <= 1.0, "Utilization should be ≤ 1 (stable_rank ≤ r)"
        assert rank == 3, "Rank should match A.shape[0]"
        
        # Stable rank should be ≤ actual rank
        assert stable_rank <= rank, f"Stable rank {stable_rank} should be ≤ rank {rank}"
        
        # Utilization = stable_rank / rank
        expected_utilization = stable_rank / rank
        assert abs(utilization - expected_utilization) < 1e-10, "Utilization should equal stable_rank / rank"
    
    def test_stable_rank_extreme_cases(self):
        """Test stable rank for extreme concentration cases."""
        # Case 1: Perfect rank-1 matrix (maximum concentration)
        # Create B @ A that is exactly rank 1
        u = torch.randn(8, 1, dtype=torch.float64)
        v = torch.randn(1, 6, dtype=torch.float64)
        target_delta = u @ v  # Rank-1 matrix
        
        # Decompose back into LoRA factors
        U, S, Vh = torch.linalg.svd(target_delta, full_matrices=False)
        r = 2  # Use rank 2 even though target is rank 1
        A = torch.zeros(r, 6, dtype=torch.float64)
        B = torch.zeros(8, r, dtype=torch.float64)
        
        # Put rank-1 structure in first component
        A[0, :] = math.sqrt(S[0]) * Vh[0, :]
        B[:, 0] = math.sqrt(S[0]) * U[:, 0]
        
        stable_rank, utilization, rank = compute_lora_stable_rank(A, B, 1.0)
        
        # For rank-1 case, stable rank should be close to 1
        assert stable_rank < 1.1, f"Rank-1 case should have stable_rank ≈ 1, got {stable_rank}"
        assert utilization < 0.6, f"Rank-1 case should have low utilization, got {utilization}"


class TestLayerExtraction:
    """Test layer index extraction from module names."""
    
    def test_common_transformer_patterns(self):
        """Test layer extraction for common transformer architectures."""
        test_cases = [
            # Standard patterns
            ("model.layers.17.self_attn.q_proj", 17),
            ("base_model.model.distilbert.transformer.layer.5.attention.out_lin", 5),
            ("encoder.layer.11.attention.self.query", 11),
            ("transformer.layer.3.attention.q_lin", 3),
            ("h.12.attn.c_proj", 12),
            ("decoder.layers.23.self_attn.v_proj", 23),
            
            # Edge cases
            ("no_layer_here", None),
            ("layers.not.a.number", None),
            ("something.layer.42.different.format", 42),  # This pattern should match
            ("", None),
            ("just_a_module_name", None)
        ]
        
        for module_name, expected_layer in test_cases:
            actual_layer = extract_layer_index(module_name)
            assert actual_layer == expected_layer, (
                f"extract_layer_index('{module_name}') = {actual_layer}, expected {expected_layer}"
            )
    
    def test_layer_extraction_robustness(self):
        """Test that layer extraction handles edge cases gracefully."""
        # Should not crash on malformed inputs
        weird_inputs = [None, 123, [], {}, "layer..5", "layer.abc.def"]
        
        for weird_input in weird_inputs:
            try:
                result = extract_layer_index(str(weird_input))
                assert result is None or isinstance(result, int), "Should return None or int"
            except Exception as e:
                pytest.fail(f"extract_layer_index should not crash on {weird_input}, got {e}")


class TestEnergyConcentrationAnalysis:
    """Test composition-style energy concentration analysis."""
    
    def test_uniform_energy_distribution(self):
        """Test analysis with uniform energy distribution."""
        # 3 layers, equal energy
        module_energies = {
            "model.layers.0.attn.q": 1.0,
            "model.layers.0.attn.k": 1.0,  # Layer 0: 2.0 total
            "model.layers.1.attn.q": 1.0,
            "model.layers.1.attn.k": 1.0,  # Layer 1: 2.0 total
            "model.layers.2.attn.q": 1.0,
            "model.layers.2.attn.k": 1.0,  # Layer 2: 2.0 total
        }
        
        result = compute_layer_energy_concentration(module_energies, top_k=2)
        
        # Should have uniform distribution
        assert result["energy_total_fro2"] == 6.0
        assert len(result["layers"]) == 3
        
        # Each layer should have 1/3 share
        for layer in result["layers"]:
            assert abs(layer["share"] - 1/3) < 1e-10, f"Layer {layer['layer']} should have 1/3 share"
            assert layer["energy_fro2"] == 2.0, f"Layer {layer['layer']} should have energy 2.0"
        
        # Top-k analysis
        assert result["top_k"]["k"] == 2
        assert abs(result["top_k"]["share"] - 2/3) < 1e-10, "Top-2 should have 2/3 of energy"
        
        # Top-10% analysis (should be top-1 layer = 30% of total)
        assert result["top_10pct"]["n"] == 1
        assert abs(result["top_10pct"]["share"] - 1/3) < 1e-10, "Top-10% should have 1/3 of energy"
        
        # HHI should be 1/3 for uniform 3-layer distribution
        expected_hhi = 3 * (1/3)**2  # Sum of squared shares
        assert abs(result["concentration_index"] - expected_hhi) < 1e-10
    
    def test_concentrated_energy_distribution(self):
        """Test analysis with highly concentrated energy."""
        # Layer 0 dominates
        module_energies = {
            "model.layers.0.attn.q": 9.0,   # Layer 0: 9.0 (90%)
            "model.layers.1.attn.q": 0.5,   # Layer 1: 0.5 (5%)
            "model.layers.2.attn.q": 0.5,   # Layer 2: 0.5 (5%)
        }
        
        result = compute_layer_energy_concentration(module_energies, top_k=2)
        
        assert result["energy_total_fro2"] == 10.0
        
        # Layer 0 should dominate
        layer_0 = next(l for l in result["layers"] if l["layer"] == 0)
        assert abs(layer_0["share"] - 0.9) < 1e-10, "Layer 0 should have 90% share"
        
        # Top-k should capture most energy
        assert result["top_k"]["share"] > 0.9, "Top-2 should capture >90% of energy"
        
        # HHI should be high (concentrated)
        assert result["concentration_index"] > 0.8, "HHI should indicate high concentration"
    
    def test_zero_energy_handling(self):
        """Test hygiene for zero energy cases."""
        # Empty case
        result_empty = compute_layer_energy_concentration({})
        assert result_empty["energy_total_fro2"] == 0.0
        assert result_empty["top_k"]["share"] is None
        assert result_empty["top_10pct"]["share"] is None
        assert result_empty["concentration_index"] is None
        
        # Near-zero case
        tiny_energies = {"model.layers.0.attn.q": 1e-15}
        result_tiny = compute_layer_energy_concentration(tiny_energies)
        assert result_tiny["top_k"]["share"] is None, "Should return None for near-zero energy"
        assert result_tiny["concentration_index"] is None, "Should return None for near-zero energy"
    
    def test_unknown_layer_handling(self):
        """Test handling of modules that don't match layer patterns."""
        module_energies = {
            "model.layers.0.attn.q": 2.0,    # Layer 0
            "model.layers.1.attn.q": 3.0,    # Layer 1
            "model.embed_tokens": 1.0,       # Unknown layer
            "model.final_layer_norm": 0.5,   # Unknown layer
        }
        
        result = compute_layer_energy_concentration(module_energies)
        
        # Should have 3 layers: 0, 1, and -1 (unknown)
        layer_indices = {l["layer"] for l in result["layers"]}
        assert 0 in layer_indices, "Should have layer 0"
        assert 1 in layer_indices, "Should have layer 1" 
        assert -1 in layer_indices, "Should have unknown layer (-1)"
        
        # Unknown layer should have combined energy
        unknown_layer = next(l for l in result["layers"] if l["layer"] == -1)
        assert unknown_layer["energy_fro2"] == 1.5, "Unknown layer should have combined energy"


class TestSqrtPsdUtility:
    """Test matrix square root utility for positive semi-definite matrices."""
    
    def test_sqrt_psd_correctness(self):
        """Test that sqrt_psd produces correct matrix square roots."""
        # Create a known PSD matrix
        A = torch.randn(4, 4, dtype=torch.float64)
        M = A @ A.T  # Guaranteed PSD
        
        sqrt_M = sqrt_psd(M)
        
        # Verify sqrt_M @ sqrt_M ≈ M
        reconstructed = sqrt_M @ sqrt_M
        error = torch.linalg.norm(reconstructed - M).item()
        assert error < 1e-10, f"sqrt_psd reconstruction error {error:.2e} too large"
        
        # Should be symmetric
        sqrt_M_T = sqrt_M.T
        symmetry_error = torch.linalg.norm(sqrt_M - sqrt_M_T).item()
        assert symmetry_error < 1e-10, f"sqrt_psd result should be symmetric, error {symmetry_error:.2e}"
    
    def test_sqrt_psd_with_zero_eigenvalues(self):
        """Test sqrt_psd handles zero eigenvalues gracefully."""
        # Create rank-deficient PSD matrix
        A = torch.randn(5, 2, dtype=torch.float64)  # Rank 2
        M = A @ A.T  # 5x5 but rank 2
        
        sqrt_M = sqrt_psd(M, eps=1e-12)
        
        # Should not crash and should produce valid result
        assert sqrt_M.shape == M.shape
        assert torch.all(torch.isfinite(sqrt_M)), "sqrt_psd should produce finite values"
        
        # Reconstruction should still work for the non-null space
        reconstructed = sqrt_M @ sqrt_M
        # Error should be small in the subspace where M has support
        frobenius_error = torch.linalg.norm(reconstructed - M).item()
        assert frobenius_error < 1e-10, f"Reconstruction error {frobenius_error:.2e} too large even for rank-deficient case"


class TestIntegrationEndToEnd:
    """Integration tests that verify end-to-end functionality."""
    
    def test_complete_workflow_small_example(self):
        """Test complete workflow from LoRA factors to concentration analysis."""
        # Create a realistic small example
        torch.manual_seed(12345)
        
        # Simulate 3 layers with different energy concentrations
        modules_and_factors = [
            ("model.layers.0.attn.q", torch.randn(2, 8, dtype=torch.float64), torch.randn(12, 2, dtype=torch.float64), 1.0),
            ("model.layers.0.attn.k", torch.randn(2, 8, dtype=torch.float64), torch.randn(12, 2, dtype=torch.float64), 1.0),
            ("model.layers.1.attn.q", torch.randn(2, 8, dtype=torch.float64), torch.randn(12, 2, dtype=torch.float64), 2.0),  # Higher scaling
            ("model.layers.2.attn.q", torch.randn(2, 8, dtype=torch.float64), torch.randn(12, 2, dtype=torch.float64), 0.5),  # Lower scaling
        ]
        
        # Compute per-module energies
        module_energies = {}
        all_norms = []
        
        for module_name, A, B, scaling in modules_and_factors:
            fro_norm, spectral_norm = compute_lora_norms(A, B, scaling)
            module_energies[module_name] = fro_norm ** 2  # Energy = squared Frobenius norm
            all_norms.append((fro_norm, spectral_norm))
            
            # Verify norms are reasonable
            assert fro_norm > 0, f"Frobenius norm should be positive for {module_name}"
            assert spectral_norm > 0, f"Spectral norm should be positive for {module_name}"
            assert spectral_norm <= fro_norm, f"Spectral norm should be ≤ Frobenius norm for {module_name}"
        
        # Compute concentration analysis
        concentration = compute_layer_energy_concentration(module_energies, top_k=2)
        
        # Verify structure
        assert "energy_total_fro2" in concentration
        assert "layers" in concentration
        assert "top_k" in concentration
        assert "top_10pct" in concentration
        assert "concentration_index" in concentration
        
        # Should have 3 layers (0, 1, 2)
        layer_indices = {l["layer"] for l in concentration["layers"]}
        assert layer_indices == {0, 1, 2}, f"Expected layers 0,1,2, got {layer_indices}"
        
        # Total energy should equal sum of module energies
        expected_total = sum(module_energies.values())
        assert abs(concentration["energy_total_fro2"] - expected_total) < 1e-10
        
        # Layer shares should sum to 1
        total_share = sum(l["share"] for l in concentration["layers"])
        assert abs(total_share - 1.0) < 1e-10, "Layer shares should sum to 1"
        
        # HHI should be reasonable (between 1/3 and 1 for 3 layers)
        hhi = concentration["concentration_index"]
        assert 1/3 <= hhi <= 1.0, f"HHI {hhi} should be between 1/3 and 1 for 3 layers"
        
        print(f"Integration test passed: {len(modules_and_factors)} modules, "
              f"3 layers, total energy {concentration['energy_total_fro2']:.6f}, "
              f"HHI {hhi:.3f}")


if __name__ == "__main__":
    # Run tests directly if this file is executed
    import sys
    
    print("Running gain_metrics unit tests...")
    
    # Basic smoke test
    test_instance = TestLoRANormComputation()
    
    # Set up random seed manually
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    
    test_instance.test_small_random_matrices_accuracy(None)
    print("✅ Norm computation accuracy test passed")
    
    test_instance.test_scaling_correctness(None)
    print("✅ Scaling correctness test passed")
    
    test_instance.test_near_zero_eigenvalues_stability(None)
    print("✅ Numerical stability test passed")
    
    layer_test = TestLayerExtraction()
    layer_test.test_common_transformer_patterns()
    print("✅ Layer extraction test passed")
    
    concentration_test = TestEnergyConcentrationAnalysis()
    concentration_test.test_uniform_energy_distribution()
    concentration_test.test_concentrated_energy_distribution()
    print("✅ Energy concentration test passed")
    
    integration_test = TestIntegrationEndToEnd()
    integration_test.test_complete_workflow_small_example()
    print("✅ End-to-end integration test passed")
    
    print("\n🎉 All gain_metrics tests passed! You can sleep at night. 😴")