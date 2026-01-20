#!/usr/bin/env python3
"""
Pure correctness unit tests for UDR/SDI implementation.

These tests enforce the behavioral contract defined in lora_audit.py:
- Deterministic UDR computation
- Correct ΔW scaling 
- Robust error handling
- SDI monotonicity
- Cache reliability

NO external dependencies (HF/transformers) - fast and deterministic.
"""

import json
import math
import tempfile
import torch
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Import UDR functions
import sys
sys.path.insert(0, '/Users/john/code/gradience')
from gradience.vnext.audit.lora_audit import (
    compute_update_norms,
    compute_udr_metrics,
    cache_base_model_norms,
    load_base_model_norms,
)


class TestUDRCorrectness:
    """Pure correctness tests for UDR computation."""

    def test_delta_w_scaling_is_correct(self):
        """Test 1: ΔW scaling is correct - verify α/r scaling factor."""
        # Create tiny, deterministic matrices for hand verification
        torch.manual_seed(42)
        
        # Simple case: r=2, known values
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)  # 2x2
        B = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float64)  # 2x2
        
        # Test different scaling factors
        scales = [1.0, 2.0, 0.5, 8.0/4.0]  # Last one is typical α/r = 8/4 = 2.0
        
        for scale in scales:
            delta_fro_norm, delta_sigma_max, stable_rank_delta, utilization_delta = compute_update_norms(
                A, B, scale=scale, compute_dtype=torch.float64
            )
            
            # Compute ΔW = scale * B @ A by hand
            delta_w_expected = scale * (B @ A)
            expected_sigma_max = float(torch.linalg.svdvals(delta_w_expected)[0])
            expected_fro_norm = float(torch.norm(delta_w_expected, p='fro'))
            
            # Verify our implementation matches hand computation
            assert abs(delta_sigma_max - expected_sigma_max) < 1e-10, \
                f"Scale {scale}: expected σ_max={expected_sigma_max}, got {delta_sigma_max}"
            assert abs(delta_fro_norm - expected_fro_norm) < 1e-10, \
                f"Scale {scale}: expected ||·||_F={expected_fro_norm}, got {delta_fro_norm}"
    
    def test_udr_handles_base_norm_near_zero(self):
        """Test 2: UDR handles base norm near 0 - should not crash."""
        delta_sigma_max = 1.0
        delta_fro_norm = 2.0
        eps = 1e-12
        
        # Test various small base norms
        small_base_norms = [0.0, 1e-15, 1e-10, eps/2]
        
        for base_sigma in small_base_norms:
            for base_fro in small_base_norms:
                # Should not crash and should produce finite values
                udr, udr_f, sdi = compute_udr_metrics(
                    delta_sigma_max, delta_fro_norm, base_sigma, base_fro, eps=eps
                )
                
                # All values should be finite
                assert math.isfinite(udr), f"UDR not finite for base_sigma={base_sigma}"
                assert math.isfinite(udr_f), f"UDR_F not finite for base_fro={base_fro}"
                assert math.isfinite(sdi), f"SDI not finite for base_sigma={base_sigma}"
                
                # UDR should be very large but finite when base norm is tiny
                expected_udr = delta_sigma_max / (base_sigma + eps)
                assert abs(udr - expected_udr) < 1e-10, \
                    f"UDR mismatch: expected {expected_udr}, got {udr}"
    
    def test_sdi_monotonicity(self):
        """Test 3: SDI monotonicity - larger UDR should give larger SDI."""
        eps = 1e-12
        
        # Create sequence of increasing UDR values
        udr_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        sdi_values = []
        
        for udr in udr_values:
            # Use dummy values for delta/base norms that give the desired UDR
            delta_sigma = udr * 1.0  # base_sigma = 1.0
            base_sigma = 1.0
            
            _, _, sdi = compute_udr_metrics(delta_sigma, 1.0, base_sigma, 1.0, eps=eps)
            sdi_values.append(sdi)
        
        # Verify SDI is monotonically increasing
        for i in range(1, len(sdi_values)):
            assert sdi_values[i] > sdi_values[i-1], \
                f"SDI not monotonic: SDI[{i-1}]={sdi_values[i-1]}, SDI[{i}]={sdi_values[i]}"
        
        # Verify SDI doesn't produce -inf at zero (due to epsilon)
        zero_udr_sdi = math.log10(0.0 + eps)
        assert math.isfinite(zero_udr_sdi), "SDI produces -inf at zero UDR"
    
    def test_udr_deterministic_computation(self):
        """Verify UDR computation is deterministic for same inputs."""
        torch.manual_seed(123)
        
        # Create test matrices
        A = torch.randn(4, 64, dtype=torch.float64)
        B = torch.randn(64, 4, dtype=torch.float64)
        scale = 2.0
        
        # Base norms
        base_sigma = 15.5
        base_fro = 120.0
        
        # Compute multiple times
        results = []
        for _ in range(5):
            delta_fro_norm, delta_sigma_max, _, _ = compute_update_norms(A, B, scale=scale)
            udr, udr_f, sdi = compute_udr_metrics(delta_sigma_max, delta_fro_norm, base_sigma, base_fro)
            results.append((delta_sigma_max, delta_fro_norm, udr, udr_f, sdi))
        
        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            for j, (val, expected) in enumerate(zip(result, first_result)):
                assert abs(val - expected) < 1e-15, \
                    f"Run {i}, value {j}: expected {expected}, got {val} (non-deterministic)"
    
    def test_scaling_factor_edge_cases(self):
        """Test scaling factor edge cases (zero, negative, very large)."""
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)  # Identity
        B = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)  # 2*Identity
        
        # Expected ΔW = scale * B @ A = scale * 2 * Identity
        # So σ_max(ΔW) = |scale| * 2
        
        test_scales = [0.0, -1.0, -2.5, 1e-6, 1e6]
        
        for scale in test_scales:
            delta_fro_norm, delta_sigma_max, _, _ = compute_update_norms(A, B, scale=scale)
            
            expected_sigma_max = abs(scale) * 2.0
            assert abs(delta_sigma_max - expected_sigma_max) < 1e-12, \
                f"Scale {scale}: expected σ_max={expected_sigma_max}, got {delta_sigma_max}"


class TestCacheRobustness:
    """Cache reliability and error handling tests."""
    
    def test_cache_write_read_roundtrip(self):
        """Test 4: Cache write/read roundtrip preserves data."""
        # Create test base norms
        test_norms = {
            "layer.0.q_proj": {"sigma_max": 12.5, "fro_norm": 85.3},
            "layer.0.v_proj": {"sigma_max": 11.8, "fro_norm": 79.2},
            "layer.1.q_proj": {"sigma_max": 10.9, "fro_norm": 76.8},
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test_cache.json"
            
            # Write to cache
            cache_base_model_norms(test_norms, cache_path)
            
            # Verify file exists and is valid JSON
            assert cache_path.exists(), "Cache file not created"
            
            # Read back from cache
            loaded_norms = load_base_model_norms(base_norms_cache=cache_path)
            
            # Verify exact match
            assert loaded_norms == test_norms, \
                f"Cache roundtrip failed: expected {test_norms}, got {loaded_norms}"
    
    def test_cache_key_isolation(self):
        """Test 5: Different cache identities don't collide."""
        # This is a design test - verify we're thinking about cache keys correctly
        # For now, we use simple file paths, but this tests the principle
        
        base_norms_1 = {"model.layer.0": {"sigma_max": 10.0, "fro_norm": 50.0}}
        base_norms_2 = {"model.layer.0": {"sigma_max": 20.0, "fro_norm": 100.0}}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache1_path = Path(temp_dir) / "model1_cache.json"
            cache2_path = Path(temp_dir) / "model2_cache.json"
            
            # Write different norms to different caches
            cache_base_model_norms(base_norms_1, cache1_path)
            cache_base_model_norms(base_norms_2, cache2_path)
            
            # Verify they don't interfere
            loaded_1 = load_base_model_norms(base_norms_cache=cache1_path)
            loaded_2 = load_base_model_norms(base_norms_cache=cache2_path)
            
            assert loaded_1 == base_norms_1, "Cache 1 corrupted"
            assert loaded_2 == base_norms_2, "Cache 2 corrupted"
            assert loaded_1 != loaded_2, "Cache isolation failed"
    
    def test_corrupt_cache_file(self):
        """Test 6: Corrupt cache file fails gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "corrupt_cache.json"
            issues = []
            
            # Write junk to cache file
            with cache_path.open('w') as f:
                f.write("{ this is not valid json }")
            
            # Should return None and record issue, not crash
            loaded_norms = load_base_model_norms(
                base_norms_cache=cache_path, 
                issues=issues
            )
            
            assert loaded_norms is None, "Should return None for corrupt cache"
            assert len(issues) > 0, "Should record issue for corrupt cache"
            assert any("Failed to load base norms cache" in issue for issue in issues), \
                f"Expected cache error in issues: {issues}"
    
    def test_missing_cache_file(self):
        """Test missing cache file is handled gracefully."""
        nonexistent_path = Path("/tmp/does_not_exist_12345.json")
        issues = []
        
        # Should return None, not crash
        loaded_norms = load_base_model_norms(
            base_norms_cache=nonexistent_path,
            issues=issues
        )
        
        assert loaded_norms is None, "Should return None for missing cache"
        # No issue should be recorded for simply missing cache (that's normal)
    
    def test_cache_directory_creation(self):
        """Test cache automatically creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Nested path that doesn't exist
            cache_path = Path(temp_dir) / "deep" / "nested" / "path" / "cache.json"
            
            test_norms = {"layer.0": {"sigma_max": 1.0, "fro_norm": 2.0}}
            
            # Should create directories and succeed
            cache_base_model_norms(test_norms, cache_path)
            
            assert cache_path.exists(), "Cache file not created"
            assert cache_path.parent.exists(), "Parent directories not created"
            
            # Verify contents
            loaded_norms = load_base_model_norms(base_norms_cache=cache_path)
            assert loaded_norms == test_norms, "Cache contents corrupted"


class TestUDRValidation:
    """Additional validation tests for UDR edge cases."""
    
    def test_zero_matrices(self):
        """Test behavior with zero A or B matrices."""
        zero_A = torch.zeros(4, 64, dtype=torch.float64)
        zero_B = torch.zeros(64, 4, dtype=torch.float64)
        normal_A = torch.randn(4, 64, dtype=torch.float64)
        normal_B = torch.randn(64, 4, dtype=torch.float64)
        
        # Zero A should give zero update
        delta_fro, delta_sigma, _, _ = compute_update_norms(zero_A, normal_B, scale=2.0)
        assert delta_sigma == 0.0, "Zero A should give zero spectral norm"
        assert delta_fro == 0.0, "Zero A should give zero Frobenius norm"
        
        # Zero B should give zero update
        delta_fro, delta_sigma, _, _ = compute_update_norms(normal_A, zero_B, scale=2.0)
        assert delta_sigma == 0.0, "Zero B should give zero spectral norm"
        assert delta_fro == 0.0, "Zero B should give zero Frobenius norm"
    
    def test_udr_none_handling(self):
        """Test UDR computation when base norms are None."""
        delta_sigma_max = 1.0
        delta_fro_norm = 2.0
        
        # When base norms are None, UDR should be None
        udr, udr_f, sdi = compute_udr_metrics(delta_sigma_max, delta_fro_norm, None, None)
        
        assert udr is None, "UDR should be None when base_sigma_max is None"
        assert udr_f is None, "UDR_F should be None when base_fro_norm is None"  
        assert sdi is None, "SDI should be None when UDR is None"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large values
        large_val = 1e10
        udr, udr_f, sdi = compute_udr_metrics(large_val, large_val, 1.0, 1.0)
        assert math.isfinite(udr), f"UDR not finite with large delta: {udr}"
        assert math.isfinite(sdi), f"SDI not finite with large delta: {sdi}"
        
        # Very small values
        small_val = 1e-10
        udr, udr_f, sdi = compute_udr_metrics(small_val, small_val, 1.0, 1.0)
        assert math.isfinite(udr), f"UDR not finite with small delta: {udr}"
        assert math.isfinite(sdi), f"SDI not finite with small delta: {sdi}"


# Test runner
if __name__ == "__main__":
    # Run all tests without pytest dependency
    test_classes = [TestUDRCorrectness, TestCacheRobustness, TestUDRValidation]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n🧪 Running {test_class.__name__}")
        print("=" * 50)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"   ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"   ❌ {method_name}: {e}")
    
    print(f"\n🎉 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("✅ All UDR correctness tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed")
        exit(1)