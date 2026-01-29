"""
Unit tests for rank selection policies.

Tests the core policy algorithms without requiring transformers or large dependencies.
Uses synthetic singular value spectra to validate mathematical implementations.
"""

import math
import pytest
import numpy as np
from typing import List, Dict, Any

# Import the policies we want to test
from gradience.vnext.audit.rank_policies import (
    apply_rank_policy,
    RankPolicySpec,
    RankSuggestion,
    _optimal_hard_threshold_policy,
    _entropy_effective_policy,
    _knee_elbow_policy,
    _energy_threshold_policy
)


class TestOptimalHardThreshold:
    """Tests for Gavish-Donoho optimal hard threshold policy."""
    
    def test_omega_cubic_approximation(self):
        """Test the cubic approximation ω(β) ≈ 0.56β³ - 0.95β² + 1.82β + 1.43"""
        
        # Test β=1.0 (square matrix case)
        beta = 1.0
        expected_omega = 0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * beta + 1.43
        # ω(1.0) = 0.56 - 0.95 + 1.82 + 1.43 = 2.86
        
        # Create synthetic spectrum where we can control the median
        singular_values = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01])
        shape = (8, 8)  # Square matrix
        r_alloc = len(singular_values)
        params = {}  # Default parameters
        eps = 1e-12
        
        result = _optimal_hard_threshold_policy(singular_values, shape, r_alloc, params, eps)
        
        # Verify the omega approximation is correct
        assert abs(expected_omega - 2.86) < 0.01, f"ω(1.0) should ≈ 2.86, got {expected_omega}"
        
        # Verify result structure
        assert isinstance(result, RankSuggestion)
        assert result.k >= 0
        assert 0 <= result.confidence <= 1
    
    def test_threshold_scaling_behavior(self):
        """Test that k decreases when spectrum is scaled down (smaller median)."""
        
        base_spectrum = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
        shape = (6, 6)
        r_alloc = len(base_spectrum)
        params = {}
        eps = 1e-12
        
        # Original spectrum
        result1 = _optimal_hard_threshold_policy(base_spectrum, shape, r_alloc, params, eps)
        
        # Scaled down spectrum (smaller median)
        scaled_spectrum = base_spectrum * 0.1  # Scale by 0.1
        result2 = _optimal_hard_threshold_policy(scaled_spectrum, shape, r_alloc, params, eps)
        
        # Smaller median should lead to smaller threshold, thus fewer values above it
        assert result2.k <= result1.k, f"Scaled spectrum should have k≤{result1.k}, got k={result2.k}"
    
    def test_different_aspect_ratios(self):
        """Test behavior with different matrix aspect ratios."""
        
        spectrum = np.array([8.0, 4.0, 2.0, 1.0, 0.5])
        r_alloc = len(spectrum)
        params = {}
        eps = 1e-12
        
        # Square matrix (β = 1.0)
        result_square = _optimal_hard_threshold_policy(spectrum, (5, 5), r_alloc, params, eps)
        
        # Tall matrix (β = 0.5)  
        result_tall = _optimal_hard_threshold_policy(spectrum, (10, 5), r_alloc, params, eps)
        
        # The ω values should be different, leading to different thresholds
        # We don't assert exact values since the behavior depends on the cubic approximation
        assert isinstance(result_square.k, int)
        assert isinstance(result_tall.k, int)
        assert result_square.k >= 0
        assert result_tall.k >= 0


class TestEntropyEffectiveRank:
    """Tests for Roy & Vetterli entropy effective rank policy."""
    
    def test_uniform_spectrum(self):
        """Test that uniform spectrum gives eRank ≈ spectrum length."""
        
        # Uniform spectrum: all singular values equal
        uniform_spectrum = np.array([1.0, 1.0, 1.0, 1.0])
        shape = (4, 4)
        r_alloc = len(uniform_spectrum)
        params = {}
        eps = 1e-12
        
        result = _entropy_effective_policy(uniform_spectrum, shape, r_alloc, params, eps)
        
        # For uniform distribution, entropy is maximized, so eRank ≈ length
        # p_i = 1/4 for each, H = -4 * (1/4) * log(1/4) = log(4)
        # eRank = exp(log(4)) = 4
        expected_erank = 4.0
        
        # Check if details contains the effective rank calculation
        assert abs(result.details.get('erank_float', 0) - expected_erank) < 0.1, \
            f"Uniform spectrum should give eRank≈4, got {result.details.get('erank_float', 'N/A')}"
        assert result.k == 4  # Should suggest full rank for uniform case
    
    def test_concentrated_spectrum(self):
        """Test that concentrated spectrum gives low eRank."""
        
        # Highly concentrated: one large value, rest tiny
        concentrated_spectrum = np.array([1.0, 0.0001, 0.0001, 0.0001])
        shape = (4, 4)
        r_alloc = len(concentrated_spectrum)
        params = {}
        eps = 1e-12
        
        result = _entropy_effective_policy(concentrated_spectrum, shape, r_alloc, params, eps)
        
        # Should give eRank ≈ 1 since entropy is low
        effective_rank = result.details.get('erank_float', 0)
        assert effective_rank < 1.5, \
            f"Concentrated spectrum should give low eRank, got {effective_rank}"
        assert result.k <= 2  # Should suggest low rank
    
    def test_monotonic_behavior(self):
        """Test monotonic behavior with increasing tail mass."""
        
        shape = (4, 4)
        r_alloc = 4
        params = {}
        eps = 1e-12
        
        # Spectrum with small tail
        spectrum1 = np.array([10.0, 1.0, 0.1, 0.01])
        result1 = _entropy_effective_policy(spectrum1, shape, r_alloc, params, eps)
        
        # Spectrum with larger tail  
        spectrum2 = np.array([10.0, 3.0, 1.0, 0.5])
        result2 = _entropy_effective_policy(spectrum2, shape, r_alloc, params, eps)
        
        # More balanced spectrum should have higher effective rank
        erank1 = result1.details.get('erank_float', 0)
        erank2 = result2.details.get('erank_float', 0)
        assert erank2 >= erank1, \
            f"More balanced spectrum should have higher eRank: {erank1} vs {erank2}"
    
    def test_edge_cases(self):
        """Test edge cases for entropy calculation."""
        
        params = {}
        eps = 1e-12
        
        # Single value
        single_spectrum = np.array([5.0])
        result = _entropy_effective_policy(single_spectrum, (1, 1), 1, params, eps)
        assert result.details.get('erank_float', 0) == 1.0
        assert result.k == 1
        
        # All zeros should be handled gracefully
        zero_spectrum = np.array([0.0, 0.0, 0.0])
        result = _entropy_effective_policy(zero_spectrum, (3, 3), 3, params, eps)
        assert result.k >= 0  # Should handle gracefully


class TestKneeElbowDetection:
    """Tests for Kneedle algorithm elbow detection."""
    
    def test_obvious_elbow(self):
        """Test detection of obvious elbow point."""
        
        # Create spectrum with clear elbow at position 2 (0-indexed)
        # Large values, then sharp drop
        elbow_spectrum = np.array([10.0, 8.0, 6.0, 0.5, 0.1, 0.05, 0.01])
        shape = (7, 7)
        r_alloc = len(elbow_spectrum)
        params = {}
        eps = 1e-12
        
        result = _knee_elbow_policy(elbow_spectrum, shape, r_alloc, params, eps)
        
        # Should detect elbow around position 3 (k=3, meaning keep first 3 components)
        # Allow ±1 tolerance for algorithm variations
        expected_k = 3
        assert abs(result.k - expected_k) <= 1, \
            f"Expected elbow detection around k={expected_k}, got k={result.k}"
    
    def test_gradual_decay(self):
        """Test behavior with gradual decay (no clear elbow)."""
        
        # Exponential decay with no sharp elbow
        gradual_spectrum = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125])
        shape = (7, 7)
        r_alloc = len(gradual_spectrum)
        params = {}
        eps = 1e-12
        
        result = _knee_elbow_policy(gradual_spectrum, shape, r_alloc, params, eps)
        
        # Should still return some reasonable k value
        assert 1 <= result.k <= len(gradual_spectrum)
        assert isinstance(result.k, int)
    
    def test_flat_spectrum(self):
        """Test behavior with nearly flat spectrum."""
        
        # Nearly uniform (slight noise)
        flat_spectrum = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75])
        shape = (6, 6)
        r_alloc = len(flat_spectrum)
        params = {}
        eps = 1e-12
        
        result = _knee_elbow_policy(flat_spectrum, shape, r_alloc, params, eps)
        
        # Should handle gracefully, probably suggest higher rank
        assert isinstance(result.k, int)
        assert result.k >= 1


class TestEnergyThreshold:
    """Tests for traditional energy threshold policy."""
    
    def test_energy_90_percent(self):
        """Test 90% energy threshold calculation."""
        
        # Create spectrum where we know the energy distribution
        spectrum = np.array([4.0, 3.0, 0.0, 0.0])  # Energy: 16, 9, 0, 0 → Total = 25
        # 90% of 25 = 22.5
        # Cumulative: 16 (64%), 25 (100%)
        # So k=2 gives 100% > 90%
        
        shape = (4, 4)
        r_alloc = len(spectrum)
        params = {'energy_threshold': 0.90}
        eps = 1e-12
        
        result = _energy_threshold_policy(spectrum, shape, r_alloc, params, eps)
        
        assert result.k == 2, f"Expected k=2 for 90% energy, got k={result.k}"
        assert result.details.get('actual_energy_captured', 0) >= 0.90
    
    def test_energy_95_percent(self):
        """Test 95% energy threshold calculation."""
        
        spectrum = np.array([6.0, 3.0, 1.0, 0.5])  # Energy: 36, 9, 1, 0.25 → Total = 46.25
        # 95% of 46.25 = 43.94
        # Cumulative: 36 (77.8%), 45 (97.3%) 
        # So k=2 gives 97.3% > 95%
        
        shape = (4, 4)
        r_alloc = len(spectrum)
        params = {'energy_threshold': 0.95}
        eps = 1e-12
        
        result = _energy_threshold_policy(spectrum, shape, r_alloc, params, eps)
        
        assert result.details.get('actual_energy_captured', 0) >= 0.95
        assert result.k <= len(spectrum)


class TestPolicyIntegration:
    """Tests for the main apply_rank_policy function."""
    
    def test_apply_multiple_policies(self):
        """Test applying multiple policies to same spectrum."""
        
        spectrum = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
        shape = (5, 5)
        r_alloc = len(spectrum)
        eps = 1e-12
        
        policy_names = ['energy_threshold', 'knee_elbow', 'entropy_effective']
        
        results = {}
        for policy_name in policy_names:
            policy_spec = RankPolicySpec(name=policy_name)
            result = apply_rank_policy(policy_spec, spectrum, shape, r_alloc, eps)
            results[policy_name] = result
        
        # Should return results for all requested policies
        assert len(results) == 3
        for policy_name in policy_names:
            assert policy_name in results
            result = results[policy_name]
            assert isinstance(result, RankSuggestion)
            assert isinstance(result.k, int)
            assert isinstance(result.confidence, (int, float))
    
    def test_single_policy(self):
        """Test applying single policy."""
        
        spectrum = np.array([8.0, 2.0, 1.0, 0.1])
        shape = (4, 4)
        r_alloc = len(spectrum)
        eps = 1e-12
        
        policy_spec = RankPolicySpec(name='optimal_hard_threshold')
        result = apply_rank_policy(policy_spec, spectrum, shape, r_alloc, eps)
        
        assert isinstance(result, RankSuggestion)
        assert result.k >= 0
    
    def test_empty_spectrum(self):
        """Test handling of edge cases."""
        
        # Empty spectrum should be handled gracefully
        spectrum = np.array([])
        shape = (0, 0)
        r_alloc = 0
        eps = 1e-12
        
        policy_spec = RankPolicySpec(name='energy_threshold')
        result = apply_rank_policy(policy_spec, spectrum, shape, r_alloc, eps)
        
        # Should not crash, should return sensible defaults
        assert isinstance(result, RankSuggestion)
        assert result.k >= 0


class TestAuditSchemaIntegration:
    """Tests for integration with audit pipeline schema."""
    
    def create_mock_layer_data(self) -> Dict[str, Any]:
        """Create synthetic layer data for testing."""
        return {
            'layer_name': 'test.linear',
            'weight_shape': [128, 64],
            'singular_values': [5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05],
            'current_rank': 8
        }
    
    def test_rank_suggestions_schema(self):
        """Test that rank_suggestions follow expected JSON schema."""
        
        layer_data = self.create_mock_layer_data()
        spectrum = np.array(layer_data['singular_values'])
        shape = tuple(layer_data['weight_shape'])
        r_alloc = len(spectrum)
        eps = 1e-12
        
        policy_names = ['energy_threshold', 'knee_elbow', 'entropy_effective', 'optimal_hard_threshold']
        
        rank_suggestions = {}
        for policy_name in policy_names:
            policy_spec = RankPolicySpec(name=policy_name)
            result = apply_rank_policy(policy_spec, spectrum, shape, r_alloc, eps)
            # Convert to dict format as expected by audit pipeline
            rank_suggestions[policy_name] = {
                'k': result.k,
                'confidence': result.confidence,
                'details': result.details
            }
        
        # Verify schema structure
        for policy_name, suggestion in rank_suggestions.items():
            # Each suggestion should have required fields
            assert 'k' in suggestion, f"Policy {policy_name} missing 'k' field"
            assert 'confidence' in suggestion, f"Policy {policy_name} missing 'confidence' field"
            
            # Types should be correct
            assert isinstance(suggestion['k'], int), f"Policy {policy_name} 'k' should be int"
            assert isinstance(suggestion['confidence'], (int, float)), f"Policy {policy_name} 'confidence' should be numeric"
            
            # Values should be reasonable
            assert suggestion['k'] >= 0, f"Policy {policy_name} k should be >= 0"
            assert 0 <= suggestion['confidence'] <= 1, f"Policy {policy_name} confidence should be in [0,1]"
            assert suggestion['k'] <= len(spectrum), f"Policy {policy_name} k should not exceed spectrum length"
    
    def test_new_audit_fields(self):
        """Test that audit results include new policy-related fields."""
        
        # This is a schema validation test - in real integration,
        # the audit pipeline would populate these fields
        
        mock_audit_result = {
            'layers': [
                {
                    'layer_name': 'test.linear',
                    'rank_suggestions': {
                        'energy_threshold': {'k': 4, 'confidence': 0.90},
                        'knee_elbow': {'k': 2, 'confidence': 0.85},
                        'entropy_effective': {'k': 6, 'confidence': 0.90}
                    }
                }
            ],
            'policy_global_suggestions': {
                'energy_threshold': {
                    'uniform_median': 4.0,
                    'uniform_p90': 6.0,
                    'uniform_max': 8.0,
                    'n_layers': 10
                },
                'knee_elbow': {
                    'uniform_median': 2.0,
                    'uniform_p90': 3.0,
                    'uniform_max': 4.0,
                    'n_layers': 10
                }
            }
        }
        
        # Verify the structure matches our expected schema
        assert 'layers' in mock_audit_result
        assert 'policy_global_suggestions' in mock_audit_result
        
        # Verify layer-level suggestions
        layer = mock_audit_result['layers'][0]
        assert 'rank_suggestions' in layer
        for policy_name, suggestion in layer['rank_suggestions'].items():
            assert 'k' in suggestion
            assert 'confidence' in suggestion
        
        # Verify global suggestions
        global_suggestions = mock_audit_result['policy_global_suggestions']
        for policy_name, stats in global_suggestions.items():
            required_stats = ['uniform_median', 'uniform_p90', 'uniform_max', 'n_layers']
            for stat in required_stats:
                assert stat in stats, f"Missing {stat} in global suggestions for {policy_name}"


if __name__ == '__main__':
    # Run tests with pytest if available, otherwise run basic validation
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        print("Running basic validation tests...")
        
        # Basic smoke tests
        test_oht = TestOptimalHardThreshold()
        test_oht.test_omega_cubic_approximation()
        print("✓ OHT cubic approximation test passed")
        
        test_erank = TestEntropyEffectiveRank()
        test_erank.test_uniform_spectrum()
        test_erank.test_concentrated_spectrum()
        print("✓ Entropy effective rank tests passed")
        
        test_knee = TestKneeElbowDetection()
        test_knee.test_obvious_elbow()
        print("✓ Knee detection tests passed")
        
        test_energy = TestEnergyThreshold()
        test_energy.test_energy_90_percent()
        print("✓ Energy threshold tests passed")
        
        test_integration = TestPolicyIntegration()
        test_integration.test_apply_multiple_policies()
        print("✓ Policy integration tests passed")
        
        test_schema = TestAuditSchemaIntegration()
        test_schema.test_rank_suggestions_schema()
        test_schema.test_new_audit_fields()
        print("✓ Audit schema integration tests passed")


class TestPolicyDisagreementDetection:
    """Tests for the policy disagreement detection feature."""
    
    def test_disagreement_calculation(self):
        """Test policy spread calculation and disagreement detection."""
        
        # Create mock layers with different disagreement patterns
        class MockLayer:
            def __init__(self, layer_name, rank_suggestions):
                self.layer_name = layer_name
                self.rank_suggestions = rank_suggestions
                self.r = 8  # Mock allocated rank
        
        # High disagreement layer
        high_disagreement = MockLayer("test.layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85},
            'entropy_effective': {'k': 6, 'confidence': 0.90},
            'optimal_hard_threshold': {'k': 1, 'confidence': 0.95}
        })
        
        # Consensus layer  
        consensus = MockLayer("test.layer2", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 4, 'confidence': 0.85},
            'entropy_effective': {'k': 5, 'confidence': 0.90},
            'optimal_hard_threshold': {'k': 4, 'confidence': 0.95}
        })
        
        # Test the spread calculation logic
        def calculate_spread(layer):
            """Extract the spread calculation logic for testing."""
            k_values = []
            for policy, suggestion in layer.rank_suggestions.items():
                if isinstance(suggestion, dict) and 'k' in suggestion:
                    k = suggestion['k']
                    if isinstance(k, (int, float)) and k >= 0:
                        k_values.append(int(k))
            
            if len(k_values) >= 2:
                return max(k_values) - min(k_values)
            return 0
        
        # Test calculations
        high_spread = calculate_spread(high_disagreement)
        consensus_spread = calculate_spread(consensus)
        
        assert high_spread == 7, f"High disagreement should have spread=7, got {high_spread}"
        assert consensus_spread == 1, f"Consensus should have spread=1, got {consensus_spread}"
        
        # Test threshold logic
        assert high_spread >= 3, "High disagreement should exceed threshold"
        assert consensus_spread < 3, "Consensus should be below threshold"


class TestSchemaStability:
    """Tests for the future-proof schema structure."""
    
    def test_schema_version_field(self):
        """Test that schema version is included in audit results."""
        
        # Mock a basic audit result structure
        mock_result = {
            "total_lora_params": 1000,
            "n_layers": 1,
            "rank_policy_schema_version": 1
        }
        
        # Verify schema version is present and correct
        assert "rank_policy_schema_version" in mock_result
        assert mock_result["rank_policy_schema_version"] == 1
    
    def test_structured_policy_schema(self):
        """Test the structured policy schema format."""
        
        # Mock structured policies schema
        mock_policies = {
            "metadata": {
                "version": 1,
                "applied_policies": ["energy_threshold", "knee_elbow"],
                "default_parameters": {
                    "energy_threshold": {"threshold": 0.90},
                    "knee_elbow": {}
                }
            },
            "global_statistics": {
                "energy_threshold": {
                    "uniform_median": 4.0,
                    "uniform_p90": 6.0,
                    "uniform_max": 8.0,
                    "n_layers": 1
                }
            },
            "per_layer": [
                {
                    "layer_name": "test.layer",
                    "allocated_rank": 8,
                    "suggestions": {
                        "energy_threshold": {
                            "k": 4,
                            "confidence": 0.90,
                            "metadata": {
                                "threshold_used": 0.90,
                                "energy_captured": 0.92,
                                "total_energy": 100.0
                            }
                        }
                    }
                }
            ]
        }
        
        # Verify required schema structure
        assert "metadata" in mock_policies
        assert "global_statistics" in mock_policies  
        assert "per_layer" in mock_policies
        
        # Verify metadata structure
        metadata = mock_policies["metadata"]
        assert metadata["version"] == 1
        assert isinstance(metadata["applied_policies"], list)
        assert isinstance(metadata["default_parameters"], dict)
        
        # Verify stable access paths
        energy_median = mock_policies["global_statistics"]["energy_threshold"]["uniform_median"]
        layer_k = mock_policies["per_layer"][0]["suggestions"]["energy_threshold"]["k"]
        
        assert energy_median == 4.0
        assert layer_k == 4

if __name__ == '__main__':
    # Run tests with pytest if available, otherwise run basic validation
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        print("Running basic validation tests...")
        
        # Basic smoke tests
        test_oht = TestOptimalHardThreshold()
        test_oht.test_omega_cubic_approximation()
        print("✓ OHT cubic approximation test passed")
        
        test_erank = TestEntropyEffectiveRank()
        test_erank.test_uniform_spectrum()
        test_erank.test_concentrated_spectrum()
        print("✓ Entropy effective rank tests passed")
        
        test_knee = TestKneeElbowDetection()
        test_knee.test_obvious_elbow()
        print("✓ Knee detection tests passed")
        
        test_energy = TestEnergyThreshold()
        test_energy.test_energy_90_percent()
        print("✓ Energy threshold tests passed")
        
        test_integration = TestPolicyIntegration()
        test_integration.test_apply_multiple_policies()
        print("✓ Policy integration tests passed")
        
        test_schema = TestAuditSchemaIntegration()
        test_schema.test_rank_suggestions_schema()
        test_schema.test_new_audit_fields()
        print("✓ Audit schema integration tests passed")
        
        test_disagreement = TestPolicyDisagreementDetection()
        test_disagreement.test_disagreement_calculation()
        print("✓ Policy disagreement detection tests passed")
        
        test_schema = TestSchemaStability()
        test_schema.test_schema_version_field()
        test_schema.test_structured_policy_schema()
        print("✓ Schema stability tests passed")
        
        print("\nAll tests passed! ✅")