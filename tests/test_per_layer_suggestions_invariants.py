"""
Invariant tests for per-layer rank suggestions.

These tests ensure the per-layer suggestion algorithm maintains key mathematical
and logical invariants across future changes.
"""

import unittest
from gradience.vnext.rank_suggestion import (
    suggest_per_layer_ranks,
    DEFAULT_ALLOWED_RANKS,
    PerLayerRankSuggestion,
)


class TestPerLayerSuggestionsInvariants(unittest.TestCase):
    """Test invariant properties of per-layer rank suggestions."""

    def setUp(self):
        """Set up test data with various layer configurations."""
        self.basic_audit = {
            "layers": [
                {"name": "layer1", "r": 8, "energy_rank_90": 2.0},
                {"name": "layer2", "r": 16, "energy_rank_90": 4.5},
                {"name": "layer3", "r": 4, "energy_rank_90": 1.0},
            ]
        }
        
        self.audit_with_optional_fields = {
            "layers": [
                {
                    "name": "layer1", 
                    "r": 8, 
                    "energy_rank_90": 2.0,
                    "stable_rank": 1.5,
                    "utilization": 0.2,
                    "module_type": "attn"
                },
                {
                    "name": "layer2", 
                    "r": 16, 
                    "energy_rank_90": 4.5,
                    "stable_rank": None,  # Test missing stable_rank
                    "utilization": 0.3
                },
            ]
        }

    def test_suggested_r_in_allowed_ranks(self):
        """Test that all suggested ranks are in allowed_ranks."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        for suggestion in rep.layers:
            with self.subTest(layer=suggestion.name):
                self.assertIn(
                    suggestion.suggested_r, 
                    DEFAULT_ALLOWED_RANKS,
                    f"suggested_r {suggestion.suggested_r} not in allowed ranks {DEFAULT_ALLOWED_RANKS}"
                )

    def test_suggested_r_never_exceeds_current_r(self):
        """Test that suggested_r <= current_r (conservative principle)."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        for suggestion in rep.layers:
            with self.subTest(layer=suggestion.name):
                self.assertLessEqual(
                    suggestion.suggested_r,
                    suggestion.current_r,
                    f"suggested_r {suggestion.suggested_r} exceeds current_r {suggestion.current_r}"
                )

    def test_suggested_r_positive(self):
        """Test that suggested_r is always positive."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        for suggestion in rep.layers:
            with self.subTest(layer=suggestion.name):
                self.assertGreater(
                    suggestion.suggested_r,
                    0,
                    f"suggested_r should be positive, got {suggestion.suggested_r}"
                )

    def test_reduction_ratio_bounds(self):
        """Test that reduction_ratio is between 0 and 1."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        for suggestion in rep.layers:
            with self.subTest(layer=suggestion.name):
                self.assertGreaterEqual(
                    suggestion.reduction_ratio,
                    0.0,
                    f"reduction_ratio should be >= 0, got {suggestion.reduction_ratio}"
                )
                self.assertLessEqual(
                    suggestion.reduction_ratio,
                    1.0,
                    f"reduction_ratio should be <= 1, got {suggestion.reduction_ratio}"
                )

    def test_reduction_ratio_calculation(self):
        """Test that reduction_ratio is correctly calculated."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        for suggestion in rep.layers:
            with self.subTest(layer=suggestion.name):
                expected = 1.0 - (suggestion.suggested_r / suggestion.current_r)
                self.assertAlmostEqual(
                    suggestion.reduction_ratio,
                    expected,
                    places=6,
                    msg=f"reduction_ratio calculation incorrect for {suggestion.name}"
                )

    def test_handles_missing_stable_rank(self):
        """Test that suggestions work when stable_rank is missing or None."""
        try:
            rep = suggest_per_layer_ranks(self.audit_with_optional_fields)
            self.assertGreater(len(rep.layers), 0, "Should generate suggestions even with missing stable_rank")
            
            # Find the layer with missing stable_rank
            layer_with_missing = next(
                (s for s in rep.layers if s.name == "layer2"), 
                None
            )
            self.assertIsNotNone(layer_with_missing)
            self.assertIsNone(layer_with_missing.stable_rank)
            
        except Exception as e:
            self.fail(f"Should not raise exception with missing stable_rank: {e}")

    def test_custom_allowed_ranks(self):
        """Test that custom allowed_ranks are respected."""
        custom_ranks = (1, 2, 4)
        rep = suggest_per_layer_ranks(self.basic_audit, allowed_ranks=custom_ranks)
        
        for suggestion in rep.layers:
            with self.subTest(layer=suggestion.name):
                self.assertIn(
                    suggestion.suggested_r,
                    custom_ranks,
                    f"suggested_r {suggestion.suggested_r} not in custom allowed ranks {custom_ranks}"
                )

    def test_margin_application_monotonic(self):
        """Test that higher margins never result in higher suggested ranks."""
        margin_low = 0.5
        margin_high = 1.5
        
        rep_low = suggest_per_layer_ranks(self.basic_audit, margin=margin_low)
        rep_high = suggest_per_layer_ranks(self.basic_audit, margin=margin_high)
        
        # Create lookup for easier comparison
        low_suggestions = {s.name: s.suggested_r for s in rep_low.layers}
        high_suggestions = {s.name: s.suggested_r for s in rep_high.layers}
        
        for name in low_suggestions:
            with self.subTest(layer=name):
                self.assertLessEqual(
                    low_suggestions[name],
                    high_suggestions[name],
                    f"Higher margin should not decrease suggested rank for {name}"
                )

    def test_default_r_is_mode(self):
        """Test that default_r is the most common suggested rank."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        # Count suggested ranks
        rank_counts = {}
        for suggestion in rep.layers:
            rank = suggestion.suggested_r
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Find most common
        most_common_rank = max(rank_counts.items(), key=lambda x: x[1])[0]
        
        self.assertEqual(
            rep.default_r,
            most_common_rank,
            f"default_r {rep.default_r} should be mode of suggestions, got {most_common_rank}"
        )

    def test_rank_pattern_only_contains_exceptions(self):
        """Test that rank_pattern only contains layers that differ from default_r."""
        rep = suggest_per_layer_ranks(self.basic_audit)
        
        for name, suggested_r in rep.rank_pattern.items():
            with self.subTest(layer=name):
                self.assertNotEqual(
                    suggested_r,
                    rep.default_r,
                    f"rank_pattern should not contain default rank {rep.default_r} for {name}"
                )

    def test_empty_layers_handling(self):
        """Test graceful handling of empty layers list."""
        empty_audit = {"layers": []}
        rep = suggest_per_layer_ranks(empty_audit)
        
        self.assertEqual(len(rep.layers), 0)
        self.assertEqual(rep.default_r, 0)
        self.assertEqual(len(rep.rank_pattern), 0)
        self.assertEqual(len(rep.by_module_type_p90), 0)
        self.assertIn("No valid layer rows", rep.notes)

    def test_invalid_layers_field_raises_error(self):
        """Test that missing or invalid layers field raises appropriate error."""
        with self.assertRaises(ValueError) as cm:
            suggest_per_layer_ranks({"not_layers": []})
        self.assertIn("layer data", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            suggest_per_layer_ranks({"layers": "not_a_list"})
        self.assertIn("layer data", str(cm.exception))

    def test_new_layer_data_structure_compatibility(self):
        """Test that new layer_data.layer_rows structure works."""
        new_structure_audit = {
            "layer_data": {
                "layer_rows_schema": "v1",
                "layer_rows": self.basic_audit["layers"]
            }
        }
        
        rep = suggest_per_layer_ranks(new_structure_audit)
        self.assertGreater(len(rep.layers), 0)
        
        # Should produce same results as old structure
        rep_old = suggest_per_layer_ranks(self.basic_audit)
        self.assertEqual(rep.default_r, rep_old.default_r)
        self.assertEqual(len(rep.layers), len(rep_old.layers))

    def test_by_module_type_p90_calculation(self):
        """Test that by_module_type_p90 calculates percentiles correctly."""
        audit_with_types = {
            "layers": [
                {"name": "attn1", "r": 8, "energy_rank_90": 1.0, "module_type": "attn"},
                {"name": "attn2", "r": 8, "energy_rank_90": 2.0, "module_type": "attn"}, 
                {"name": "attn3", "r": 8, "energy_rank_90": 4.0, "module_type": "attn"},
                {"name": "mlp1", "r": 8, "energy_rank_90": 6.0, "module_type": "mlp"},
            ]
        }
        
        rep = suggest_per_layer_ranks(audit_with_types)
        
        # Should have entries for both module types
        self.assertIn("attn", rep.by_module_type_p90)
        self.assertIn("mlp", rep.by_module_type_p90)
        
        # For attn: suggested ranks will be [1, 2, 4], p90 index = int(0.9 * 2) = 1, so p90 = 2
        self.assertEqual(rep.by_module_type_p90["attn"], 2)
        
        # For mlp: only one layer, p90 should equal that value
        mlp_layer = next(s for s in rep.layers if s.module_type == "mlp")
        self.assertEqual(rep.by_module_type_p90["mlp"], mlp_layer.suggested_r)


if __name__ == "__main__":
    unittest.main()