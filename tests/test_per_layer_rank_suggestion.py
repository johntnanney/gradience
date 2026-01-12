import unittest
from gradience.vnext.rank_suggestion import suggest_per_layer_ranks

class TestPerLayerRankSuggestion(unittest.TestCase):
    def test_suggest_per_layer(self):
        audit = {
            "layers": [
                {"name": "layer.0.attn.q", "r": 8, "energy_rank_90": 1.0, "stable_rank": 1.2, "utilization": 0.15},
                {"name": "layer.0.attn.k", "r": 8, "energy_rank_90": 2.1, "stable_rank": 1.4, "utilization": 0.17},
                {"name": "layer.0.attn.v", "r": 8, "energy_rank_90": 3.0, "stable_rank": 2.0, "utilization": 0.25},
            ]
        }

        rep = suggest_per_layer_ranks(audit, margin=1.0)

        self.assertEqual(len(rep.layers), 3)
        # energy_rank_90 1.0 -> suggested 1
        self.assertEqual(rep.layers[0].suggested_r, 1)  # Direct check on layer
        # energy_rank_90 2.1 -> suggested 4 (bucket)
        self.assertTrue(any(s.name == "layer.0.attn.k" and s.suggested_r == 4 for s in rep.layers))
        # never suggests > current_r
        self.assertTrue(all(s.suggested_r <= s.current_r for s in rep.layers))

    def test_default_r_calculation(self):
        audit = {
            "layers": [
                {"name": "layer.0", "r": 8, "energy_rank_90": 1.5},  # -> suggested 2
                {"name": "layer.1", "r": 8, "energy_rank_90": 1.8},  # -> suggested 2
                {"name": "layer.2", "r": 8, "energy_rank_90": 3.5},  # -> suggested 4
            ]
        }

        rep = suggest_per_layer_ranks(audit)
        
        # Most common suggestion should be default_r (2 appears twice, 4 once)
        self.assertEqual(rep.default_r, 2)
        # Only layer.2 should be in rank_pattern (differs from default)
        self.assertEqual(len(rep.rank_pattern), 1)
        self.assertEqual(rep.rank_pattern["layer.2"], 4)

    def test_missing_layers_field(self):
        audit = {"summary": "no layers field"}
        
        with self.assertRaises(ValueError) as cm:
            suggest_per_layer_ranks(audit)
        
        self.assertIn("layers", str(cm.exception))

    def test_margin_application(self):
        audit = {
            "layers": [
                {"name": "layer.0", "r": 8, "energy_rank_90": 2.0},
            ]
        }

        rep = suggest_per_layer_ranks(audit, margin=1.5)
        
        # 2.0 * 1.5 = 3.0 -> buckets to 4
        self.assertEqual(rep.layers[0].suggested_r, 4)

if __name__ == "__main__":
    unittest.main()