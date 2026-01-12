import unittest

from gradience.vnext.rank_suggestion import (
    suggest_global_ranks_from_audit,
)

class TestRankSuggestion(unittest.TestCase):
    def test_infer_current_r(self):
        audit = {
            "stable_rank_mean": 1.6,
            "utilization_mean": 0.2,  # implies r=8
            "total_lora_params": 1000,
        }
        result = suggest_global_ranks_from_audit(audit)
        self.assertEqual(result.current_r, 8)

    def test_suggest_uniform_prefers_existing_fields(self):
        audit = {
            "stable_rank_mean": 1.38,
            "utilization_mean": 0.173,  # ~8
            "energy_rank_90_p50": 2.0,
            "energy_rank_90_p90": 3.0,
            "suggested_r_global_median": 2,
            "suggested_r_global_90": 4,
            "total_lora_params": 1000,
        }
        result = suggest_global_ranks_from_audit(audit)
        self.assertEqual(result.current_r, 8)
        self.assertEqual(result.suggested_r_median, 2)
        self.assertEqual(result.suggested_r_p90, 4)
        self.assertEqual(result.params_at_r_median, 250)  # 1000*(2/8)
        self.assertEqual(result.params_at_r_p90, 500)     # 1000*(4/8)

    def test_never_suggest_increase(self):
        audit = {
            "stable_rank_mean": 6.0,
            "utilization_mean": 0.75,  # implies r=8
            "energy_rank_90_p50": 16.0,  # would round to 16
            "energy_rank_90_p90": 32.0,
            "total_lora_params": 800,
        }
        result = suggest_global_ranks_from_audit(audit)
        # must cap at current_r=8
        self.assertEqual(result.current_r, 8)
        self.assertEqual(result.suggested_r_median, 8)
        self.assertEqual(result.suggested_r_p90, 8)

if __name__ == "__main__":
    unittest.main()