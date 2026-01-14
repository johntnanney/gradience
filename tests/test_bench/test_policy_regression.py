"""
Policy regression tests to protect calibrated safe uniform baseline decisions.

These tests ensure that future refactors or docs rewrites don't silently undo
the empirical calibration work by checking that canonical policy decisions
remain intact.
"""

import unittest
import yaml
from pathlib import Path


class TestSafeUniformPolicyRegression(unittest.TestCase):
    """Regression tests for safe uniform baseline policy integrity."""

    def setUp(self):
        """Load the canonical policy file."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.policy_path = self.repo_root / "gradience" / "bench" / "policies" / "safe_uniform.yaml"
        self.validation_policy_path = self.repo_root / "VALIDATION_POLICY.md"
        self.bench_readme_path = self.repo_root / "gradience" / "bench" / "README.md"
        
        # Load machine-consumable policy
        with open(self.policy_path) as f:
            self.policy = yaml.safe_load(f)
        
        # Load prose documentation
        with open(self.validation_policy_path) as f:
            self.validation_policy_text = f.read()
        
        with open(self.bench_readme_path) as f:
            self.bench_readme_text = f.read()

    def test_policy_file_exists_and_loads(self):
        """Verify the canonical policy file exists and is valid YAML."""
        self.assertTrue(self.policy_path.exists(), 
                       f"Safe uniform policy file missing: {self.policy_path}")
        self.assertIsInstance(self.policy, dict, "Policy file should contain valid YAML dict")

    def test_policy_version_and_metadata(self):
        """Verify policy version and critical metadata."""
        self.assertEqual(self.policy["policy_version"], "0.1", 
                        "Policy version changed - check if this is intentional")
        self.assertEqual(self.policy["policy_name"], "Safe Uniform Baseline Policy")
        self.assertEqual(self.policy["scope"]["model_family"], "distilbert")
        self.assertEqual(self.policy["scope"]["task"], "glue/sst2")

    def test_safety_criteria_thresholds(self):
        """Verify the core safety thresholds remain intact."""
        criteria = self.policy["criteria"]
        
        # Core safety policy: ≥67% seeds PASS AND worst seed Δ≥-2.5%
        self.assertEqual(criteria["pass_rate_min"], 0.67, 
                        "Pass rate threshold changed - this invalidates calibration")
        self.assertEqual(criteria["worst_delta_min"], -0.025, 
                        "Worst delta threshold changed - this invalidates calibration")

    def test_primary_recommendation_r20(self):
        """Verify r=20 remains the primary safe uniform baseline."""
        recommendations = self.policy["recommendations"]
        
        self.assertEqual(recommendations["primary_uniform_rank"], 20,
                        "Primary uniform rank changed from r=20 - calibration invalidated")
        self.assertEqual(recommendations["conservative_uniform_rank"], 24,
                        "Conservative uniform rank changed from r=24")
        self.assertIn(16, recommendations["avoid_uniform_ranks"],
                     "r=16 no longer marked unsafe - this contradicts empirical evidence")

    def test_empirical_evidence_integrity(self):
        """Verify the empirical evidence that justifies the policy remains documented."""
        evidence = self.policy["empirical_evidence"]
        
        # r=16: unsafe (0% pass rate, worst delta -0.080)
        r16_evidence = evidence["uniform_r16"]
        self.assertEqual(r16_evidence["pass_rate"], 0.0,
                        "r=16 pass rate evidence changed - this was empirically unsafe")
        self.assertEqual(r16_evidence["worst_delta"], -0.080,
                        "r=16 worst delta evidence changed")
        self.assertEqual(r16_evidence["status"], "RISKY",
                        "r=16 safety status changed - contradicts empirical evidence")
        
        # r=20: safe (-0.008 worst delta, 100% pass rate)  
        r20_evidence = evidence["uniform_r20"]
        self.assertEqual(r20_evidence["pass_rate"], 1.0,
                        "r=20 pass rate evidence changed")
        self.assertEqual(r20_evidence["worst_delta"], -0.008,
                        "r=20 worst delta evidence changed")
        self.assertEqual(r20_evidence["status"], "SAFE",
                        "r=20 safety status changed - contradicts empirical evidence")

    def test_fallback_strategy_documented(self):
        """Verify the fallback strategy for unsafe uniform scenarios."""
        fallback = self.policy["fallback_strategy"]
        
        self.assertEqual(fallback["default_recommendation"], "per_layer_adaptive",
                        "Fallback strategy changed from per-layer adaptive")
        self.assertIn("No policy-compliant uniform compression found", 
                     fallback["fallback_reason"])

    def test_prose_documentation_consistency(self):
        """Verify prose documentation remains consistent with machine policy."""
        # VALIDATION_POLICY.md should contain key policy elements
        self.assertIn("≥ 67% seeds PASS AND worst seed Δ ≥ -2.5%", 
                     self.validation_policy_text,
                     "Safety policy definition missing from VALIDATION_POLICY.md")
        self.assertIn("Uniform r=20", self.validation_policy_text,
                     "r=20 recommendation missing from validation policy")
        self.assertIn("Uniform r=16", self.validation_policy_text,
                     "r=16 safety information missing from validation policy")
        
        # Bench README should contain current baseline
        self.assertIn("**r=20**", self.bench_readme_text,
                     "r=20 primary recommendation missing from Bench README")
        self.assertIn("TASK/MODEL DEPENDENT", self.bench_readme_text,
                     "Task/model dependency warning missing from Bench README")

    def test_policy_limitations_documented(self):
        """Verify critical limitations are properly documented."""
        limitations = self.policy["limitations"]
        
        limitation_text = " ".join(limitations)
        self.assertIn("Task/model specific", limitation_text,
                     "Task/model limitation warning missing")
        self.assertIn("Re-validate", limitation_text,
                     "Re-validation requirement missing")
        self.assertIn("SST-2", limitation_text,
                     "Dataset-specific limitation missing")

    def test_machine_consumable_api_compatibility(self):
        """Verify the policy structure supports the documented API usage."""
        # Test the example from policies/README.md works with current structure
        self.assertIn("scope", self.policy)
        self.assertIn("model_family", self.policy["scope"])
        self.assertIn("task", self.policy["scope"])
        self.assertIn("recommendations", self.policy)
        self.assertIn("primary_uniform_rank", self.policy["recommendations"])
        self.assertIn("conservative_uniform_rank", self.policy["recommendations"])
        self.assertIn("avoid_uniform_ranks", self.policy["recommendations"])
        
        # Verify data types are correct for programmatic use
        self.assertIsInstance(self.policy["recommendations"]["primary_uniform_rank"], int)
        self.assertIsInstance(self.policy["recommendations"]["conservative_uniform_rank"], int)
        self.assertIsInstance(self.policy["recommendations"]["avoid_uniform_ranks"], list)
        self.assertIsInstance(self.policy["criteria"]["pass_rate_min"], float)
        self.assertIsInstance(self.policy["criteria"]["worst_delta_min"], float)


class TestPolicyUsageExamples(unittest.TestCase):
    """Test that documented usage examples continue to work."""
    
    def setUp(self):
        """Load policy for testing usage examples."""
        repo_root = Path(__file__).parent.parent.parent
        policy_path = repo_root / "gradience" / "bench" / "policies" / "safe_uniform.yaml"
        with open(policy_path) as f:
            self.policy = yaml.safe_load(f)

    def test_documented_check_uniform_safety_function(self):
        """Test the example function from policies/README.md works correctly."""
        def check_uniform_safety(rank, model_family="distilbert", task="glue/sst2"):
            if self.policy["scope"]["model_family"] != model_family:
                return {"compliant": False, "reason": "Model family not covered by policy"}
            
            if self.policy["scope"]["task"] != task:
                return {"compliant": False, "reason": "Task not covered by policy"}
            
            primary_rank = self.policy["recommendations"]["primary_uniform_rank"]
            conservative_rank = self.policy["recommendations"]["conservative_uniform_rank"] 
            avoid_ranks = self.policy["recommendations"]["avoid_uniform_ranks"]
            
            if rank in avoid_ranks:
                return {"compliant": False, "reason": f"Rank {rank} violates safety policy"}
            elif rank == primary_rank:
                return {"compliant": True, "level": "primary", "compression": "25.0%"}
            elif rank == conservative_rank:
                return {"compliant": True, "level": "conservative", "compression": "16.6%"}
            else:
                return {"compliant": "unknown", "reason": f"Rank {rank} not validated by policy"}
        
        # Test cases that should remain stable
        r20_result = check_uniform_safety(20)
        self.assertTrue(r20_result["compliant"])
        self.assertEqual(r20_result["level"], "primary")
        
        r24_result = check_uniform_safety(24)
        self.assertTrue(r24_result["compliant"])
        self.assertEqual(r24_result["level"], "conservative")
        
        r16_result = check_uniform_safety(16)
        self.assertFalse(r16_result["compliant"])
        self.assertIn("violates safety policy", r16_result["reason"])


if __name__ == "__main__":
    unittest.main()