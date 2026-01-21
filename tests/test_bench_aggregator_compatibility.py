"""
Smoke tests for bench aggregator backward compatibility.

Tests that the aggregator can handle both old-style (string) and new-style (dict)
task field formats without breaking.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np


class TestBenchAggregatorCompatibility(unittest.TestCase):
    """Test aggregator compatibility with different report formats."""
    
    def setUp(self):
        """Set up test data for aggregation."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Common report structure
        self.base_report_data = {
            "bench_version": "0.1",
            "timestamp": "2024-01-01T00:00:00Z",
            "model": "microsoft/DialoGPT-small",
            "env": {
                "python_version": "3.10.0",
                "torch_version": "2.0.0",
                "seed": 42
            },
            "probe_quality_gate": {
                "metric_key": "eval_accuracy",
                "metric_value": 0.85,
                "min_value": 0.1,
                "passed": True
            },
            "probe": {
                "rank": 16,
                "params": 10240,
                "accuracy": 0.85
            },
            "compressed": {
                "uniform_median": {
                    "rank": 4,
                    "params": 2560,
                    "accuracy": 0.82,
                    "delta_vs_probe": -0.03,
                    "param_reduction": 0.75,
                    "verdict": "PASS"
                }
            },
            "summary": {
                "recommendations_validated": True,
                "best_compression": {
                    "variant": "uniform_median",
                    "param_reduction": 0.75
                }
            }
        }
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_old_style_task_string_format(self):
        """Test aggregation with old-style task field (string format)."""
        # Create reports with old-style string task field
        reports = []
        for seed in [42, 43, 44]:
            report = self.base_report_data.copy()
            report["task"] = "glue/sst2"  # Old-style: string format
            report["env"]["seed"] = seed
            # Add some variance
            report["probe"]["accuracy"] = 0.85 + (seed - 42) * 0.01
            report["compressed"]["uniform_median"]["accuracy"] = 0.82 + (seed - 42) * 0.01
            reports.append(report)
        
        # Test aggregation
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, self.temp_dir)
        
        # Should handle old-style task field
        assert "task" in result
        assert result["task"] == "glue/sst2"  # Should preserve string format
        assert result["n_seeds"] == 3
        assert "probe" in result
        assert "accuracy" in result["probe"]
        assert "mean" in result["probe"]["accuracy"]
    
    def test_new_style_task_dict_format(self):
        """Test aggregation with new-style task field (dict format)."""
        # Create reports with new-style dict task field
        reports = []
        for seed in [42, 43, 44]:
            report = self.base_report_data.copy()
            report["task"] = {  # New-style: dict format
                "dataset": "glue",
                "subset": "sst2",
                "profile": "seqcls_glue"
            }
            report["env"]["seed"] = seed
            # Add some variance
            report["probe"]["accuracy"] = 0.85 + (seed - 42) * 0.01
            report["compressed"]["uniform_median"]["accuracy"] = 0.82 + (seed - 42) * 0.01
            reports.append(report)
        
        # Test aggregation
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, self.temp_dir)
        
        # Should handle new-style task field
        assert "task" in result
        assert isinstance(result["task"], dict)  # Should preserve dict format
        assert result["task"]["dataset"] == "glue"
        assert result["task"]["subset"] == "sst2"
        assert result["n_seeds"] == 3
    
    def test_mixed_task_formats_graceful_handling(self):
        """Test that aggregation handles mixed task formats gracefully."""
        # Mix of old and new style (edge case)
        reports = []
        
        # First report: old-style string
        report1 = self.base_report_data.copy()
        report1["task"] = "glue/sst2"
        report1["env"]["seed"] = 42
        reports.append(report1)
        
        # Second report: new-style dict  
        report2 = self.base_report_data.copy()
        report2["task"] = {"dataset": "glue", "subset": "sst2"}
        report2["env"]["seed"] = 43
        report2["probe"]["accuracy"] = 0.86
        reports.append(report2)
        
        # Should not crash (uses first report's format)
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, self.temp_dir)
        
        assert "task" in result
        assert result["n_seeds"] == 2
        # Uses first report's format
        assert result["task"] == "glue/sst2"
    
    def test_aggregation_preserves_metadata(self):
        """Test that aggregation preserves new metadata fields."""
        reports = []
        for seed in [42, 43]:
            report = self.base_report_data.copy()
            report["task"] = "glue/sst2"
            report["git_commit"] = "abc123def456"
            report["config_metadata"] = {
                "primary_metric_key": "eval_accuracy",
                "config_hash": "hash123",
                "embedded_config": {"model": {"name": "test"}}
            }
            report["env"]["seed"] = seed
            report["probe"]["accuracy"] = 0.85 + seed * 0.001
            reports.append(report)
        
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, self.temp_dir)
        
        # Should preserve metadata from first report
        assert "git_commit" in result
        assert result["git_commit"] == "abc123def456"
        assert "config_metadata" in result
        assert result["config_metadata"]["primary_metric_key"] == "eval_accuracy"
    
    def test_aggregation_statistical_calculations(self):
        """Test that statistical calculations work correctly."""
        # Create reports with known variance
        reports = []
        probe_accuracies = [0.80, 0.85, 0.90]  # Known values
        
        for i, accuracy in enumerate(probe_accuracies):
            import copy
            report = copy.deepcopy(self.base_report_data)
            report["task"] = "glue/sst2"
            report["env"]["seed"] = 42 + i
            report["probe"]["accuracy"] = accuracy
            report["compressed"]["uniform_median"]["accuracy"] = accuracy - 0.02
            reports.append(report)
        
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, self.temp_dir)
        
        # Check statistical calculations
        probe_stats = result["probe"]["accuracy"]
        expected_mean = float(np.mean(probe_accuracies))
        expected_std = float(np.std(probe_accuracies))
        
        # Check that we get reasonable statistics (allow some floating point variance)
        assert abs(probe_stats["mean"] - expected_mean) < 1e-6
        assert abs(probe_stats["std"] - expected_std) < 1e-6
        # Values might be in different order, so check the content
        assert sorted(probe_stats["values"]) == sorted(probe_accuracies)
    
    def test_aggregation_empty_reports_error(self):
        """Test that aggregation with empty reports raises appropriate error."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        with self.assertRaises(ValueError) as cm:
            create_multi_seed_aggregated_report([], config, self.temp_dir)
        self.assertIn("No seed reports provided", str(cm.exception))
    
    def test_aggregation_single_report(self):
        """Test aggregation with single report."""
        report = self.base_report_data.copy()
        report["task"] = "glue/sst2"
        report["env"]["seed"] = 42
        
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report([report], config, self.temp_dir)
        
        assert result["n_seeds"] == 1
        assert result["probe"]["accuracy"]["mean"] == 0.85
        assert result["probe"]["accuracy"]["std"] == 0.0  # No variance with single value
        assert result["summary"]["statistical_power"] == "limited"  # < 3 seeds
    
    def test_variant_aggregation_with_mixed_results(self):
        """Test variant aggregation when some variants are missing in some seeds."""
        reports = []
        
        # First report: has uniform_median only
        report1 = self.base_report_data.copy()
        report1["task"] = "glue/sst2"
        report1["env"]["seed"] = 42
        # Create a deep copy for compressed data
        report1["compressed"] = {
            "uniform_median": {
                "rank": 4,
                "params": 2560,
                "accuracy": 0.82,
                "delta_vs_probe": -0.03,
                "param_reduction": 0.75,
                "verdict": "PASS"
            }
        }
        reports.append(report1)
        
        # Second report: has both uniform_median and uniform_p90
        report2 = self.base_report_data.copy()
        report2["task"] = "glue/sst2" 
        report2["env"]["seed"] = 43
        report2["compressed"] = {
            "uniform_median": {
                "rank": 4,
                "params": 2560,
                "accuracy": 0.83,
                "delta_vs_probe": -0.02,
                "param_reduction": 0.75,
                "verdict": "PASS"
            },
            "uniform_p90": {
                "rank": 8,
                "params": 5120,
                "accuracy": 0.84,
                "delta_vs_probe": -0.01,
                "param_reduction": 0.5,
                "verdict": "PASS"
            }
        }
        reports.append(report2)
        
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, self.temp_dir)
        
        # Should handle both variants
        compressed = result["compressed"]
        assert "uniform_median" in compressed  # Present in both reports
        assert "uniform_p90" in compressed     # Present in only one report
        
        # uniform_median should have 2 data points
        assert len(compressed["uniform_median"]["accuracy"]["values"]) == 2
        
        # uniform_p90 should have 1 data point (from second report only)
        assert len(compressed["uniform_p90"]["accuracy"]["values"]) == 1


if __name__ == "__main__":
    unittest.main()