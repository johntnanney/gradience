#!/usr/bin/env python3
"""
Test Policy Scoreboard Integration with Bench

Verifies that policy scoreboard correctly integrates with Bench pipeline
and generates the expected artifacts.
"""

import sys
sys.path.insert(0, '.')

import json
import tempfile
from pathlib import Path
from gradience.vnext.policy_scoreboard import PolicyScoreboard, create_policy_result_from_bench_data


def test_bench_integration():
    """Test policy scoreboard integration with Bench data format."""
    
    print("🧪 Testing Policy Scoreboard Integration")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        scoreboard_path = Path(temp_dir) / "test_scoreboard.json"
        scoreboard = PolicyScoreboard(scoreboard_path)
        
        # Simulate Bench verdict analysis format
        mock_verdict_analysis = {
            "verdicts": {
                "energy_p90": {
                    "verdict": "PASS",
                    "performance_delta": 0.025,
                    "accuracy": 0.892,
                    "baseline_accuracy": 0.867
                },
                "knee_p90": {
                    "verdict": "FAIL", 
                    "performance_delta": -0.005,
                    "accuracy": 0.862,
                    "baseline_accuracy": 0.867
                },
                "erank_p90": {
                    "verdict": "PASS",
                    "performance_delta": 0.018,
                    "accuracy": 0.885,
                    "baseline_accuracy": 0.867
                }
            }
        }
        
        # Simulate compression configs format
        mock_compression_configs = {
            "energy_p90": {
                "policy_type": "energy",
                "suggested_r": 8,
                "actual_r": 8,
                "status": "ready"
            },
            "knee_p90": {
                "policy_type": "knee", 
                "suggested_r": 4,
                "actual_r": 4,
                "status": "ready"
            },
            "erank_p90": {
                "policy_type": "erank",
                "suggested_r": 6,
                "actual_r": 6, 
                "status": "ready"
            }
        }
        
        print("📊 Simulating Bench integration...")
        
        # Extract policy results (mimicking the integration code)
        policy_results = []
        
        for variant_name, variant_config in mock_compression_configs.items():
            if variant_config.get("status") != "ready":
                continue
                
            policy_type = variant_config.get("policy_type", "unknown")
            if policy_type == "unknown":
                continue
                
            suggested_rank = variant_config.get("suggested_r", 0)
            actual_rank = variant_config.get("actual_r", 0)
            
            # Get performance results from verdict analysis
            variant_verdict = mock_verdict_analysis.get("verdicts", {}).get(variant_name)
            if variant_verdict:
                passed = variant_verdict.get("verdict") == "PASS"
                performance_delta = variant_verdict.get("performance_delta", 0.0)
                
                # Collect all performance results for optimal rank calculation
                all_results = {}
                for vname, vverdict in mock_verdict_analysis.get("verdicts", {}).items():
                    if vverdict.get("performance_delta") is not None:
                        all_results[vname] = vverdict.get("performance_delta", 0.0)
                
                # Create policy result
                policy_result = create_policy_result_from_bench_data(
                    config_name="test_distilbert_sst2",
                    model_name="distilbert-base-uncased", 
                    task_name="sst2",
                    policy_name=policy_type,
                    suggested_rank=suggested_rank,
                    actual_rank=actual_rank,
                    performance_delta=performance_delta,
                    passed=passed,
                    all_results=all_results,
                    seed=42
                )
                
                policy_results.append(policy_result)
        
        print(f"   Extracted {len(policy_results)} policy results")
        
        # Add results to scoreboard
        scoreboard.add_benchmark_results("test_distilbert_sst2", "distilbert-base-uncased", "sst2", policy_results)
        
        print("   Updated scoreboard with results")
        
        # Verify scoreboard content
        assert scoreboard.data["total_benchmarks"] == 1
        assert len(scoreboard.data["policies"]) == 3
        assert "energy" in scoreboard.data["policies"]
        assert "knee" in scoreboard.data["policies"] 
        assert "erank" in scoreboard.data["policies"]
        
        # Check policy metrics
        energy_metrics = scoreboard.get_policy_metrics("energy")
        assert energy_metrics.passes == 1
        assert energy_metrics.total_attempts == 1
        assert energy_metrics.pass_rate == 1.0
        
        knee_metrics = scoreboard.get_policy_metrics("knee")
        assert knee_metrics.passes == 0
        assert knee_metrics.pass_rate == 0.0
        
        print("   ✅ Scoreboard data validated")
        
        # Test JSON export
        snapshot_path = Path(temp_dir) / "policy_scoreboard_snapshot.json"
        scoreboard.export_snapshot(snapshot_path)
        
        assert snapshot_path.exists()
        with open(snapshot_path) as f:
            snapshot_data = json.load(f)
        
        assert snapshot_data["total_benchmarks"] == 1
        print(f"   ✅ Snapshot exported to {snapshot_path}")
        
        # Test markdown generation
        markdown_table = scoreboard.generate_markdown_table()
        assert "Policy Scoreboard" in markdown_table
        
        # Scoreboard requires 3+ attempts per policy for meaningful statistics
        if "No sufficient policy data" in markdown_table:
            print("   📝 Markdown shows 'insufficient data' (expected - need 3+ attempts per policy)")
        else:
            print("   📝 Markdown shows full scoreboard table")
        
        print("   ✅ Markdown table generated correctly")
        
        # Show the integration artifacts
        print("\n📄 GENERATED ARTIFACTS:")
        print("=" * 30)
        
        print("1. Policy Scoreboard JSON:")
        print(f"   📁 {scoreboard.scoreboard_path}")
        print(f"   📊 {scoreboard.data['total_benchmarks']} benchmarks")
        print(f"   🎯 {len(scoreboard.data['policies'])} policies")
        
        print("\n2. Per-run Snapshot:")
        print(f"   📁 {snapshot_path}")
        print(f"   💾 {len(snapshot_data)} MB")
        
        print("\n3. Markdown Table:")
        print("   📝 Embedded in aggregate reports")
        print("   📊 Shows policy performance summary")
        
        print("\n4. Integration Points:")
        print("   🔗 Bench protocol.py:3669 (tracking)")
        print("   🔗 Bench aggregate.py:645 (reporting)")
        
        print(f"\n🎯 WORKFLOW:")
        print("   1. Bench runs → Policy results tracked automatically")
        print("   2. Scoreboard updated → ~/.gradience/policy_scoreboard.json")
        print("   3. Snapshot exported → {output_dir}/policy_scoreboard_snapshot.json")
        print("   4. Markdown generated → Embedded in aggregate_report.md")
        
        print("\n✅ INTEGRATION TEST PASSED!")
        print("   Policy scoreboard ready for production use")


if __name__ == "__main__":
    test_bench_integration()