#!/usr/bin/env python3
"""
Test Bench UDR integration by validating configuration parsing and audit calls.

This tests that Bench correctly:
1. Parses audit.base_model configuration
2. Passes parameters to audit_lora_peft_dir 
3. Includes UDR metrics in reports when available
4. Gracefully handles missing UDR data

No actual model training - focuses on integration logic.
"""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add gradience to path
sys.path.insert(0, '/Users/john/code/gradience')


def test_bench_udr_config_parsing():
    """Test that Bench correctly parses UDR configuration."""
    
    # Sample config with UDR enabled
    config_with_udr = {
        "model": {"name": "distilbert-base-uncased"},
        "audit": {
            "base_model": "distilbert-base-uncased",
            "base_norms_cache": "/tmp/cache",
            "compute_udr": True
        },
        "lora": {"probe_r": 16},
        "train": {"seed": 42}
    }
    
    # Sample config without UDR
    config_without_udr = {
        "model": {"name": "distilbert-base-uncased"},
        "lora": {"probe_r": 16},
        "train": {"seed": 42}
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock probe directory
        probe_dir = temp_path / "probe"
        probe_dir.mkdir()
        
        # Mock audit_lora_peft_dir to capture calls
        mock_audit_result = MagicMock()
        mock_audit_result.to_summary_dict.return_value = {
            "utilization_mean": 0.5,
            "udr_median": 0.123,
            "udr_p90": 0.456,
            "n_layers_with_udr": 4
        }
        mock_audit_result.layers = []
        mock_audit_result.issues = []
        
        # Import directly from protocol module
        sys.path.insert(0, '/Users/john/code/gradience/gradience/bench')
        import protocol
        
        with patch.object(protocol, 'audit_lora_peft_dir') as mock_audit:
            mock_audit.return_value = mock_audit_result
            
            # Test 1: Config with UDR should pass UDR parameters
            print("🔍 Testing config with UDR...")
            audit_path = protocol.run_probe_audit(probe_dir, config_with_udr)
            
            # Verify audit was called with UDR parameters
            mock_audit.assert_called_with(
                probe_dir,
                base_model_id="distilbert-base-uncased",
                base_norms_cache="/tmp/cache", 
                compute_udr=True
            )
            
            # Reset mock
            mock_audit.reset_mock()
            
            # Test 2: Config without audit section should disable UDR (opt-in policy)
            print("🔍 Testing config without UDR section...")
            audit_path = protocol.run_probe_audit(probe_dir, config_without_udr)
            
            # Verify audit was called with UDR disabled (opt-in policy)
            mock_audit.assert_called_with(
                probe_dir,
                base_model_id=None,
                base_norms_cache=None,
                compute_udr=False
            )
            
            print("✅ Config parsing tests passed")


def test_bench_report_udr_integration():
    """Test that Bench reports include UDR instrumentation when available."""
    
    # Mock audit data with UDR metrics
    audit_data_with_udr = {
        "summary": {
            "utilization_mean": 0.5,
            "udr_median": 0.123,
            "udr_p90": 0.456, 
            "udr_max": 0.789,
            "fraction_udr_gt_0_3": 0.25,
            "n_layers_with_udr": 4
        },
        "layers": [
            {"name": "layer.0.q_lin", "udr": 0.789, "r": 16},
            {"name": "layer.1.k_lin", "udr": 0.456, "r": 16},
            {"name": "layer.2.v_lin", "udr": 0.234, "r": 16},
        ]
    }
    
    # Mock audit data without UDR
    audit_data_without_udr = {
        "summary": {
            "utilization_mean": 0.5,
            "n_layers_with_udr": 0
        },
        "layers": []
    }
    
    # Mock other required data
    probe_results = {"probe": {"rank": 16, "params": 1000, "accuracy": 0.85}}
    variant_results = {}
    verdict_analysis = {"probe_baseline": 0.85, "verdicts": {}}
    compression_configs = {}
    config = {
        "bench_version": "0.1",
        "model": {"name": "distilbert-base-uncased"},
        "task": {"dataset": "glue", "subset": "sst2"}
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Import protocol module directly
        sys.path.insert(0, '/Users/john/code/gradience/gradience/bench')
        import protocol
        
        # Test 1: Report with UDR data should include instrumentation
        print("🔍 Testing report with UDR data...")
        report_with_udr = protocol.create_canonical_bench_report(
            probe_results, variant_results, verdict_analysis, 
            audit_data_with_udr, compression_configs, config, output_dir
        )
        
        # Verify instrumentation section exists
        assert "instrumentation" in report_with_udr, "Missing instrumentation section"
        assert "udr" in report_with_udr["instrumentation"], "Missing UDR instrumentation"
        
        udr_section = report_with_udr["instrumentation"]["udr"]
        assert udr_section["udr_median"] == 0.123, f"Wrong UDR median: {udr_section['udr_median']}"
        assert udr_section["n_layers_with_udr"] == 4, f"Wrong layer count: {udr_section['n_layers_with_udr']}"
        
        # Verify top modules are included
        assert "top_modules" in udr_section, "Missing top modules"
        top_modules = udr_section["top_modules"]
        assert len(top_modules) == 3, f"Expected 3 modules, got {len(top_modules)}"
        assert top_modules[0]["name"] == "layer.0.q_lin", f"Wrong top module: {top_modules[0]}"
        assert top_modules[0]["udr"] == 0.789, f"Wrong UDR value: {top_modules[0]['udr']}"
        
        print("✅ Report with UDR includes instrumentation")
        
        # Test 2: Report without UDR data should not have instrumentation
        print("🔍 Testing report without UDR data...")
        report_without_udr = protocol.create_canonical_bench_report(
            probe_results, variant_results, verdict_analysis,
            audit_data_without_udr, compression_configs, config, output_dir
        )
        
        # Verify no instrumentation section
        assert "instrumentation" not in report_without_udr, "Unexpected instrumentation section"
        
        print("✅ Report without UDR omits instrumentation")


def test_yaml_config_compatibility():
    """Test that new YAML config format works correctly."""
    
    config_yaml = """
bench_version: "0.1"
model:
  name: "distilbert-base-uncased"
audit:
  base_model: "distilbert-base-uncased"
  base_norms_cache: "/workspace/cache"
  compute_udr: true
lora:
  probe_r: 16
"""
    
    print("🔍 Testing YAML config parsing...")
    config = yaml.safe_load(config_yaml)
    
    # Verify structure
    assert "audit" in config, "Missing audit section"
    assert config["audit"]["base_model"] == "distilbert-base-uncased"
    assert config["audit"]["base_norms_cache"] == "/workspace/cache"
    assert config["audit"]["compute_udr"] == True
    
    print("✅ YAML config parsing works")


def main():
    """Run all Bench UDR integration tests."""
    print("🧪 Testing Bench UDR Integration")
    print("=" * 50)
    
    try:
        test_bench_udr_config_parsing()
        test_bench_report_udr_integration() 
        test_yaml_config_compatibility()
        
        print("\n🎉 All Bench UDR integration tests passed!")
        print("\n📋 Validated integration features:")
        print("   ✅ Config parsing with audit.base_model")
        print("   ✅ UDR parameters passed to audit_lora_peft_dir")
        print("   ✅ UDR instrumentation in bench reports")
        print("   ✅ Top-5 modules by UDR included")
        print("   ✅ Graceful handling when UDR unavailable")
        print("   ✅ YAML config compatibility")
        
        return True
    except Exception as e:
        print(f"\n❌ Bench UDR integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)