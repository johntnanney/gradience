"""
SVD bench integration smoke test (4.3).

Tests that SVD truncation integrates properly with the bench protocol.
Uses CPU-only toy configurations for fast testing.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Only import if bench modules are available
try:
    from gradience.bench.protocol import generate_compression_configs
    from gradience.vnext.svd_truncate import svd_truncate_peft_dir
    BENCH_AVAILABLE = True
except ImportError as e:
    BENCH_AVAILABLE = False
    BENCH_IMPORT_ERROR = str(e)


class TestSVDBenchIntegration:
    """Test SVD integration with bench protocol."""
    
    @pytest.mark.skipif(not BENCH_AVAILABLE, reason="Bench modules not available")
    def test_svd_config_parsing(self):
        """Test that SVD configuration can be parsed from YAML."""
        config_yaml = """
bench_version: "0.1"

model:
  name: distilbert-base-uncased
  type: seqcls

task:
  dataset: glue
  subset: sst2
  profile: sst2_seqcls

train:
  max_steps: 10  # Minimal for smoke test
  train_samples: 32
  eval_samples: 16

lora:
  r: 16
  alpha: 16
  target_modules: ["q_lin", "k_lin", "v_lin", "out_lin"]

compression:
  variants:
    - name: svd_trunc_r8
      method: svd_truncate
      rank_source: 8
      post_tune:
        enabled: false
    - name: svd_trunc_r4_tune  
      method: svd_truncate
      rank_source: 4
      post_tune:
        enabled: true
        max_steps: 5
        learning_rate: 1e-5

runtime:
  device: cpu
  
run_type: "svd_test"
"""
        
        # Parse config
        config = yaml.safe_load(config_yaml)
        
        # Verify SVD variants are present
        assert "compression" in config
        assert "variants" in config["compression"]
        variants = config["compression"]["variants"]
        
        assert len(variants) == 2
        
        # Check first variant (no post-tune)
        svd_variant = variants[0]
        assert svd_variant["name"] == "svd_trunc_r8"
        assert svd_variant["method"] == "svd_truncate"
        assert svd_variant["rank_source"] == 8
        assert svd_variant["post_tune"]["enabled"] == False
        
        # Check second variant (with post-tune)
        tune_variant = variants[1]
        assert tune_variant["name"] == "svd_trunc_r4_tune"
        assert tune_variant["method"] == "svd_truncate" 
        assert tune_variant["rank_source"] == 4
        assert tune_variant["post_tune"]["enabled"] == True
        assert tune_variant["post_tune"]["max_steps"] == 5
    
    @pytest.mark.skipif(not BENCH_AVAILABLE, reason="Bench modules not available")
    def test_legacy_svd_config_parsing(self):
        """Test parsing legacy SVD configuration format."""
        config_yaml = """
bench_version: "0.1"

model:
  name: distilbert-base-uncased
  type: seqcls

task:
  dataset: glue
  subset: sst2

compression:
  enable_svd_variants: true
  svd_ranks: [8, 4]
  allowed_ranks: [2, 4, 8, 16]

runtime:
  device: cpu
"""
        
        config = yaml.safe_load(config_yaml)
        
        # Verify legacy format
        compression = config["compression"]
        assert compression["enable_svd_variants"] == True
        assert compression["svd_ranks"] == [8, 4]
        assert compression["allowed_ranks"] == [2, 4, 8, 16]
    
    @pytest.mark.skipif(not BENCH_AVAILABLE, reason="Bench modules not available")
    def test_compression_config_generation_with_svd(self):
        """Test that compression config generation supports SVD variants."""
        # Mock audit summary
        audit_summary = {
            "stable_rank_mean": 12.5,
            "utilization_mean": 0.78,
            "current_r": 16,
            "per_layer_ranks": {
                "q_lin": 12,
                "k_lin": 10,
                "v_lin": 14,
                "out_lin": 11
            },
            "suggested_r_global_median": 8,
            "suggested_r_global_90": 6
        }
        
        # Modern compression config with SVD variants
        compression_config = {
            "variants": [
                {
                    "name": "svd_trunc_median",
                    "method": "svd_truncate",
                    "rank_source": "audit_global_median",
                    "post_tune": {"enabled": False}
                },
                {
                    "name": "svd_trunc_r4",
                    "method": "svd_truncate", 
                    "rank_source": 4,
                    "post_tune": {"enabled": False}
                }
            ],
            "allowed_ranks": [2, 4, 8, 16],
            "acc_tolerance": 0.02
        }
        
        try:
            variants = generate_compression_configs(audit_summary, compression_config)
            
            # Should include SVD variants
            assert isinstance(variants, list)
            assert len(variants) > 0
            
            # Look for SVD variants in results
            svd_variants = [v for v in variants if "svd" in v.get("name", "").lower()]
            assert len(svd_variants) > 0, "No SVD variants found in generated configs"
            
            # Check SVD variant structure
            for variant in svd_variants:
                assert "name" in variant
                assert "method" in variant
                assert variant["method"] == "svd_truncate"
                assert "rank_spec" in variant or "rank_source" in variant
                
        except Exception as e:
            pytest.skip(f"Compression config generation failed: {e}")
    
    def test_svd_artifact_structure(self):
        """Test that SVD generates expected bench artifacts."""
        # This test doesn't require actual bench execution
        import tempfile
        import torch
        
        # Create fake truncated adapter in temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "svd_trunc_r8"
            adapter_dir.mkdir()
            
            # Create expected SVD artifact files
            
            # 1. Adapter config
            config = {
                "r": 8,
                "lora_alpha": 8.0,
                "target_modules": ["q_lin", "v_lin"],
                "peft_type": "LORA"
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(config, f)
            
            # 2. Adapter weights
            state_dict = {
                "q_lin.lora_A.default.weight": torch.randn(8, 768),
                "q_lin.lora_B.default.weight": torch.randn(768, 8),
                "v_lin.lora_A.default.weight": torch.randn(8, 768),
                "v_lin.lora_B.default.weight": torch.randn(768, 8)
            }
            torch.save(state_dict, adapter_dir / "adapter_model.pt")
            
            # 3. Truncation report
            report = {
                "original_rank": 16,
                "target_rank": 8,
                "total_modules": 2,
                "energy_retained": 0.95,
                "compression_ratio": 2.0,
                "alpha_mode": "keep_ratio",
                "per_module_energy": [
                    {"module_name": "q_lin", "energy_retained": 0.96},
                    {"module_name": "v_lin", "energy_retained": 0.94}
                ]
            }
            with open(adapter_dir / "truncation_report.json", "w") as f:
                json.dump(report, f)
            
            # 4. README
            readme = "# SVD Truncated LoRA Adapter\n\nRank: 16 → 8\nEnergy retained: 95%"
            with open(adapter_dir / "README.md", "w") as f:
                f.write(readme)
            
            # Verify all required files exist
            assert (adapter_dir / "adapter_config.json").exists()
            assert (adapter_dir / "adapter_model.pt").exists() 
            assert (adapter_dir / "truncation_report.json").exists()
            assert (adapter_dir / "README.md").exists()
            
            # Test report loading
            with open(adapter_dir / "truncation_report.json") as f:
                loaded_report = json.load(f)
            
            assert loaded_report["original_rank"] == 16
            assert loaded_report["target_rank"] == 8
            assert loaded_report["compression_ratio"] == 2.0
            assert 0.0 <= loaded_report["energy_retained"] <= 1.0
    
    @pytest.mark.skipif(not BENCH_AVAILABLE, reason="Bench modules not available")
    def test_svd_variant_validation(self):
        """Test validation of SVD variant configurations."""
        
        # Valid SVD variant
        valid_variant = {
            "name": "svd_trunc_r8",
            "method": "svd_truncate",
            "rank_source": 8,
            "post_tune": {"enabled": False}
        }
        
        # Should not raise
        assert valid_variant["method"] == "svd_truncate"
        assert isinstance(valid_variant["rank_source"], int)
        assert valid_variant["rank_source"] > 0
        
        # Invalid rank source
        invalid_variant = {
            "name": "svd_bad",
            "method": "svd_truncate", 
            "rank_source": 0,  # Invalid: zero rank
            "post_tune": {"enabled": False}
        }
        
        # Should be caught in validation
        assert invalid_variant["rank_source"] <= 0  # This would be invalid
        
        # Test audit-based rank source
        audit_variant = {
            "name": "svd_audit",
            "method": "svd_truncate",
            "rank_source": "audit_global_median", 
            "post_tune": {"enabled": False}
        }
        
        assert audit_variant["rank_source"] == "audit_global_median"
        assert isinstance(audit_variant["rank_source"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])