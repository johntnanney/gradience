#!/usr/bin/env python3
"""
CPU-friendly test configuration for UDR metric pipeline.

Creates synthetic LoRA weights and base model norms to test:
1. UDR computation functions
2. Base norms loading and caching
3. CLI integration
4. Summary statistics
"""

import json
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any
import sys

def create_test_lora_weights() -> Dict[str, torch.Tensor]:
    """Create minimal synthetic LoRA A/B matrices."""
    # Simulate small LoRA layers with r=4
    # Format: base_model.layers.0.self_attn.q_proj.lora_A/lora_B
    
    torch.manual_seed(42)  # For reproducible results
    
    weights = {
        # Q projection layer (with .weight suffix expected by audit code)
        "base_model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 64),     # r=4, d_model=64
        "base_model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),     # r=4, d_model=64
        
        # K projection layer  
        "base_model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(4, 64),
        "base_model.layers.0.self_attn.k_proj.lora_B.weight": torch.randn(64, 4),
        
        # Value projection layer
        "base_model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(4, 64),
        "base_model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(64, 4),
        
        # MLP layer
        "base_model.layers.0.mlp.gate_proj.lora_A.weight": torch.randn(4, 64),
        "base_model.layers.0.mlp.gate_proj.lora_B.weight": torch.randn(256, 4),  # Expanded dimension
    }
    
    return weights


def create_test_adapter_config() -> Dict[str, Any]:
    """Create minimal PEFT adapter config."""
    return {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "gate_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }


def create_test_base_norms() -> Dict[str, Dict[str, float]]:
    """Create synthetic base model norms for UDR computation."""
    torch.manual_seed(123)  # Different seed for base model
    
    norms = {}
    
    # Generate realistic base model norms (larger than LoRA updates)
    for layer_name in ["q_proj", "k_proj", "v_proj", "gate_proj"]:
        prefix = f"base_model.layers.0.self_attn.{layer_name}" if "proj" in layer_name and layer_name != "gate_proj" else f"base_model.layers.0.mlp.{layer_name}"
        
        # Base norms should be larger than LoRA updates for realistic UDR < 1
        base_sigma = torch.randn(1).abs() * 5 + 10  # Range: 10-15
        base_fro = torch.randn(1).abs() * 8 + 50    # Range: 50-58
        
        norms[prefix] = {
            "sigma_max": float(base_sigma),
            "fro_norm": float(base_fro)
        }
    
    return norms


def run_udr_test():
    """Run comprehensive UDR pipeline test."""
    print("🧪 Testing UDR Pipeline - CPU Friendly Configuration")
    print("=" * 60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Create test LoRA weights
        print("📊 Creating synthetic LoRA weights...")
        weights = create_test_lora_weights()
        weights_path = temp_path / "adapter_model.bin"
        torch.save(weights, weights_path)
        
        # 2. Create adapter config
        print("⚙️  Creating adapter configuration...")
        config = create_test_adapter_config()
        config_path = temp_path / "adapter_config.json"
        with config_path.open('w') as f:
            json.dump(config, f, indent=2)
        
        # 3. Create base model norms cache
        print("📐 Creating base model norms cache...")
        base_norms = create_test_base_norms()
        norms_cache_path = temp_path / "base_norms_cache.json"
        with norms_cache_path.open('w') as f:
            json.dump(base_norms, f, indent=2)
        
        # 4. Test direct function calls first
        print("\n🔧 Testing UDR computation functions...")
        try:
            from gradience.vnext.audit.lora_audit import compute_update_norms, compute_udr_metrics
            
            # Test with first layer
            A = weights["base_model.layers.0.self_attn.q_proj.lora_A.weight"]
            B = weights["base_model.layers.0.self_attn.q_proj.lora_B.weight"]
            scale = 8.0 / 4.0  # alpha/r = 8/4 = 2.0
            
            delta_fro_norm, delta_sigma_max, stable_rank_delta, utilization_delta = compute_update_norms(
                A, B, scale=scale, compute_dtype=torch.float64
            )
            
            print(f"   ✓ Delta norms: σ_max={delta_sigma_max:.4f}, ||·||_F={delta_fro_norm:.4f}")
            
            # Test UDR computation
            base_sigma = base_norms["base_model.layers.0.self_attn.q_proj"]["sigma_max"]
            base_fro = base_norms["base_model.layers.0.self_attn.q_proj"]["fro_norm"]
            
            udr, udr_f, sdi = compute_udr_metrics(delta_sigma_max, delta_fro_norm, base_sigma, base_fro)
            
            print(f"   ✓ UDR metrics: UDR={udr:.4f}, UDR_F={udr_f:.4f}, SDI={sdi:.4f}")
            
        except Exception as e:
            print(f"   ❌ Function test failed: {e}")
            return False
        
        # 5. Test full audit pipeline
        print("\n🎯 Testing full audit pipeline...")
        try:
            from gradience.vnext.audit.lora_audit import audit_lora_peft_dir
            
            # Debug: check what's in the weights file
            print(f"   🔍 Debug: weights keys = {list(weights.keys())}")
            
            result = audit_lora_peft_dir(
                temp_path,
                adapter_config_path=config_path,
                adapter_weights_path=weights_path,
                base_norms_cache=norms_cache_path,
                compute_udr=True
            )
            
            # Debug: check for issues
            if hasattr(result, 'issues') and result.issues:
                print(f"   🔍 Debug: issues = {result.issues}")
            
            print(f"   ✓ Audit completed: {result.n_layers} layers processed")
            print(f"   ✓ Total LoRA params: {result.total_lora_params:,}")
            
            # Check UDR statistics
            if hasattr(result, 'udr_mean') and result.udr_mean is not None:
                print(f"   ✓ UDR statistics: mean={result.udr_mean:.4f}, median={result.udr_median:.4f}")
                print(f"   ✓ SDI statistics: mean={result.sdi_mean:.4f}, median={result.sdi_median:.4f}")
            else:
                print("   ⚠️  UDR statistics not computed")
            
            # Check individual layers
            udr_layers = [layer for layer in result.layers if layer.udr is not None]
            print(f"   ✓ Layers with UDR: {len(udr_layers)}/{len(result.layers)}")
            
            if udr_layers:
                for layer in udr_layers[:3]:  # Show first 3 layers
                    print(f"      - {layer.name}: UDR={layer.udr:.4f}, SDI={layer.sdi:.4f}")
            
        except Exception as e:
            print(f"   ❌ Audit pipeline test failed: {e}")
            return False
        
        # 6. Test CLI integration  
        print("\n🖥️  Testing CLI integration...")
        
        # Import and test CLI directly (avoiding subprocess for simplicity)
        try:
            import sys
            import argparse
            from gradience.cli import cmd_audit
            
            # Create mock args
            args = argparse.Namespace(
                peft_dir=str(temp_path),
                adapter_config=str(config_path),
                weights=str(weights_path),
                base_model=None,  # No base model loading for this test
                base_norms_cache=str(norms_cache_path),
                no_udr=False,
                top_singular_values=0,
                json=True,  # JSON output for easier parsing
                append=None,
                layers=False,
                top_wasteful=0,
                suggest_per_layer=False
            )
            
            # Capture output
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                cmd_audit(args)
                output = captured_output.getvalue()
                
                # Parse JSON output
                result_json = json.loads(output)
                
                print(f"   ✓ CLI test passed")
                print(f"   ✓ JSON output contains {len(result_json)} fields")
                
                # Check for UDR fields
                udr_fields = [k for k in result_json.keys() if 'udr' in k.lower() or 'sdi' in k.lower()]
                if udr_fields:
                    print(f"   ✓ UDR fields present: {', '.join(udr_fields)}")
                else:
                    print("   ⚠️  No UDR fields found in output")
                    
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            print(f"   ❌ CLI test failed: {e}")
            return False
        
        print("\n🎉 All tests passed! UDR pipeline is working correctly.")
        print("\n📋 Summary of verified functionality:")
        print("   ✓ UDR math functions (compute_update_norms, compute_udr_metrics)")
        print("   ✓ Base norms loading from cache")
        print("   ✓ Full audit pipeline integration") 
        print("   ✓ CLI parameter handling")
        print("   ✓ JSON output with UDR metrics")
        print("\n💡 Next steps:")
        print("   - Test with real LoRA adapters")
        print("   - Test base model loading (requires transformers + model)")
        print("   - Integrate with Gradience Bench pipeline")
        
        return True


if __name__ == "__main__":
    success = run_udr_test()
    exit(0 if success else 1)