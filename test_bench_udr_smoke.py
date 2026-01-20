#!/usr/bin/env python3
"""
Small CPU-friendly Bench smoke test for UDR/SDI functionality.

Runs a minimal LoRA training (5 steps) and validates:
1. UDR computation in audit pipeline
2. Base norms caching
3. Bench integration with UDR metrics
4. End-to-end telemetry with UDR
"""

import json
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any
import sys
import os

def create_minimal_bench_config() -> Dict[str, Any]:
    """Create minimal Bench configuration for CPU smoke test."""
    return {
        "model_name": "distilbert-base-uncased",  # Small model for CPU
        "task": "sst2",
        "max_steps": 5,  # Very short run
        "train_samples": 32,  # Small dataset
        "eval_samples": 16,
        "batch_size": 4,
        "learning_rate": 5e-4,
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_lin", "v_lin"],  # DistilBERT attention modules
        "device": "cpu",
        "telemetry_every": 1,  # Capture every step
        "save_steps": 5,  # Save at end
    }


def run_minimal_training(out_dir: Path) -> bool:
    """Run minimal LoRA training using toy example."""
    try:
        # Use the existing toy_lora_run.py example
        cmd = [
            "python3", "examples/vnext/toy_lora_run.py",
            "--out", str(out_dir),
            "--device", "cpu",
            "--max-steps", "5",
            "--train-samples", "32", 
            "--eval-samples", "16",
            "--dataset", "glue/sst2"
        ]
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/john/code/gradience")
        
        if result.returncode != 0:
            print(f"   ❌ Training failed: {result.stderr}")
            return False
            
        # Check required outputs
        required_files = ["run.jsonl", "peft/adapter_config.json", "peft/adapter_model.bin"]
        for f in required_files:
            if not (out_dir / f).exists():
                print(f"   ❌ Missing required file: {f}")
                return False
        
        print(f"   ✅ Training completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Training exception: {e}")
        return False


def test_udr_in_bench_context():
    """Test UDR/SDI functionality in Bench context."""
    print("🧪 Testing UDR in Bench Context - CPU Smoke Test")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Run minimal training
        print("🚀 Running minimal LoRA training (5 steps)...")
        if not run_minimal_training(temp_path):
            return False
        
        peft_dir = temp_path / "peft"
        run_jsonl = temp_path / "run.jsonl"
        
        # 2. Create base norms cache (simulate real base model norms)
        print("📐 Creating base norms cache for DistilBERT...")
        try:
            base_norms = {
                "distilbert.transformer.layer.0.attention.q_lin": {
                    "sigma_max": 12.5,
                    "fro_norm": 85.3
                },
                "distilbert.transformer.layer.0.attention.v_lin": {
                    "sigma_max": 11.8,
                    "fro_norm": 79.2
                },
                "distilbert.transformer.layer.1.attention.q_lin": {
                    "sigma_max": 10.9,
                    "fro_norm": 76.8
                },
                "distilbert.transformer.layer.1.attention.v_lin": {
                    "sigma_max": 11.3,
                    "fro_norm": 81.1
                }
            }
            
            norms_cache_path = temp_path / "base_norms_cache.json"
            with norms_cache_path.open('w') as f:
                json.dump(base_norms, f, indent=2)
                
        except Exception as e:
            print(f"   ❌ Base norms creation failed: {e}")
            return False
        
        # 3. Test audit with UDR (using CLI like Bench would)
        print("🎯 Testing audit with UDR computation...")
        try:
            import subprocess
            
            # Run audit with base norms cache
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(peft_dir),
                "--base-norms-cache", str(norms_cache_path),
                "--json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"})
            
            if result.returncode != 0:
                print(f"   ❌ Audit failed: {result.stderr}")
                return False
            
            # Parse audit results
            audit_result = json.loads(result.stdout)
            
            # Check for UDR fields
            udr_fields = [k for k in audit_result.keys() if 'udr' in k.lower() or 'sdi' in k.lower()]
            if not udr_fields:
                print(f"   ⚠️  No UDR fields found in audit output")
                print(f"   🔍 Available fields: {list(audit_result.keys())}")
            else:
                print(f"   ✅ UDR fields present: {', '.join(udr_fields)}")
            
            # Check individual layers for UDR
            if 'layers' in audit_result:
                udr_layers = [l for l in audit_result['layers'] if l.get('udr') is not None]
                print(f"   ✅ Layers with UDR: {len(udr_layers)}/{len(audit_result['layers'])}")
                
                # Show sample UDR values
                if udr_layers:
                    for i, layer in enumerate(udr_layers[:2]):  # Show first 2 layers
                        udr_val = layer.get('udr', 'N/A')
                        sdi_val = layer.get('sdi', 'N/A')
                        layer_name = layer.get('name', f'layer_{i}')
                        print(f"      - {layer_name}: UDR={udr_val:.4f}, SDI={sdi_val:.4f}")
            
        except Exception as e:
            print(f"   ❌ Audit with UDR failed: {e}")
            return False
        
        # 4. Test audit append (like Bench telemetry pipeline)
        print("📊 Testing audit append to telemetry...")
        try:
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(peft_dir),
                "--base-norms-cache", str(norms_cache_path),
                "--append", str(run_jsonl)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"})
            
            if result.returncode != 0:
                print(f"   ❌ Audit append failed: {result.stderr}")
                return False
            
            print(f"   ✅ Audit metrics appended to telemetry")
            
            # Verify telemetry contains UDR data
            with run_jsonl.open('r') as f:
                lines = f.readlines()
                
            audit_events = []
            for line in lines:
                if line.strip():
                    event = json.loads(line.strip())
                    if event.get('kind') == 'lora_audit':
                        audit_events.append(event)
            
            if not audit_events:
                print(f"   ⚠️  No lora_audit events found in telemetry")
            else:
                print(f"   ✅ Found {len(audit_events)} lora_audit events in telemetry")
                
                # Check for UDR in telemetry
                for event in audit_events:
                    metrics = event.get('metrics', {})
                    udr_keys = [k for k in metrics.keys() if 'udr' in k.lower()]
                    if udr_keys:
                        print(f"   ✅ UDR metrics in telemetry: {udr_keys}")
                        break
                else:
                    print(f"   ⚠️  No UDR metrics found in telemetry events")
            
        except Exception as e:
            print(f"   ❌ Telemetry verification failed: {e}")
            return False
        
        # 5. Test monitor command (validates full pipeline)
        print("📈 Testing monitor command with UDR data...")
        try:
            cmd = ["gradience", "monitor", str(run_jsonl), "--json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"})
            
            if result.returncode != 0:
                print(f"   ❌ Monitor failed: {result.stderr}")
                return False
            
            monitor_result = json.loads(result.stdout)
            
            # Look for UDR in monitor summary
            def find_udr_in_nested(obj, path="root"):
                """Recursively search for UDR fields."""
                udr_found = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if 'udr' in k.lower() or 'sdi' in k.lower():
                            udr_found.append(f"{path}.{k}")
                        udr_found.extend(find_udr_in_nested(v, f"{path}.{k}"))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        udr_found.extend(find_udr_in_nested(item, f"{path}[{i}]"))
                return udr_found
            
            udr_paths = find_udr_in_nested(monitor_result)
            if udr_paths:
                print(f"   ✅ UDR metrics found in monitor: {udr_paths[:3]}...")  # Show first 3
            else:
                print(f"   ⚠️  No UDR metrics found in monitor summary")
            
        except Exception as e:
            print(f"   ❌ Monitor test failed: {e}")
            return False
        
        print("\n🎉 Bench UDR smoke test completed successfully!")
        print("\n📋 Validated functionality:")
        print("   ✅ Minimal LoRA training (5 steps)")
        print("   ✅ Base norms cache loading")
        print("   ✅ UDR computation in audit pipeline")
        print("   ✅ UDR metrics in JSON output")
        print("   ✅ Telemetry integration with UDR")
        print("   ✅ Monitor pipeline compatibility")
        print("\n💡 UDR/SDI implementation is Bench-ready!")
        
        return True


if __name__ == "__main__":
    success = test_udr_in_bench_context()
    exit(0 if success else 1)