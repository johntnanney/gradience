#!/usr/bin/env python3
"""
Gradience API Usage Demo

This script demonstrates the stable gradience.api module for programmatic access
to Gradience functionality without depending on internal implementation details.
"""

import gradience.api as gradience
from pathlib import Path


def demo_audit_api():
    """Demonstrate stable audit API."""
    print("🔍 Running LoRA audit via stable API...")
    
    # Use the tiny example adapter
    adapter_path = "examples/adapters/tiny_lora/"
    
    try:
        result = gradience.audit(
            peft_dir=adapter_path,
            layers=True,
            check=False  # Continue even if CLI has issues
        )
        
        if result.returncode == 0:
            print("✅ Audit completed successfully")
        else:
            print("⚠️  Audit had issues (expected in some environments)")
            
    except Exception as e:
        print(f"❌ Audit failed: {e}")


def demo_artifact_loading():
    """Demonstrate loading canonical artifacts."""
    print("\n📊 Loading bench artifacts via stable API...")
    
    try:
        # Load single-seed bench result
        bench_data = gradience.load_bench_report("examples/bench_artifacts/")
        
        print(f"✅ Loaded bench.json:")
        print(f"   Version: {bench_data['bench_version']}")
        print(f"   Model: {bench_data['model']}")
        print(f"   Task: {bench_data['task']}")
        print(f"   Probe accuracy: {bench_data['probe']['accuracy']:.3f}")
        
        # Extract compression results
        compressed = bench_data.get("compressed", {})
        if compressed:
            for variant, data in compressed.items():
                reduction = data.get("param_reduction", 0)
                print(f"   {variant}: {reduction:.1%} parameter reduction")
        
        # Load multi-seed aggregate
        agg_data = gradience.load_bench_aggregate("examples/bench_artifacts/")
        print(f"\n✅ Loaded bench_aggregate.json:")
        print(f"   Seeds: {agg_data['n_seeds']}")
        print(f"   Statistical power: {agg_data['summary']['statistical_power']}")
        
    except Exception as e:
        print(f"❌ Artifact loading failed: {e}")


def demo_data_classes():
    """Demonstrate API data classes."""
    print("\n📦 Testing API data structures...")
    
    # Create example artifacts structure
    output_dir = Path("example_output")
    
    bench_artifacts = gradience.BenchRunArtifacts(
        output_dir=output_dir,
        bench_json=output_dir / "bench.json",
        bench_md=output_dir / "bench.md"
    )
    
    print(f"✅ BenchRunArtifacts:")
    print(f"   Output dir: {bench_artifacts.output_dir}")
    print(f"   JSON path: {bench_artifacts.bench_json}")
    print(f"   Markdown path: {bench_artifacts.bench_md}")
    
    agg_artifacts = gradience.BenchAggregateArtifacts(
        output_dir=output_dir / "aggregate",
        aggregate_json=output_dir / "aggregate" / "bench_aggregate.json",
        aggregate_md=output_dir / "aggregate" / "bench_aggregate.md"
    )
    
    print(f"\n✅ BenchAggregateArtifacts:")
    print(f"   Output dir: {agg_artifacts.output_dir}")
    print(f"   JSON path: {agg_artifacts.aggregate_json}")


def demo_stable_imports():
    """Demonstrate stable import patterns."""
    print("\n🔗 Testing stable import patterns...")
    
    # Test direct import
    import gradience.api
    print("✅ Direct import: gradience.api")
    
    # Test discoverable import
    import gradience
    api_module = gradience.api
    print("✅ Discoverable import: gradience.api")
    
    # Test key functions exist
    required_functions = [
        "audit", "monitor", "run_bench", "aggregate_bench_runs",
        "load_bench_report", "load_bench_aggregate"
    ]
    
    print("\n📋 Available stable functions:")
    for func_name in required_functions:
        if hasattr(gradience.api, func_name):
            print(f"   ✅ {func_name}")
        else:
            print(f"   ❌ {func_name} MISSING")


def main():
    """Run all API demos."""
    print("🚀 Gradience API Stability Demo")
    print("=" * 50)
    
    # Test stable import patterns
    demo_stable_imports()
    
    # Test audit functionality
    demo_audit_api()
    
    # Test artifact loading
    demo_artifact_loading()
    
    # Test data classes
    demo_data_classes()
    
    print("\n🎯 API demo complete!")
    print("\n💡 Key benefits of using gradience.api:")
    print("   - Stable interfaces across versions")
    print("   - No dependency on internal implementation")
    print("   - Consistent behavior and error handling")
    print("   - Future-proof against refactors")


if __name__ == "__main__":
    main()