#!/usr/bin/env python3
"""
Dependency doctor for Gradience bench - checks what's installed and what's missing.
Provides exact install commands without stack traces.
"""

import importlib.util
import sys
from typing import Dict, List, Tuple


def check_module(module_name: str) -> bool:
    """Check if a module is available for import."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def get_module_version(module_name: str) -> str:
    """Get the version of an installed module."""
    try:
        module = __import__(module_name)
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'version'):
            return module.version
        elif hasattr(module, 'VERSION'):
            return module.VERSION
        else:
            return "installed"
    except:
        return "not found"


def check_dependencies() -> Dict[str, Dict[str, any]]:
    """Check all Gradience dependencies and categorize them."""
    
    deps = {
        "core": {
            "torch": ("PyTorch", True),
            "numpy": ("NumPy", True),
            "yaml": ("PyYAML", True),
        },
        "hf": {
            "transformers": ("Transformers", False),
            "peft": ("PEFT", False),
            "accelerate": ("Accelerate", False),
            "safetensors": ("Safetensors", False),
        },
        "bench": {
            "transformers": ("Transformers", False),
            "peft": ("PEFT", False),
            "datasets": ("Datasets", False),
            "safetensors": ("Safetensors", False),
            "accelerate": ("Accelerate", False),
        },
        "dev": {
            "pytest": ("Pytest", False),
            "build": ("Build", False),
            "ruff": ("Ruff", False),
            "mypy": ("MyPy", False),
        }
    }
    
    results = {}
    for category, modules in deps.items():
        results[category] = {}
        for module, (display_name, is_required) in modules.items():
            installed = check_module(module)
            version = get_module_version(module) if installed else None
            results[category][module] = {
                "display_name": display_name,
                "installed": installed,
                "version": version,
                "required": is_required
            }
    
    return results


def get_missing_for_command(command: str, results: Dict) -> List[str]:
    """Get list of missing modules for a specific command."""
    if command == "bench":
        missing = []
        for module, info in results["bench"].items():
            if not info["installed"]:
                missing.append(info["display_name"])
        return missing
    elif command == "hf":
        missing = []
        for module, info in results["hf"].items():
            if not info["installed"]:
                missing.append(info["display_name"])
        return missing
    return []


def main():
    """Main dependency doctor function."""
    print("🩺 Gradience Dependency Doctor")
    print("=" * 50)
    print()
    
    results = check_dependencies()
    
    # Check core dependencies
    core_missing = []
    for module, info in results["core"].items():
        if info["installed"]:
            print(f"✅ {info['display_name']:12} {info['version']}")
        else:
            print(f"❌ {info['display_name']:12} MISSING (required)")
            core_missing.append(info['display_name'])
    
    if core_missing:
        print("\n⚠️  Core dependencies missing!")
        print(f"   Install with: pip install gradience")
        print()
    
    # Check HF integration
    print("\n📦 HuggingFace Integration ([hf] extra):")
    hf_missing = []
    hf_modules = set()  # Track unique modules
    for module, info in results["hf"].items():
        if module not in hf_modules:
            hf_modules.add(module)
            if info["installed"]:
                print(f"   ✅ {info['display_name']:12} {info['version']}")
            else:
                print(f"   ⭕ {info['display_name']:12} not installed")
                hf_missing.append(info['display_name'])
    
    # Check bench dependencies
    print("\n🧪 Benchmarking Suite ([bench] extra):")
    bench_missing = []
    bench_modules = set()  # Track unique modules
    for module, info in results["bench"].items():
        if module not in bench_modules:
            bench_modules.add(module)
            if info["installed"]:
                print(f"   ✅ {info['display_name']:12} {info['version']}")
            else:
                print(f"   ⭕ {info['display_name']:12} not installed")
                bench_missing.append(module)
    
    # Check dev dependencies
    print("\n🔧 Development Tools ([dev] extra):")
    dev_missing = []
    for module, info in results["dev"].items():
        if info["installed"]:
            print(f"   ✅ {info['display_name']:12} {info['version']}")
        else:
            print(f"   ⭕ {info['display_name']:12} not installed")
            dev_missing.append(info['display_name'])
    
    # Provide installation commands
    print("\n" + "=" * 50)
    print("📋 Installation Commands:")
    print()
    
    if hf_missing:
        print("To use HuggingFace integration:")
        print('  pip install -e ".[hf]"')
        print()
    
    if bench_missing:
        print("To run benchmarks:")
        print('  pip install -e ".[bench]"')
        print()
    
    if dev_missing:
        print("For development:")
        print('  pip install -e ".[dev]"')
        print()
    
    if hf_missing and bench_missing and dev_missing:
        print("To install everything:")
        print('  pip install -e ".[all]"')
        print()
    
    # Check common use cases
    if not results["hf"]["transformers"]["installed"]:
        print("💡 Tip: Most users want HuggingFace integration:")
        print('   pip install -e ".[hf]"')
    elif not results["bench"]["datasets"]["installed"]:
        print("💡 Tip: To validate compression suggestions:")
        print('   pip install -e ".[bench]"')
    else:
        print("✨ All commonly used dependencies are installed!")
    
    print()
    
    # Return appropriate exit code
    if core_missing:
        return 1  # Core dependencies missing
    else:
        return 0  # Core OK, optional deps are optional


if __name__ == "__main__":
    sys.exit(main())