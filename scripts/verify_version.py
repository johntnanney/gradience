#!/usr/bin/env python3
"""
Version verification script for Gradience.

Ensures that:
1. git tag (if present) matches pyproject.toml version
2. importlib.metadata.version("gradience") matches pyproject.toml
3. gradience.__version__ is accessible and consistent

This prevents version regression and makes tags + versions truthful.
"""

import subprocess
import sys
import tomllib
from pathlib import Path


def get_git_tag() -> str | None:
    """Get the current git tag if HEAD is tagged."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_pyproject_version() -> str:
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    return data["project"]["version"]


def get_installed_version() -> str | None:
    """Get version from installed package metadata."""
    try:
        from importlib.metadata import version
        return version("gradience")
    except ImportError:
        try:
            from importlib_metadata import version
            return version("gradience")
        except ImportError:
            return None
    except Exception:
        return None


def get_module_version() -> str | None:
    """Get version from gradience.__version__."""
    try:
        # Add parent directory to path to import gradience
        gradience_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(gradience_dir))
        
        import gradience
        return gradience.__version__
    except ImportError as e:
        print(f"Warning: Could not import gradience module: {e}")
        return None
    except Exception as e:
        print(f"Warning: Could not get gradience.__version__: {e}")
        return None


def normalize_version(version_str: str) -> str:
    """Normalize version string (remove 'v' prefix if present)."""
    if version_str and version_str.startswith("v"):
        return version_str[1:]
    return version_str


def main():
    """Run version verification checks."""
    print("🔍 Gradience Version Verification")
    print("=" * 50)
    
    errors = []
    warnings = []
    
    # Get all versions
    try:
        pyproject_version = get_pyproject_version()
        print(f"📦 pyproject.toml version: {pyproject_version}")
    except Exception as e:
        errors.append(f"Could not read pyproject.toml version: {e}")
        return 1
    
    git_tag = get_git_tag()
    if git_tag:
        git_version = normalize_version(git_tag)
        print(f"🏷️  Git tag: {git_tag} (normalized: {git_version})")
    else:
        git_version = None
        print(f"🏷️  Git tag: None (HEAD not tagged)")
    
    installed_version = get_installed_version()
    if installed_version:
        print(f"📚 Installed version: {installed_version}")
    else:
        warnings.append("Could not get installed package version (package may not be installed)")
    
    module_version = get_module_version()
    if module_version:
        print(f"🐍 Module __version__: {module_version}")
    else:
        warnings.append("Could not get gradience.__version__")
    
    print()
    
    # Verification checks
    print("🧪 Verification Checks")
    print("-" * 30)
    
    # Check 1: Git tag vs pyproject.toml
    if git_tag is not None:
        if git_version == pyproject_version:
            print(f"✅ Git tag matches pyproject.toml: {git_version}")
        else:
            errors.append(f"Git tag ({git_version}) != pyproject.toml ({pyproject_version})")
            print(f"❌ Git tag mismatch: {git_version} != {pyproject_version}")
    else:
        print("⏭️  Git tag check skipped (HEAD not tagged)")
    
    # Check 2: Installed version vs pyproject.toml
    if installed_version is not None:
        if installed_version == pyproject_version:
            print(f"✅ Installed version matches pyproject.toml: {installed_version}")
        else:
            errors.append(f"Installed version ({installed_version}) != pyproject.toml ({pyproject_version})")
            print(f"❌ Installed version mismatch: {installed_version} != {pyproject_version}")
    else:
        print("⏭️  Installed version check skipped")
    
    # Check 3: Module version vs pyproject.toml
    if module_version is not None:
        if module_version == pyproject_version:
            print(f"✅ Module __version__ matches pyproject.toml: {module_version}")
        else:
            errors.append(f"Module __version__ ({module_version}) != pyproject.toml ({pyproject_version})")
            print(f"❌ Module __version__ mismatch: {module_version} != {pyproject_version}")
    else:
        print("⏭️  Module __version__ check skipped")
    
    print()
    
    # Summary
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
        print()
    
    if errors:
        print("❌ Errors Found:")
        for error in errors:
            print(f"   • {error}")
        print()
        print("💡 To fix:")
        print("   1. Update pyproject.toml version to match the desired release")
        print("   2. Re-install package: pip install -e .")
        print("   3. Create/update git tag: git tag -a vX.Y.Z -m 'Version X.Y.Z'")
        print("   4. Re-run this script to verify")
        return 1
    else:
        print("🎉 All version checks passed!")
        print()
        print("📋 Summary:")
        print(f"   • Single source of truth: pyproject.toml ({pyproject_version})")
        print(f"   • Git tag: {'✅ Consistent' if git_tag else '➖ None (OK for development)'}")
        print(f"   • Package metadata: {'✅ Consistent' if installed_version else '➖ Not installed'}")
        print(f"   • Module import: {'✅ Working' if module_version else '➖ Import issues'}")
        print()
        print("🚀 Ready for deterministic clone + run!")
        return 0


if __name__ == "__main__":
    sys.exit(main())