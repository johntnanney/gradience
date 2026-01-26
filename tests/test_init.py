"""
Minimal test to ensure pytest can run.
"""

def test_python_version():
    """Test that Python version is supported."""
    import sys
    assert sys.version_info >= (3, 10)

def test_basic_imports():
    """Test that core gradience imports work."""
    import gradience
    assert hasattr(gradience, '__version__')
    
def test_pytest_works():
    """Sanity check that pytest itself is working."""
    assert True