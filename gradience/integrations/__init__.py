"""
Gradience integrations with ML frameworks.

DEPRECATED: This legacy integration has been moved to docs/legacy/
Use gradience.vnext.integrations instead:

    from gradience.vnext.integrations.hf import GradienceCallback
    trainer.add_callback(GradienceCallback())
"""

import warnings

def __getattr__(name):
    if name == "GradienceCallback":
        warnings.warn(
            "gradience.integrations.huggingface.GradienceCallback is deprecated. "
            "Use gradience.vnext.integrations.hf.GradienceCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(
            "Legacy GradienceCallback has been moved. "
            "Use: from gradience.vnext.integrations.hf import GradienceCallback"
        )
    raise AttributeError(f"module 'gradience.integrations' has no attribute '{name}'")

__all__ = []
