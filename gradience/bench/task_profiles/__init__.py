"""
Task profiles for different model types and evaluation tasks.
"""

from .base import TaskProfile
from .registry import get_task_profile, get_task_profile_from_config


# Lazy imports for heavy dependencies
def GLUESequenceClassificationProfile():
    """Lazy import for GLUE profile."""
    from .seqcls_glue import GLUESequenceClassificationProfile as _Profile

    return _Profile


def GSM8KCausalLMProfile():
    """Lazy import for GSM8K profile."""
    from .gsm8k_causal_lm import GSM8KCausalLMProfile as _Profile

    return _Profile


__all__ = [
    "TaskProfile",
    "GLUESequenceClassificationProfile",
    "GSM8KCausalLMProfile",
    "get_task_profile",
    "get_task_profile_from_config",
]
