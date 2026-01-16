"""
Task profiles for different model types and evaluation tasks.
"""

from .base import TaskProfile
from .seqcls_glue import GLUESequenceClassificationProfile
from .gsm8k_causal_lm import GSM8KCausalLMProfile
from .registry import get_task_profile, get_task_profile_from_config

__all__ = [
    "TaskProfile",
    "GLUESequenceClassificationProfile", 
    "GSM8KCausalLMProfile",
    "get_task_profile",
    "get_task_profile_from_config",
]