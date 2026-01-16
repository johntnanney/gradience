"""
Task profile registry for Bench.
"""

from typing import Dict, Type
from .base import TaskProfile
from .seqcls_glue import GLUESequenceClassificationProfile
from .gsm8k_causal_lm import GSM8KCausalLMProfile


# Registry of available task profiles
TASK_PROFILES: Dict[str, Type[TaskProfile]] = {
    "seqcls_glue": GLUESequenceClassificationProfile,
    "gsm8k_causal_lm": GSM8KCausalLMProfile,
}


def get_task_profile(profile_name: str) -> TaskProfile:
    """
    Get task profile instance by name.
    
    Args:
        profile_name: Name of the task profile (e.g., 'seqcls_glue', 'gsm8k_causal_lm')
        
    Returns:
        Task profile instance
        
    Raises:
        ValueError: If profile name is not registered
    """
    if profile_name not in TASK_PROFILES:
        available = list(TASK_PROFILES.keys())
        raise ValueError(f"Unknown task profile '{profile_name}'. Available profiles: {available}")
    
    return TASK_PROFILES[profile_name]()


def get_task_profile_from_config(cfg: Dict[str, any]) -> TaskProfile:
    """
    Get task profile from configuration.
    
    Args:
        cfg: Bench configuration dictionary
        
    Returns:
        Task profile instance
        
    Raises:
        ValueError: If profile is not specified or not found
    """
    task_config = cfg.get("task", {})
    profile_name = task_config.get("profile")
    
    if not profile_name:
        # Backward compatibility: infer from model type
        model_config = cfg.get("model", {})
        model_type = model_config.get("type", "seqcls")  # Default to sequence classification
        
        if model_type == "causal_lm":
            # For now, assume causal_lm means GSM8K, but this could be extended
            dataset = task_config.get("dataset", "")
            if "gsm8k" in dataset.lower():
                profile_name = "gsm8k_causal_lm"
            else:
                raise ValueError(f"No task profile available for causal_lm with dataset '{dataset}'")
        else:
            # Default to GLUE-style sequence classification
            profile_name = "seqcls_glue"
    
    return get_task_profile(profile_name)