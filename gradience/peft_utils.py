"""
Utilities for working with PEFT (Parameter Efficient Fine-Tuning) configurations.
"""

def normalize_peft_module_name(name: str) -> str:
    """
    Normalize module names for PEFT compatibility.
    
    PEFT expects module names without wrapper prefixes that are added by 
    various model loading mechanisms. This function strips common prefixes
    to ensure rank_pattern and alpha_pattern keys match what PEFT sees.
    
    Args:
        name: Module name that may include wrapper prefixes
        
    Returns:
        Normalized module name compatible with PEFT
        
    Examples:
        >>> normalize_peft_module_name("base_model.model.bert.encoder.layer.0.attention.self.query")
        "bert.encoder.layer.0.attention.self.query"
        >>> normalize_peft_module_name("model.transformer.h.0.attn.c_attn")  
        "transformer.h.0.attn.c_attn"
        >>> normalize_peft_module_name("distilbert.transformer.layer.0.attention.q_lin")
        "distilbert.transformer.layer.0.attention.q_lin"
    """
    for prefix in ("base_model.model.", "base_model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def normalize_rank_pattern(rank_pattern: dict) -> dict:
    """
    Normalize all module names in a rank_pattern for PEFT compatibility.
    
    Args:
        rank_pattern: Dictionary mapping module names to ranks
        
    Returns:
        Dictionary with normalized module names
    """
    return {
        normalize_peft_module_name(name): rank
        for name, rank in rank_pattern.items()
    }


def normalize_alpha_pattern(alpha_pattern: dict) -> dict:
    """
    Normalize all module names in an alpha_pattern for PEFT compatibility.
    
    Args:
        alpha_pattern: Dictionary mapping module names to alpha values
        
    Returns:
        Dictionary with normalized module names
    """
    return {
        normalize_peft_module_name(name): alpha
        for name, alpha in alpha_pattern.items()
    }


def create_complete_rank_pattern(
    partial_rank_pattern: dict, 
    audit_layers: list, 
    default_rank: int
) -> dict:
    """
    Create a complete rank pattern that includes ALL target modules.
    
    PEFT's rank_pattern has compatibility issues when not all modules are explicitly listed.
    This function ensures every target module from the audit has an explicit rank.
    
    Current approach (conservative, for PEFT 0.18.1 compatibility):
    - Creates explicit entries for ALL modules, not just overrides
    - Works around PEFT issues where default_r > some pattern ranks fails
    - Results in larger patterns but guarantees correct application
    
    Future cleaner approach (once PEFT improves):
    - Use max(pattern_ranks) as default_r
    - Only include modules that differ from default
    - Would reduce pattern size significantly
    
    Args:
        partial_rank_pattern: Partial rank pattern (typically from audit per-layer suggestions)
        audit_layers: List of audit layer objects with 'name' field
        default_rank: Rank to use for modules not in partial_rank_pattern
        
    Returns:
        Complete rank pattern with normalized module names
    """
    # Normalize the partial pattern first
    normalized_partial = normalize_rank_pattern(partial_rank_pattern)
    
    # Create complete pattern by adding default ranks for missing modules
    complete_pattern = {}
    for layer in audit_layers:
        module_name = normalize_peft_module_name(layer["name"])
        if module_name in normalized_partial:
            # Use the specified rank
            complete_pattern[module_name] = normalized_partial[module_name]
        else:
            # Use default rank for modules not explicitly specified
            complete_pattern[module_name] = default_rank
    
    return complete_pattern


def create_complete_alpha_pattern(
    partial_alpha_pattern: dict,
    audit_layers: list,
    default_alpha: int
) -> dict:
    """
    Create a complete alpha pattern that includes ALL target modules.
    
    Args:
        partial_alpha_pattern: Partial alpha pattern (typically from audit per-layer suggestions)
        audit_layers: List of audit layer objects with 'name' field
        default_alpha: Alpha to use for modules not in partial_alpha_pattern
        
    Returns:
        Complete alpha pattern with normalized module names
    """
    # Normalize the partial pattern first
    normalized_partial = normalize_alpha_pattern(partial_alpha_pattern)
    
    # Create complete pattern by adding default alphas for missing modules
    complete_pattern = {}
    for layer in audit_layers:
        module_name = normalize_peft_module_name(layer["name"])
        if module_name in normalized_partial:
            # Use the specified alpha
            complete_pattern[module_name] = normalized_partial[module_name]
        else:
            # Use default alpha for modules not explicitly specified
            complete_pattern[module_name] = default_alpha
    
    return complete_pattern


def check_heterogeneous_ranks(adapter_weights_path: str, allowed_ranks: list) -> dict:
    """
    Regression check for per-layer rank patterns.
    
    Verifies that heterogeneous ranks are actually applied in the trained adapter,
    preventing silent fallbacks to uniform ranks.
    
    Args:
        adapter_weights_path: Path to adapter_model.safetensors
        allowed_ranks: List of allowed rank values from config
        
    Returns:
        Dict with check results:
        {
            "passed": bool,
            "unique_ranks": set,
            "rank_histogram": dict,
            "total_modules": int,
            "reason": str or None
        }
    """
    try:
        from safetensors.torch import load_file
        from pathlib import Path
        
        # Load safetensors weights
        tensors = load_file(Path(adapter_weights_path))
        
        # Extract ranks from lora_A weights
        ranks = []
        for key, tensor in tensors.items():
            if 'lora_A' in key:
                rank = tensor.shape[0]
                ranks.append(rank)
        
        if not ranks:
            return {
                "passed": False,
                "unique_ranks": [],
                "rank_histogram": {},
                "total_modules": 0,
                "reason": "No lora_A weights found in adapter"
            }
        
        unique_ranks = set(ranks)
        rank_histogram = {rank: ranks.count(rank) for rank in unique_ranks}
        
        # Check invariants
        if len(unique_ranks) < 2:
            return {
                "passed": False,
                "unique_ranks": sorted(list(unique_ranks)),
                "rank_histogram": rank_histogram,
                "total_modules": len(ranks),
                "reason": f"Only {len(unique_ranks)} unique rank(s) found: {sorted(unique_ranks)}. Expected at least 2 distinct ranks for per-layer compression"
            }
        
        # Check all ranks are in allowed set
        unexpected_ranks = unique_ranks - set(allowed_ranks)
        if unexpected_ranks:
            return {
                "passed": False,
                "unique_ranks": sorted(list(unique_ranks)),
                "rank_histogram": rank_histogram,
                "total_modules": len(ranks),
                "reason": f"Unexpected ranks found: {sorted(unexpected_ranks)}. Must be subset of allowed_ranks: {allowed_ranks}"
            }
        
        return {
            "passed": True,
            "unique_ranks": sorted(list(unique_ranks)),  # Convert set to sorted list for JSON
            "rank_histogram": rank_histogram,
            "total_modules": len(ranks),
            "reason": None
        }
        
    except Exception as e:
        return {
            "passed": False,
            "unique_ranks": [],
            "rank_histogram": {},
            "total_modules": 0,
            "reason": f"Failed to check ranks: {str(e)}"
        }