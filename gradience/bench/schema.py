"""
Compression variant schema and canonical accessors.

This module defines the expected data structure contracts for compression variants
and provides canonical accessor functions to prevent schema-related bugs.
"""

from typing import Dict, Tuple, Any


def get_variant_payload(compression_config: dict) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns (rank_pattern, alpha_pattern) for a compressed variant.
    
    Canonical location is top-level; falls back to nested variant_config for backwards compatibility.
    This accessor abstracts away the storage location and provides a stable interface.
    
    Args:
        compression_config: Compression configuration dictionary
        
    Returns:
        Tuple of (rank_pattern, alpha_pattern) dictionaries
        
    Raises:
        TypeError: If rank_pattern or alpha_pattern are not dictionaries
        KeyError: If compression_config is malformed
        
    Example:
        >>> config = {"rank_pattern": {"layer.0": 4}, "alpha_pattern": {"layer.0": 4}}
        >>> rank_pattern, alpha_pattern = get_variant_payload(config)
        >>> print(rank_pattern)  # {"layer.0": 4}
    """
    if not isinstance(compression_config, dict):
        raise TypeError(f"compression_config must be dict, got {type(compression_config)}")
    
    # Extract variant_config if it exists (backwards compatibility)
    variant_config = compression_config.get("config", {})
    if not isinstance(variant_config, dict):
        variant_config = {}

    # Canonical location: top-level in compression_config
    # Fallback: nested in variant_config (backwards compatibility)
    rank_pattern = (
        compression_config.get("rank_pattern")
        or variant_config.get("rank_pattern")
        or {}
    )

    alpha_pattern = (
        compression_config.get("alpha_pattern")
        or variant_config.get("alpha_pattern")
        or {}
    )

    # Type validation
    if not isinstance(rank_pattern, dict):
        raise TypeError(f"rank_pattern must be dict, got {type(rank_pattern)}")
    if not isinstance(alpha_pattern, dict):
        raise TypeError(f"alpha_pattern must be dict, got {type(alpha_pattern)}")

    return rank_pattern, alpha_pattern


def validate_compression_config(compression_config: dict) -> None:
    """
    Validates a compression configuration for required fields and types.
    
    Args:
        compression_config: Compression configuration to validate
        
    Raises:
        TypeError: If required fields have wrong types
        KeyError: If required fields are missing
        ValueError: If field values are invalid
    """
    if not isinstance(compression_config, dict):
        raise TypeError(f"compression_config must be dict, got {type(compression_config)}")
    
    # Required top-level fields
    required_fields = ["variant", "status"]
    for field in required_fields:
        if field not in compression_config:
            raise KeyError(f"Missing required field: {field}")
    
    # Validate variant field
    variant = compression_config["variant"]
    if not isinstance(variant, str):
        raise TypeError(f"variant must be str, got {type(variant)}")
    
    # Validate status field
    status = compression_config["status"]
    valid_statuses = ["ready", "failed", "skipped"]
    if status not in valid_statuses:
        raise ValueError(f"status must be one of {valid_statuses}, got {status}")
    
    # If this is a per-layer variant, validate patterns exist
    if variant in ["per_layer", "per_layer_shuffled"]:
        rank_pattern, alpha_pattern = get_variant_payload(compression_config)
        if not rank_pattern:
            raise ValueError(f"per-layer variant {variant} must have non-empty rank_pattern")
        if not alpha_pattern:
            raise ValueError(f"per-layer variant {variant} must have non-empty alpha_pattern")


def get_variant_rank(compression_config: dict) -> int:
    """
    Returns the effective rank for a compression variant.
    
    For uniform variants, returns actual_r.
    For per-layer variants, returns the most common rank from rank_pattern.
    
    Args:
        compression_config: Compression configuration dictionary
        
    Returns:
        Effective rank as integer
        
    Raises:
        KeyError: If required fields are missing
        ValueError: If rank cannot be determined
    """
    variant = compression_config.get("variant", "")
    
    if variant in ["per_layer", "per_layer_shuffled"]:
        # For per-layer variants, use the most common rank
        rank_pattern, _ = get_variant_payload(compression_config)
        if not rank_pattern:
            raise ValueError(f"Cannot determine rank for {variant}: empty rank_pattern")
        
        # Count rank frequencies
        from collections import Counter
        rank_counts = Counter(rank_pattern.values())
        most_common_rank = rank_counts.most_common(1)[0][0]
        return most_common_rank
    else:
        # For uniform variants, use actual_r
        actual_r = compression_config.get("actual_r")
        if actual_r is None:
            raise KeyError(f"Missing actual_r for uniform variant {variant}")
        return actual_r