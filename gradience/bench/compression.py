"""
Compression config generation for bench protocol.

Generates candidate compression variants from audit results,
handles de-duplication, capping, and SVD variant configs.

Extracted from protocol.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gradience.bench._util import round_to_allowed_ranks
from gradience.bench.constants import (
    CONSERVATISM_SCORES,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_POST_TUNE_LR_SCALE,
    DEFAULT_POST_TUNE_STEPS,
    DEFAULT_SEED,
    SHUFFLE_SEED_OFFSET,
)
from gradience.bench.decision_trace import DecisionTrace, create_decision_trace, maybe_add_second_rung_candidates
from gradience.peft_utils import normalize_alpha_pattern, normalize_rank_pattern


def _resolve_policy_rank_source(audit_data: dict[str, Any], rank_source: str) -> float | None:
    """
    Resolve policy-based rank sources like 'audit.rank_suggestions.knee.uniform_p90'.

    Args:
        audit_data: Audit data dict with policy_global_suggestions
        rank_source: Dotted path like "audit.rank_suggestions.POLICY.STATISTIC"

    Returns:
        Rank suggestion as float, or None if path cannot be resolved
    """
    try:
        # Parse the path: audit.rank_suggestions.POLICY.STATISTIC
        parts = rank_source.split(".")
        if len(parts) != 4 or parts[0] != "audit" or parts[1] != "rank_suggestions":
            return None

        policy_name = parts[2]  # e.g., "knee", "erank", "oht", "energy_90"
        statistic = parts[3]  # e.g., "uniform_median", "uniform_p90", "uniform_max"

        # Check if we have policy global suggestions
        policy_suggestions = audit_data.get("policy_global_suggestions")
        if not policy_suggestions:
            return None

        # Check if the requested policy exists
        if policy_name not in policy_suggestions:
            return None

        # Check if the requested statistic exists for that policy
        policy_data = policy_suggestions[policy_name]
        if statistic not in policy_data:
            return None

        return float(policy_data[statistic])

    except (KeyError, ValueError, TypeError):
        return None


def _create_shuffled_rank_pattern(original_rank_pattern: dict[str, int], seed: int) -> dict[str, int]:
    """
    Create a shuffled control by redistributing ranks across different modules.

    This is the key scientific control: if audit-guided placement matters,
    per_layer should outperform per_layer_shuffled. If any heterogeneity
    is enough, they should perform similarly.

    Args:
        original_rank_pattern: Dict mapping module names to ranks
        seed: Random seed for deterministic shuffling

    Returns:
        Dict with same module names but redistributed rank values
    """
    import random

    # Extract module names and rank values
    module_names = list(original_rank_pattern.keys())
    rank_values = list(original_rank_pattern.values())

    # Create deterministic shuffle using seed + offset
    rng = random.Random(seed + SHUFFLE_SEED_OFFSET)

    # Shuffle the rank values while keeping module names fixed
    shuffled_ranks = rank_values.copy()
    rng.shuffle(shuffled_ranks)

    # Recombine: same modules, redistributed ranks
    shuffled_pattern = dict(zip(module_names, shuffled_ranks))

    return shuffled_pattern


def generate_svd_variant_config(
    variant_def: dict[str, Any],
    audit_data: dict[str, Any],
    probe_rank: int,
    lora_config: dict[str, Any],
    allowed_ranks: list[int],
) -> dict[str, Any]:
    """
    Generate a compression config for a single SVD truncation variant.

    Args:
        variant_def: Variant definition from compression.variants
        audit_data: Audit results containing suggested ranks
        probe_rank: Original probe rank
        lora_config: LoRA configuration
        allowed_ranks: Allowed rank values

    Returns:
        Compression config dict compatible with existing variant format
    """
    variant_name = variant_def["name"]
    rank_source = variant_def.get("rank_source")
    post_tune_config = variant_def.get("post_tune", {})

    # Resolve target rank from rank_source
    if rank_source == "audit_global_median":
        suggested_rank = audit_data["suggested_r_global_median"]
    elif rank_source == "audit_global_p90":
        suggested_rank = audit_data["suggested_r_global_90"]
    elif isinstance(rank_source, str) and rank_source.startswith("audit.rank_suggestions."):
        # New policy-based rank sources (Step 7)
        suggested_rank = _resolve_policy_rank_source(audit_data, rank_source)
        if suggested_rank is None:
            return {
                "variant": variant_name,
                "suggested_r": None,
                "actual_r": None,
                "rank_pattern": {},
                "alpha_pattern": {},
                "config": None,
                "status": "skipped",
                "reason": f"Failed to resolve policy rank_source: {rank_source}",
            }
    elif isinstance(rank_source, (int, float)):
        # Direct rank specification
        suggested_rank = int(rank_source)
    else:
        return {
            "variant": variant_name,
            "suggested_r": None,
            "actual_r": None,
            "rank_pattern": {},
            "alpha_pattern": {},
            "config": None,
            "status": "skipped",
            "reason": f"Unsupported rank_source: {rank_source}",
        }

    # Round to allowed ranks
    actual_rank = round_to_allowed_ranks(suggested_rank, allowed_ranks)

    # Safety check: no rank > probe rank
    if actual_rank > probe_rank:
        actual_rank = probe_rank

    # Check if this would be a no-op (no compression)
    if actual_rank >= probe_rank:
        return {
            "variant": variant_name,
            "suggested_r": suggested_rank,
            "actual_r": actual_rank,
            "rank_pattern": {},
            "alpha_pattern": {},
            "config": None,
            "status": "skipped",
            "reason": f"SVD truncation rank r={actual_rank} >= probe rank r={probe_rank} (no compression)",
        }

    # Build variant configuration
    variant_config = {
        "variant": variant_name,
        "suggested_r": suggested_rank,
        "actual_r": actual_rank,
        "rank_pattern": {},
        "alpha_pattern": {},
        "config": {
            **lora_config,
            "probe_r": actual_rank,  # Use truncated rank
            "alpha": actual_rank,  # Preserve alpha=r scaling
        },
        "status": "ready",
        "reason": f"SVD truncation from r={probe_rank} to r={actual_rank}",
        "compression_method": "svd_truncation",
        "source_rank": probe_rank,
        "rank_source": rank_source,  # Store original rank_source for artifact capture
    }

    # Add post-tuning configuration if specified
    if post_tune_config.get("enabled", False):
        variant_config["post_tune"] = {
            "enabled": True,
            "steps": post_tune_config.get("steps", DEFAULT_POST_TUNE_STEPS),
            "lr_scale": post_tune_config.get("lr_scale", DEFAULT_POST_TUNE_LR_SCALE),
        }

    return variant_config


def get_rank_source_from_config(compression_config: dict[str, Any]) -> str:
    """
    Extract the rank_source from a compression config for SVD variants.

    This reconstructs the original rank_source specification from the
    compression config data.
    """
    # Check if this is from the new variants format
    if "rank_source" in compression_config:
        return str(compression_config["rank_source"])

    # Legacy format: try to infer from rank and reason
    actual_r = compression_config.get("actual_r")
    reason = compression_config.get("reason", "")

    # Try to infer from reason string
    if "audit_global_median" in reason.lower() or "median" in reason.lower():
        return "audit_global_median"
    elif "audit_global_90" in reason.lower() or "p90" in reason.lower():
        return "audit_global_p90"
    elif actual_r:
        # Direct rank specification
        return str(actual_r)

    return "unknown"


def generate_compression_configs(
    probe_dir: Path, config: dict[str, Any], fast_mode: bool = True, max_candidates: int = DEFAULT_MAX_CANDIDATES
) -> tuple[dict[str, dict[str, Any]], DecisionTrace]:
    """
    Step 3.4: Generate compression configs from probe audit with candidate control.

    Args:
        probe_dir: Directory containing probe results and audit data
        config: Bench configuration
        fast_mode: If True, only generate energy_p90, knee_p90, erank_p90 (default)
        max_candidates: Maximum number of candidates to generate (default 4)

    Returns dict with compression variant configs:
    - Fast mode: energy_p90, knee_p90, erank_p90 (plus per_layer if different)
    - Full mode: All policy variants, but de-duplicated and capped

    Features:
    1. De-duplicate ranks: if multiple policies suggest same rank, run once
    2. Cap candidates: limit to top N by conservatism score
    3. Fast mode: practitioner-friendly default subset
    """

    # Read compression candidate settings from config (override CLI defaults)
    compression_config = config.get("compression", {})
    config_fast_mode = compression_config.get("fast_mode")
    config_max_candidates = compression_config.get("max_candidates")
    config_candidate_policies = compression_config.get("candidate_policies")

    # Use config values if specified, otherwise fall back to CLI arguments
    if config_fast_mode is not None:
        fast_mode = config_fast_mode
    if config_max_candidates is not None:
        max_candidates = config_max_candidates

    print("\U0001f3af Candidate Control Settings:")
    print(f"   Fast mode: {fast_mode}")
    print(f"   Max candidates: {max_candidates}")
    if config_candidate_policies:
        print(f"   Explicit policies: {config_candidate_policies}")

    # Load audit results
    audit_path = probe_dir / "audit.json"
    print(f"\U0001f4cb Loading audit results from: {audit_path}")
    with open(audit_path) as f:
        audit_data = json.load(f)

    # Get compression configuration (already loaded above, just reference it)
    base_compression_config = config["compression"]
    allowed_ranks = base_compression_config["allowed_ranks"]
    lora_config = config["lora"]
    probe_rank = lora_config["probe_r"]

    # Collect all candidate variants with their properties
    candidates = []

    # Helper to create candidate entry
    def add_candidate(name, policy_type, suggested_r, conservatism_score, priority=1):
        actual_r = round_to_allowed_ranks(suggested_r, allowed_ranks)
        if actual_r > probe_rank:
            actual_r = probe_rank

        # Skip control runs (no compression)
        if actual_r == probe_rank:
            return

        candidates.append(
            {
                "name": name,
                "policy_type": policy_type,
                "suggested_r": suggested_r,
                "actual_r": actual_r,
                "conservatism_score": conservatism_score,  # Higher = more conservative
                "priority": priority,  # 1=fast_mode, 2=full_mode_only
                "config": {
                    **lora_config,
                    "probe_r": actual_r,
                    "alpha": actual_r,
                },
            }
        )

    # Gather policy-based candidates (robust handling of various schema versions)
    print("\U0001f3af Analyzing policy-based rank suggestions...")
    policy_suggestions = (
        audit_data.get("policy_global_suggestions")
        or audit_data.get("policy_suggestions")
        or audit_data.get("rank_policy_suggestions")
        or audit_data.get("rank_suggestions_by_policy")
        or {}
    )
    if not isinstance(policy_suggestions, dict):
        policy_suggestions = {}

    print(f"   found policy suggestions: {list(policy_suggestions.keys())}")

    # Fast mode candidates (priority=1)
    # If explicit candidate_policies specified in config, use only those
    if config_candidate_policies:
        print(f"\u26a1 Generating explicit candidate policies: {config_candidate_policies}")
        requested_policies = set(config_candidate_policies)
    else:
        print("\u26a1 Generating default fast mode candidates...")
        # Default fast mode: energy_p90, knee_p90, erank_p90
        requested_policies = {"energy_p90", "knee_p90", "erank_p90"}

    # Energy-based rank suggestions (handles multiple possible key names)
    if "energy_p90" in requested_policies:
        energy90 = policy_suggestions.get("energy_90") or policy_suggestions.get("energy@0.90") or {}
        if isinstance(energy90, dict) and "uniform_p90" in energy90:
            print(f"   energy_p90: suggested_r={energy90['uniform_p90']}")
            add_candidate(
                "energy_p90",
                "energy",
                energy90["uniform_p90"],
                conservatism_score=CONSERVATISM_SCORES["energy_p90"],
                priority=1,
            )

    # Knee-based rank suggestions
    if "knee_p90" in requested_policies:
        knee = policy_suggestions.get("knee") or policy_suggestions.get("knee_detection") or {}
        if isinstance(knee, dict) and "uniform_p90" in knee:
            print(f"   knee_p90: suggested_r={knee['uniform_p90']}")
            add_candidate(
                "knee_p90", "knee", knee["uniform_p90"], conservatism_score=CONSERVATISM_SCORES["knee_p90"], priority=1
            )

    # Effective rank suggestions
    if "erank_p90" in requested_policies:
        erank = policy_suggestions.get("erank") or policy_suggestions.get("effective_rank") or {}
        if isinstance(erank, dict) and "uniform_p90" in erank:
            print(f"   erank_p90: suggested_r={erank['uniform_p90']}")
            add_candidate(
                "erank_p90",
                "erank",
                erank["uniform_p90"],
                conservatism_score=CONSERVATISM_SCORES["erank_p90"],
                priority=1,
            )

    # Full mode additional candidates (priority=2)
    if not fast_mode:
        print("\U0001f3af Generating additional full mode candidates...")
        # Legacy median/p90 from audit
        if "suggested_r_global_median" in audit_data:
            add_candidate(
                "uniform_median",
                "legacy",
                audit_data["suggested_r_global_median"],
                conservatism_score=CONSERVATISM_SCORES["uniform_median"],
                priority=2,
            )

        if "suggested_r_global_90" in audit_data:
            add_candidate(
                "uniform_p90",
                "legacy",
                audit_data["suggested_r_global_90"],
                conservatism_score=CONSERVATISM_SCORES["uniform_p90"],
                priority=2,
            )

        # OHT policy
        if "oht" in policy_suggestions and "uniform_p90" in policy_suggestions["oht"]:
            add_candidate(
                "oht_p90",
                "oht",
                policy_suggestions["oht"]["uniform_p90"],
                conservatism_score=CONSERVATISM_SCORES["oht_p90"],
                priority=2,
            )

        # Energy with median aggregation
        if "energy_90" in policy_suggestions and "uniform_median" in policy_suggestions["energy_90"]:
            add_candidate(
                "energy_median",
                "energy",
                policy_suggestions["energy_90"]["uniform_median"],
                conservatism_score=CONSERVATISM_SCORES["energy_median"],
                priority=2,
            )

    # Step 1: De-duplicate by actual_r (if multiple policies suggest same rank, pick best)
    print(f"\U0001f504 Processing {len(candidates)} initial candidates...")
    rank_to_candidates: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        rank = candidate["actual_r"]
        if rank not in rank_to_candidates:
            rank_to_candidates[rank] = []
        rank_to_candidates[rank].append(candidate)

    print(f"   found candidates for ranks: {sorted(rank_to_candidates.keys())}")

    # Enhanced deduplication: keep diversity by mapping duplicates to "second rung" alternatives
    print("\U0001f3af De-duplicating candidates by rank with diversity preservation...")
    deduplicated_candidates = []
    used_ranks = set()

    # Helper to find next most aggressive available rank
    def find_next_aggressive_rank(original_rank, used_ranks, allowed_ranks):
        """Find the next more aggressive (smaller) rank that's available."""
        available_ranks = [r for r in allowed_ranks if r < original_rank and r not in used_ranks]
        return min(available_ranks) if available_ranks else None

    for rank, rank_candidates in rank_to_candidates.items():
        if len(rank_candidates) == 1:
            # No collision, keep as-is
            deduplicated_candidates.append(rank_candidates[0])
            used_ranks.add(rank)
        else:
            # Multiple candidates for same rank - apply diversity preservation
            print(f"   \U0001f500 Rank collision at r={rank}: {len(rank_candidates)} candidates")

            # Sort by priority (1=fast_mode first), then conservatism (lower first)
            sorted_candidates = sorted(rank_candidates, key=lambda c: (c["priority"], c["conservatism_score"]))

            # Keep the best candidate at original rank
            best_candidate = sorted_candidates[0]
            policies = [c["policy_type"] for c in rank_candidates]
            best_candidate["name"] = (
                f"{best_candidate['name']}_r{rank}" if len(set(policies)) > 1 else best_candidate["name"]
            )
            best_candidate["dedup_note"] = f"Preferred choice from: {', '.join(set(policies))}"
            deduplicated_candidates.append(best_candidate)
            used_ranks.add(rank)

            # For remaining candidates, try to map to "second rung" alternatives
            for i, displaced_candidate in enumerate(sorted_candidates[1:], 1):
                next_rank = find_next_aggressive_rank(rank, used_ranks, allowed_ranks)
                if next_rank is not None:
                    print(
                        f"     \U0001f4cd Remapping {displaced_candidate['policy_type']} from r={rank} \u2192 r={next_rank} (second rung)"
                    )
                    displaced_candidate["actual_r"] = next_rank
                    displaced_candidate["name"] = f"{displaced_candidate['policy_type']}_r{next_rank}"
                    displaced_candidate["dedup_note"] = f"Remapped from r={rank} to avoid collision (second rung)"

                    # Update LoRA config
                    displaced_candidate["config"]["probe_r"] = next_rank
                    displaced_candidate["config"]["alpha"] = next_rank

                    deduplicated_candidates.append(displaced_candidate)
                    used_ranks.add(next_rank)
                else:
                    print(f"     \u274c No available second rung for {displaced_candidate['policy_type']} (r={rank})")
                    # Could not find alternative - candidate is dropped

    # Step 2: Filter by mode (fast_mode keeps only priority=1)
    print(f"\U0001f4cb Applying {'fast' if fast_mode else 'full'} mode filtering...")
    if fast_mode:
        filtered_candidates = [c for c in deduplicated_candidates if c["priority"] == 1]
    else:
        filtered_candidates = deduplicated_candidates

    print(f"   after mode filtering: {len(filtered_candidates)} candidates")

    # Step 3: Cap candidates by conservatism/diversity
    if len(filtered_candidates) > max_candidates:
        print(f"\u2696\ufe0f  Applying candidate limit (max {max_candidates})...")
        # Sort by conservatism score for diversity (keep range of conservative to aggressive)
        filtered_candidates.sort(key=lambda c: c["conservatism_score"])

        # Take every N-th to ensure diversity across conservatism spectrum
        step = len(filtered_candidates) / max_candidates
        capped_candidates = []
        for i in range(max_candidates):
            idx = int(i * step)
            capped_candidates.append(filtered_candidates[idx])
        filtered_candidates = capped_candidates

    # Convert to compression_configs format
    compression_configs = {}
    for candidate in filtered_candidates:
        compression_configs[candidate["name"]] = {
            "variant": candidate["name"],
            "suggested_r": candidate["suggested_r"],
            "actual_r": candidate["actual_r"],
            "rank_pattern": {},  # Uniform variants
            "alpha_pattern": {},
            "config": candidate["config"],
            "status": "ready",
            "reason": candidate.get("dedup_note"),
            "policy_type": candidate["policy_type"],
            "conservatism_score": candidate["conservatism_score"],
        }

    # Add per-layer candidate if available and different from uniform candidates
    per_layer_suggestions = audit_data.get("per_layer_suggestions")
    # Only add per_layer if requested and available (not in fast_mode by default)
    if per_layer_suggestions and (not fast_mode or len(filtered_candidates) < max_candidates):
        rank_pattern = per_layer_suggestions["rank_pattern"]

        # Clamp ranks to allowed values
        clamped_rank_pattern = {}
        for module_name, suggested_r in rank_pattern.items():
            clamped_r = min(suggested_r, probe_rank)
            # Round to nearest allowed rank
            valid_ranks = [r for r in allowed_ranks if r <= clamped_r]
            if valid_ranks:
                clamped_rank_pattern[module_name] = max(valid_ranks)
            else:
                clamped_rank_pattern[module_name] = min(allowed_ranks) if allowed_ranks else 1

        # Check if per-layer is different from uniform candidates
        avg_rank = sum(clamped_rank_pattern.values()) / len(clamped_rank_pattern) if clamped_rank_pattern else 0
        rounded_avg = round_to_allowed_ranks(avg_rank, allowed_ranks)

        # Only include if sufficiently different from uniform candidates
        uniform_ranks = {c["actual_r"] for c in filtered_candidates}
        if rounded_avg not in uniform_ranks and rounded_avg < probe_rank:
            alpha_pattern = {name: rank for name, rank in clamped_rank_pattern.items()}
            compression_configs["per_layer"] = {
                "variant": "per_layer",
                "suggested_r": avg_rank,
                "actual_r": rounded_avg,
                "rank_pattern": clamped_rank_pattern,
                "alpha_pattern": alpha_pattern,
                "config": {
                    **lora_config,
                    "probe_r": None,  # Per-layer uses rank_pattern
                    "alpha": None,
                },
                "status": "ready",
                "reason": None,
                "policy_type": "per_layer",
                "conservatism_score": CONSERVATISM_SCORES["per_layer"],
            }

    # Skip legacy SVD and per_layer_shuffled logic - handled by policy system above
    # All candidate generation is now done, proceed to final filtering

    # Jump directly to final candidate control section
    # (All legacy logic removed - policy system handles everything)
    pass  # No additional processing needed - policy system handles everything

    # D) per_layer_shuffled (control for mechanism testing)
    # Create shuffled control only if we have a successful per_layer variant
    if "per_layer" in compression_configs and compression_configs["per_layer"]["status"] == "ready":
        original_rank_pattern = compression_configs["per_layer"]["rank_pattern"]
        shuffled_rank_pattern = _create_shuffled_rank_pattern(
            original_rank_pattern, seed=config.get("train", {}).get("seed", DEFAULT_SEED)
        )

        # Create alpha pattern matching the shuffled ranks
        shuffled_alpha_pattern = {}
        for module_name, suggested_r in shuffled_rank_pattern.items():
            if suggested_r > 0:  # Only for active modules
                shuffled_alpha_pattern[module_name] = suggested_r

        # Normalize patterns
        shuffled_rank_pattern = normalize_rank_pattern(shuffled_rank_pattern)
        shuffled_alpha_pattern = normalize_alpha_pattern(shuffled_alpha_pattern)

        compression_configs["per_layer_shuffled"] = {
            "variant": "per_layer_shuffled",
            "suggested_r": len(shuffled_rank_pattern),
            "actual_r": len([r for r in shuffled_rank_pattern.values() if r > 0]),
            "rank_pattern": shuffled_rank_pattern,
            "alpha_pattern": shuffled_alpha_pattern,
            # Attach same audit metadata for consistency
            "_audit_layers": audit_data.get("layers", []),
            "_probe_rank": probe_rank,
            "_shuffle_seed": config.get("train", {}).get("seed", DEFAULT_SEED) + SHUFFLE_SEED_OFFSET,
            "config": {
                **lora_config,
                "rank_pattern": shuffled_rank_pattern,
                "alpha_pattern": shuffled_alpha_pattern,
                "probe_r": None,
                "alpha": None,
            },
            "status": "ready",
            "reason": "Shuffled control for audit-guided per-layer variant",
        }
    else:
        # No per_layer to shuffle
        compression_configs["per_layer_shuffled"] = {
            "variant": "per_layer_shuffled",
            "suggested_r": None,
            "actual_r": None,
            "rank_pattern": {},
            "alpha_pattern": {},
            "config": None,
            "status": "SKIPPED",
            "reason": "No per-layer variant to create shuffled control from",
        }

    # D) SVD truncation variants (both legacy and new format)

    # Legacy format support: enable_svd_variants + svd_ranks
    if compression_config.get("enable_svd_variants", False):
        svd_ranks = compression_config.get("svd_ranks", [])
        for rank in svd_ranks:
            if rank >= probe_rank:
                # Skip if no compression would happen
                compression_configs[f"svd_trunc_r{rank}"] = {
                    "variant": f"svd_trunc_r{rank}",
                    "suggested_r": rank,
                    "actual_r": rank,
                    "rank_pattern": {},
                    "alpha_pattern": {},
                    "config": None,
                    "status": "skipped",
                    "reason": f"SVD truncation rank r={rank} >= probe rank r={probe_rank} (no compression)",
                }
            else:
                compression_configs[f"svd_trunc_r{rank}"] = {
                    "variant": f"svd_trunc_r{rank}",
                    "suggested_r": rank,
                    "actual_r": rank,
                    "rank_pattern": {},
                    "alpha_pattern": {},
                    "config": {
                        **lora_config,
                        "probe_r": rank,  # Use truncated rank
                        "alpha": rank,  # Preserve alpha=r scaling
                    },
                    "status": "ready",
                    "reason": f"SVD truncation from r={probe_rank} to r={rank}",
                    "compression_method": "svd_truncation",
                    "source_rank": probe_rank,
                }

    # New format: compression.variants array (Step 3.1 enhancement)
    compression_variants = compression_config.get("variants", [])
    for variant_def in compression_variants:
        variant_name = variant_def.get("name")
        method = variant_def.get("method")

        if method == "svd_truncate":
            svd_variant_config = generate_svd_variant_config(
                variant_def, audit_data, probe_rank, lora_config, allowed_ranks
            )
            compression_configs[variant_name] = svd_variant_config

    # Apply second rung decision logic (audit-driven compression candidates)
    print("\U0001f3af Evaluating second rung compression candidates...")
    decision_trace = create_decision_trace(probe_rank, audit_data)

    # Collect both existing names and used ranks to avoid duplicates
    existing_candidate_names = list(compression_configs.keys())
    used_ranks = {cfg["actual_r"] for cfg in compression_configs.values()}

    # Add rank-based names to prevent second rung from creating duplicates
    used_rank_names = [f"uniform_r{rank}" for rank in used_ranks]
    existing_candidates_with_ranks = existing_candidate_names + used_rank_names

    second_rung_candidates = maybe_add_second_rung_candidates(
        probe_rank=probe_rank,
        audit_metrics=decision_trace.audit_metrics,
        allowed_ranks=allowed_ranks,
        existing_candidates=existing_candidates_with_ranks,
        decision_trace=decision_trace,
    )

    # Add second rung candidates to compression configs
    for candidate in second_rung_candidates:
        lora_config_copy = lora_config.copy()
        lora_config_copy["probe_r"] = candidate["actual_r"]
        lora_config_copy["alpha"] = candidate["actual_r"]  # Match rank for scaling

        compression_configs[candidate["name"]] = {
            "variant": candidate["name"],
            "suggested_r": candidate["suggested_r"],
            "actual_r": candidate["actual_r"],
            "method": "uniform",
            "policy_type": candidate["policy_type"],
            "conservatism_score": candidate["conservatism_score"],
            "priority": candidate["priority"],
            "config": lora_config_copy,
            "status": "ready",
        }
        print(f"   Added {candidate['name']}: r={candidate['actual_r']} ({candidate['policy_type']})")

    # Store decision trace for reporting
    globals()["_last_decision_trace"] = decision_trace

    # Apply final candidate control (remove old logic artifacts and enforce caps)
    final_configs = {}

    # First, collect configs from my new system only
    for name, config in compression_configs.items():
        if config.get("policy_type") in [
            "energy",
            "knee",
            "erank",
            "oht",
            "legacy",
            "per_layer",
            "second_rung_tier_a",
            "second_rung_tier_b",
        ]:
            final_configs[name] = config

    # Apply capping if we have too many
    if len(final_configs) > max_candidates:
        # Sort by conservatism for diversity
        sorted_configs = sorted(final_configs.items(), key=lambda x: x[1].get("conservatism_score", 999))

        # Take every N-th for diversity
        step = len(sorted_configs) / max_candidates
        capped_configs = {}
        for i in range(max_candidates):
            idx = int(i * step)
            name, config = sorted_configs[idx]
            capped_configs[name] = config
        final_configs = capped_configs

    # Print candidate summary
    if len(final_configs) > 0:
        print(f"\U0001f4ca Bench candidate control: {len(final_configs)} variants generated")
        if fast_mode:
            print("   Mode: FAST (energy_p90, knee_p90, erank_p90)")
        else:
            print(f"   Mode: FULL (capped at {max_candidates})")

        rank_summary: dict[int | None, list[str]] = {}
        for name, config in final_configs.items():
            r = config["actual_r"]
            if r not in rank_summary:
                rank_summary[r] = []
            rank_summary[r].append(name)

        for rank in sorted(rank_summary.keys(), key=lambda x: (x is None, x)):
            variants = rank_summary[rank]
            if len(variants) > 1:
                print(f"   r={rank}: {', '.join(variants)} (deduplicated)")
            else:
                print(f"   r={rank}: {variants[0]}")
        print()

    # Final candidate generation summary
    total_configs = len(final_configs)
    ready_configs = sum(1 for cfg in final_configs.values() if cfg.get("status") == "ready")
    print(f"\u2705 Candidate generation completed: {ready_configs}/{total_configs} configs ready for training")

    # Return both configs and decision trace
    return final_configs, decision_trace
