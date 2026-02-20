"""
Merge executor — takes a MergePlan and produces a PEFT-compatible adapter.

The executor processes each layer sequentially:
1. Load LoRA factors (A, B) from both adapters
2. Materialize dense ΔW = scaling * B @ A for each adapter
3. Apply the merge strategy to produce merged ΔW
4. Refactor merged ΔW back into LoRA (B', A') via truncated SVD
5. Store in the output state dict

Output is a standard PEFT directory:
    output_dir/
        adapter_config.json
        adapter_model.safetensors
        merge_plan.json
        merge_result.json

Usage::

    from gradience.vnext.merge import execute_merge, MergePlan

    plan = MergePlan.from_json("merge_plan.json")
    result = execute_merge(plan, output_dir="./merged_adapter")
    print(f"Mean reconstruction error: {result.mean_reconstruction_error:.4f}")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import logging

import torch

from gradience.exceptions import MergeError
from gradience.vnext.merge.io import AdapterInfo, extract_factors, load_adapter
from gradience.vnext.merge.plan import MergePlan
from gradience.vnext.merge.refactor import refactor_to_lora
from gradience.vnext.merge.strategies import get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "fp64": torch.float64,
    "fp32": torch.float32,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LayerMergeResult:
    """Metrics from merging one layer."""

    module_prefix: str
    strategy_used: str
    reconstruction_error: float
    input_rank_a: int
    input_rank_b: int
    output_rank: int
    merged_frobenius: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_prefix": self.module_prefix,
            "strategy_used": self.strategy_used,
            "reconstruction_error": self.reconstruction_error,
            "input_rank_a": self.input_rank_a,
            "input_rank_b": self.input_rank_b,
            "output_rank": self.output_rank,
            "merged_frobenius": self.merged_frobenius,
        }


@dataclass
class MergeResult:
    """Complete merge execution result."""

    plan: MergePlan
    output_dir: Path
    layer_results: List[LayerMergeResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    @property
    def mean_reconstruction_error(self) -> float:
        """Mean reconstruction error across all layers."""
        if not self.layer_results:
            return 0.0
        return sum(lr.reconstruction_error for lr in self.layer_results) / len(
            self.layer_results
        )

    @property
    def max_reconstruction_error(self) -> float:
        """Maximum reconstruction error across layers."""
        if not self.layer_results:
            return 0.0
        return max(lr.reconstruction_error for lr in self.layer_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "gradience.merge_result/v1",
            "plan_id": self.plan.plan_id,
            "strategy_name": self.plan.strategy_name,
            "output_dir": str(self.output_dir),
            "output_rank": self.plan.output_rank,
            "output_alpha": self.plan.output_alpha,
            "n_layers": len(self.layer_results),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "mean_reconstruction_error": round(self.mean_reconstruction_error, 6),
            "max_reconstruction_error": round(self.max_reconstruction_error, 6),
            "per_layer": [lr.to_dict() for lr in self.layer_results],
        }

    def to_json(self, path: Path) -> None:
        """Write merge result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def execute_merge(
    plan: MergePlan,
    output_dir: Union[str, Path],
    *,
    compute_dtype: str = "float64",
    verbose: bool = False,
) -> MergeResult:
    """Execute a merge plan and write a PEFT-compatible adapter.

    Parameters
    ----------
    plan : MergePlan specifying per-layer strategies, coefficients, and ranks
    output_dir : directory to write the merged adapter
    compute_dtype : ``"float64"`` (default) or ``"float32"`` for SVD precision
    verbose : if True, print progress to stdout

    Returns
    -------
    MergeResult with per-layer metrics and output location.

    Raises
    ------
    FileNotFoundError
        If adapter directories from the plan don't exist.
    ValueError
        If plan has no layer configs or adapters are incompatible.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Starting merge: strategy=%s, %d layers, output=%s", plan.strategy_name, len(plan.layer_configs), output_dir)

    if not plan.layer_configs:
        raise MergeError("MergePlan has no layer_configs — nothing to merge.")

    start_time = time.monotonic()

    # --- Load both adapters ---
    if verbose:
        print(f"Loading adapter A: {plan.adapter_a_dir}")
    info_a = load_adapter(plan.adapter_a_dir)

    if verbose:
        print(f"Loading adapter B: {plan.adapter_b_dir}")
    info_b = load_adapter(plan.adapter_b_dir)

    if verbose:
        print(
            f"  A: rank={info_a.rank}, alpha={info_a.alpha}, "
            f"{len(info_a.lora_pairs)} layers"
        )
        print(
            f"  B: rank={info_b.rank}, alpha={info_b.alpha}, "
            f"{len(info_b.lora_pairs)} layers"
        )

    # --- Process each layer ---
    output_state_dict: Dict[str, torch.Tensor] = {}
    layer_results: List[LayerMergeResult] = []

    for i, layer_config in enumerate(plan.layer_configs):
        prefix = layer_config.module_prefix

        if verbose:
            print(
                f"  [{i + 1}/{len(plan.layer_configs)}] "
                f"Merging {prefix} ({layer_config.strategy})...",
                end="",
                flush=True,
            )

        # Step 1: Extract LoRA factors
        A_a, B_a, r_a = extract_factors(info_a, prefix)
        A_b, B_b, r_b = extract_factors(info_b, prefix)

        # Step 2: Materialize dense ΔW
        dW_a = info_a.scaling * (B_a @ A_a)
        dW_b = info_b.scaling * (B_b @ A_b)

        # Step 3: Apply merge strategy
        strategy = get_strategy(layer_config.strategy)
        merged_dW = strategy.merge(dW_a, dW_b, layer_config)

        merged_frobenius = torch.linalg.norm(merged_dW).item()

        # Step 4: Refactor back to LoRA
        target_rank = layer_config.target_rank or plan.output_rank
        B_new, A_new, recon_error = refactor_to_lora(
            merged_dW,
            target_rank=target_rank,
            alpha=plan.output_alpha,
            compute_dtype=compute_dtype,
        )

        # Step 5: Store in output state dict
        output_state_dict[f"{prefix}.lora_A.weight"] = A_new.contiguous()
        output_state_dict[f"{prefix}.lora_B.weight"] = B_new.contiguous()

        layer_results.append(
            LayerMergeResult(
                module_prefix=prefix,
                strategy_used=layer_config.strategy,
                reconstruction_error=recon_error,
                input_rank_a=r_a,
                input_rank_b=r_b,
                output_rank=target_rank,
                merged_frobenius=merged_frobenius,
            )
        )

        if verbose:
            print(f" recon_error={recon_error:.4f}")

    # --- Release loaded adapters ---
    try:
        object.__delattr__(info_a, "state_dict")
        object.__delattr__(info_b, "state_dict")
    except (AttributeError, TypeError):
        pass

    # --- Write output ---
    _write_peft_adapter(
        output_dir=output_dir,
        state_dict=output_state_dict,
        plan=plan,
        info_a=info_a,
    )

    # Write plan and result
    plan.to_json(output_dir / "merge_plan.json")

    total_time = time.monotonic() - start_time

    result = MergeResult(
        plan=plan,
        output_dir=output_dir,
        layer_results=layer_results,
        total_time_seconds=total_time,
    )
    result.to_json(output_dir / "merge_result.json")

    logger.debug("Merge completed in %.1fs: mean_recon=%.4f, max_recon=%.4f", total_time, result.mean_reconstruction_error, result.max_reconstruction_error)

    if verbose:
        print(
            f"\nMerge complete in {total_time:.1f}s"
            f"\n  Output: {output_dir}"
            f"\n  Mean recon error: {result.mean_reconstruction_error:.4f}"
            f"\n  Max recon error: {result.max_reconstruction_error:.4f}"
        )

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_peft_adapter(
    output_dir: Path,
    state_dict: Dict[str, torch.Tensor],
    plan: MergePlan,
    info_a: AdapterInfo,
) -> None:
    """Write PEFT-compatible adapter directory.

    Creates:
        adapter_config.json — PEFT-compatible adapter configuration
        adapter_model.safetensors — merged weights
    """
    # Build adapter config from plan + source adapter metadata
    target_modules = sorted(set(
        lc.module_prefix.split(".")[-1]
        for lc in plan.layer_configs
    ))

    # Infer base model from source adapter config
    base_model = getattr(info_a.config, "raw", {}).get(
        "base_model_name_or_path", "unknown"
    )

    adapter_config = {
        "peft_type": "LORA",
        "r": plan.output_rank,
        "lora_alpha": plan.output_alpha,
        "target_modules": target_modules,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "task_type": getattr(info_a.config, "raw", {}).get("task_type", "CAUSAL_LM"),
        "fan_in_fan_out": False,
        "lora_dropout": 0.0,
        # Merge provenance
        "gradience_merge": {
            "plan_id": plan.plan_id,
            "strategy": plan.strategy_name,
            "adapter_a": plan.adapter_a_dir,
            "adapter_b": plan.adapter_b_dir,
        },
    }

    config_path = output_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Save weights as safetensors
    weights_path = output_dir / "adapter_model.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(weights_path))
    except ImportError:
        # Fallback to torch.save if safetensors not available
        torch.save(state_dict, str(output_dir / "adapter_model.bin"))
