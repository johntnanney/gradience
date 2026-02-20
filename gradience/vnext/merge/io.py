"""
Adapter weight loading and layer matching for merge compatibility audit.

Reuses existing adapter infrastructure from gradience.vnext.audit for file
discovery, config parsing, and weight loading. Adds layer matching logic to
pair modules across two adapters by their canonical prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import logging

import torch

from gradience.exceptions import MergeError
from gradience.vnext.audit import (
    LoRAAdapterConfig,
    find_peft_files,
    load_peft_adapter_config,
    load_adapter_state_dict,
    iter_lora_pairs,
    orient_lora_factors,
    infer_module_type,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterInfo:
    """Loaded adapter metadata and weights."""

    path: Path
    config: LoRAAdapterConfig
    state_dict: Dict[str, torch.Tensor]
    lora_pairs: Tuple[Tuple[str, str, str], ...]  # (module_prefix, a_key, b_key)

    @property
    def rank(self) -> int:
        if self.config.r is None:
            raise MergeError(
                f"Adapter at {self.path} has no 'r' (rank) in config. "
                f"Cannot compute merge audit without a known rank."
            )
        return self.config.r

    @property
    def alpha(self) -> float:
        if self.config.lora_alpha is None:
            raise MergeError(
                f"Adapter at {self.path} has no 'lora_alpha' in config. "
                f"Cannot compute merge audit without a known alpha."
            )
        return float(self.config.lora_alpha)

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        r = self.rank  # triggers the None check
        return self.alpha / max(r, 1)

    @property
    def module_prefixes(self) -> List[str]:
        return [prefix for prefix, _, _ in self.lora_pairs]


def load_adapter(adapter_dir: Union[str, Path]) -> AdapterInfo:
    """Load a PEFT adapter: config + weights + discovered LoRA pairs.

    Parameters
    ----------
    adapter_dir : path to PEFT adapter directory containing
        adapter_config.json and adapter_model.safetensors (or .bin).

    Returns
    -------
    AdapterInfo with parsed config, loaded state_dict, and LoRA pair list.

    Raises
    ------
    FileNotFoundError
        If adapter directory, config, or weights file is missing.
    ValueError
        If no LoRA pairs are found in the weights.
    """
    logger.debug("Loading adapter from %s", adapter_dir)
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    config_path, weights_path, issues = find_peft_files(adapter_dir)

    if config_path is None:
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_dir}"
        )
    if weights_path is None:
        raise FileNotFoundError(
            f"No adapter weights (safetensors or bin) found in {adapter_dir}"
        )

    config = load_peft_adapter_config(config_path)

    if config.r is None:
        raise MergeError(
            f"Adapter config at {config_path} is missing 'r' (rank). "
            f"Merge audit requires a known rank."
        )
    if config.lora_alpha is None:
        raise MergeError(
            f"Adapter config at {config_path} is missing 'lora_alpha'. "
            f"Merge audit requires a known alpha."
        )

    state_dict = load_adapter_state_dict(weights_path, map_location="cpu")

    pairs = tuple(iter_lora_pairs(state_dict))
    logger.debug("Loaded %d LoRA pairs from %s", len(pairs), adapter_dir)
    if not pairs:
        raise MergeError(
            f"No LoRA A/B pairs found in weights at {weights_path}. "
            f"Keys: {list(state_dict.keys())[:5]}..."
        )

    return AdapterInfo(
        path=adapter_dir,
        config=config,
        state_dict=state_dict,
        lora_pairs=pairs,
    )


def match_layers(
    info_a: AdapterInfo,
    info_b: AdapterInfo,
) -> Tuple[List[str], List[str], List[str]]:
    """Match LoRA layers between two adapters by module prefix.

    Returns
    -------
    shared : list of module prefixes present in both adapters
    only_a : list of module prefixes only in adapter A
    only_b : list of module prefixes only in adapter B

    The shared list preserves the order from adapter A.
    """
    set_a = set(info_a.module_prefixes)
    set_b = set(info_b.module_prefixes)

    shared = [p for p in info_a.module_prefixes if p in set_b]
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    return shared, only_a, only_b


def extract_factors(
    info: AdapterInfo,
    module_prefix: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Extract oriented (A, B, rank) for a specific layer.

    Parameters
    ----------
    info : loaded adapter
    module_prefix : canonical module name (e.g. ``model.layers.0.self_attn.q_proj``)

    Returns
    -------
    A : tensor of shape (r, d_in)
    B : tensor of shape (d_out, r)
    r : rank (inner dimension)

    Raises
    ------
    KeyError
        If module_prefix is not found in the adapter's LoRA pairs.
    """
    for prefix, a_key, b_key in info.lora_pairs:
        if prefix == module_prefix:
            A_raw = info.state_dict[a_key]
            B_raw = info.state_dict[b_key]
            A, B, r = orient_lora_factors(A_raw, B_raw)
            return A, B, r

    raise KeyError(
        f"Module prefix '{module_prefix}' not found in adapter at {info.path}. "
        f"Available: {info.module_prefixes[:5]}..."
    )


def get_module_type(module_prefix: str) -> str:
    """Infer module type (attn/mlp/other) from module prefix name."""
    return infer_module_type(module_prefix)
