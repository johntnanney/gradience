"""
gradience.vnext.svd_truncate

Post-hoc SVD truncation for LoRA adapters.

Takes an already trained LoRA adapter and creates a rank-k approximation via SVD,
then refactors back into new B' and A' matrices of rank k.

Key features:
- Preserves scaling behavior (lora_alpha / r ratio by default)  
- Supports uniform rank truncation across all modules
- Generates energy retention reports
- Maintains PEFT directory structure for easy drop-in replacement
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import numpy as np

from .audit.lora_audit import (
    find_peft_files,
    load_peft_adapter_config, 
    load_adapter_state_dict,
    LoRAAdapterConfig
)


@dataclass
class SVDTruncationReport:
    """Report from SVD truncation process."""
    original_rank: int
    target_rank: int
    total_modules: int
    energy_retained: float  # Average across all modules
    per_module_energy: List[Dict[str, Union[str, int, float]]]  # Per-module details
    alpha_mode: str
    compression_ratio: float  # original_params / truncated_params
    # Future-proofing fields to prevent audit mixups
    source_directory: Optional[str] = None
    output_directory: Optional[str] = None  
    timestamp: Optional[str] = None
    total_original_params: Optional[int] = None
    total_new_params: Optional[int] = None


def _matches_lora_A(key: str, adapter_name: str = "default") -> bool:
    """Check if a key corresponds to a LoRA A matrix."""
    # Handle both patterns:
    # - .lora_A.default.weight (newer PEFT)
    # - .lora_A.weight (older PEFT)
    return (f".lora_A.{adapter_name}.weight" in key or 
            (key.endswith(".lora_A.weight") and adapter_name == "default"))


def _corresponding_lora_B_key(lora_A_key: str, adapter_name: str = "default") -> str:
    """Generate corresponding LoRA B key from LoRA A key."""
    if f".lora_A.{adapter_name}.weight" in lora_A_key:
        return lora_A_key.replace(f".lora_A.{adapter_name}.weight", f".lora_B.{adapter_name}.weight")
    elif lora_A_key.endswith(".lora_A.weight"):
        return lora_A_key.replace(".lora_A.weight", ".lora_B.weight")
    else:
        raise ValueError(f"Unexpected LoRA A key pattern: {lora_A_key}")


def _parse_and_pair_lora_matrices(
    state_dict: Dict[str, torch.Tensor], 
    adapter_name: str = "default"
) -> Dict[str, Dict[str, Tuple[str, torch.Tensor]]]:
    """
    Parse and pair LoRA matrices robustly.
    
    Handles different PEFT key patterns:
    - .lora_A.default.weight / .lora_B.default.weight (newer)
    - .lora_A.weight / .lora_B.weight (older)
    
    Returns dict mapping base_key -> {"A": (key, tensor), "B": (key, tensor)}
    Only returns complete pairs with compatible shapes.
    """
    complete_pairs = {}
    
    # Find all LoRA A matrices
    for key, tensor in state_dict.items():
        if _matches_lora_A(key, adapter_name):
            try:
                # Find corresponding B key
                lora_B_key = _corresponding_lora_B_key(key, adapter_name)
                
                # Check if B matrix exists
                if lora_B_key not in state_dict:
                    print(f"Warning: Found LoRA A key '{key}' but no matching B key '{lora_B_key}'")
                    continue
                
                lora_A = tensor
                lora_B = state_dict[lora_B_key]
                
                # Validate shapes are compatible for LoRA: A (r, d_in), B (d_out, r)
                if lora_A.dim() != 2 or lora_B.dim() != 2:
                    print(f"Warning: LoRA tensors must be 2D. A: {lora_A.shape}, B: {lora_B.shape}")
                    continue
                
                r_A, d_in = lora_A.shape
                d_out, r_B = lora_B.shape
                
                if r_A != r_B:
                    print(f"Warning: Rank mismatch between A and B. A rank: {r_A}, B rank: {r_B}")
                    continue
                
                # Create base key for grouping (remove lora_A/lora_B specific parts)
                if f".lora_A.{adapter_name}.weight" in key:
                    base_key = key.replace(f".lora_A.{adapter_name}.weight", f".lora_PAIR.{adapter_name}.weight")
                else:
                    base_key = key.replace(".lora_A.weight", ".lora_PAIR.weight")
                
                complete_pairs[base_key] = {
                    "A": (key, lora_A),
                    "B": (lora_B_key, lora_B)
                }
                
            except Exception as e:
                print(f"Warning: Failed to process LoRA A key '{key}': {e}")
                continue
    
    print(f"Found {len(complete_pairs)} complete LoRA A/B pairs")
    
    # Debug: print first few pairs for verification
    for i, (base_key, pair) in enumerate(complete_pairs.items()):
        if i < 3:  # Show first 3 pairs
            A_key, A_tensor = pair["A"]
            B_key, B_tensor = pair["B"]
            print(f"  Pair {i+1}: {A_key} {A_tensor.shape} <-> {B_key} {B_tensor.shape}")
    
    return complete_pairs


def _compute_svd_truncation(
    lora_A: torch.Tensor,  # shape: (r, d_in) 
    lora_B: torch.Tensor,  # shape: (d_out, r)
    target_rank: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute fast SVD truncation of LoRA update ΔW = B @ A without materializing ΔW.
    
    Uses QR decomposition method to avoid computing the large ΔW matrix:
    1. QR factorize B = Qb @ Rb  
    2. QR factorize A^T = Qa @ Ra
    3. SVD the small r×r matrix M = Rb @ Ra^T
    4. Reconstruct truncated factors
    
    Returns:
        - A_new: (target_rank, d_in)
        - B_new: (d_out, target_rank) 
        - energy_retained: fraction of spectral energy retained
    """
    original_dtype = lora_A.dtype
    r, d_in = lora_A.shape
    d_out, r_B = lora_B.shape
    
    assert r == r_B, f"Rank mismatch: A has rank {r}, B has rank {r_B}"
    
    # Cast to float32 for numerical stability in QR/SVD operations
    A_work = lora_A.cpu().to(torch.float32)
    B_work = lora_B.cpu().to(torch.float32)
    
    # Step 1: QR factorize B = Qb @ Rb
    # B: (d_out, r) -> Qb: (d_out, r), Rb: (r, r)
    Qb, Rb = torch.linalg.qr(B_work)
    
    # Step 2: QR factorize A^T = Qa @ Ra  
    # A^T: (d_in, r) -> Qa: (d_in, r), Ra: (r, r)
    At = A_work.T  # (d_in, r)
    Qa, Ra = torch.linalg.qr(At)
    
    # Step 3: SVD the small r×r matrix M = Rb @ Ra^T
    # This is much cheaper than SVD of the full ΔW matrix
    M = Rb @ Ra.T  # (r, r)
    U_small, S, Vt_small = torch.linalg.svd(M, full_matrices=False)
    
    # Step 4: Reconstruct the singular vectors of ΔW = B @ A
    # Left singular vectors: U_big = Qb @ U_small  
    # Right singular vectors: V_big = Qa @ V_small
    U_big = Qb @ U_small      # (d_out, r)  
    V_big = Qa @ Vt_small.T   # (d_in, r)
    
    # Truncate to target rank
    k = min(target_rank, len(S))
    S_trunc = S[:k]                    # (k,)
    U_big_trunc = U_big[:, :k]         # (d_out, k)
    V_big_trunc = V_big[:, :k]         # (d_in, k)
    
    # Refactor into LoRA form: B' @ A' ≈ ΔW_k
    # Split singular values between B' and A' for numerical stability
    sqrt_S = torch.sqrt(S_trunc.clamp(min=1e-16))  # (k,)
    
    B_new = U_big_trunc * sqrt_S[None, :]         # (d_out, k)
    A_new = sqrt_S[:, None] * V_big_trunc.T       # (k, d_in)
    
    # Compute energy retention from singular values
    total_energy = torch.sum(S**2).item()
    retained_energy = torch.sum(S_trunc**2).item()
    energy_fraction = retained_energy / total_energy if total_energy > 0 else 0.0
    
    # Convert back to original dtype
    A_new = A_new.to(original_dtype)
    B_new = B_new.to(original_dtype)
    
    return A_new, B_new, energy_fraction


def _update_adapter_config(
    config: LoRAAdapterConfig,
    target_rank: int,
    alpha_mode: Literal["keep_ratio", "keep_alpha"] = "keep_ratio"
) -> Dict:
    """
    Update adapter config safely for truncated rank.
    
    Follows PEFT conventions:
    - Updates r to target_rank
    - Updates lora_alpha based on alpha_mode
    - Sets rank_pattern and alpha_pattern to empty dicts (uniform truncation)
    - Preserves all other config fields
    """
    new_config = dict(config.raw)
    
    # Update rank
    new_config["r"] = target_rank
    
    # Update alpha based on mode
    if alpha_mode == "keep_ratio":
        # Keep lora_alpha / r ratio constant: alpha_new = alpha_old * (k / r_old)
        if config.r is not None and config.lora_alpha is not None and config.r > 0:
            alpha_new = config.lora_alpha * (target_rank / config.r)
            new_config["lora_alpha"] = alpha_new
        else:
            # Fallback: use target_rank as alpha (common default)
            new_config["lora_alpha"] = float(target_rank)
            print(f"Warning: Could not preserve alpha ratio, using alpha={target_rank}")
    elif alpha_mode == "keep_alpha":
        # Keep original alpha unchanged (scaling will change)
        if config.lora_alpha is not None:
            new_config["lora_alpha"] = config.lora_alpha
        else:
            new_config["lora_alpha"] = float(target_rank)
            print(f"Warning: No original alpha found, using alpha={target_rank}")
    
    # Set rank_pattern and alpha_pattern to empty dicts for uniform truncation
    # IMPORTANT: Don't remove these keys entirely - PEFT expects them to exist
    # Empty dicts indicate uniform configuration across all target modules
    new_config["rank_pattern"] = {}
    new_config["alpha_pattern"] = {}
    
    # Ensure consistent types for JSON serialization
    new_config["r"] = int(target_rank)
    new_config["lora_alpha"] = float(new_config["lora_alpha"])
    
    return new_config


def svd_truncate_peft_dir(
    peft_dir: Path,
    out_dir: Path,
    target_rank: int,
    *,
    alpha_mode: Literal["keep_ratio", "keep_alpha"] = "keep_ratio",
    adapter_name: str = "default", 
    save_dtype: Literal["fp16", "bf16", "fp32"] = "fp16",
) -> SVDTruncationReport:
    """
    SVD truncate a PEFT adapter directory to target rank.
    
    Args:
        peft_dir: Input PEFT adapter directory
        out_dir: Output directory for truncated adapter
        target_rank: Target rank for all LoRA modules
        alpha_mode: How to handle lora_alpha scaling
        adapter_name: PEFT adapter name (usually "default")
        save_dtype: Dtype for saved weights
        
    Returns:
        Report with energy retention and other metrics
    """
    peft_dir = Path(peft_dir)
    out_dir = Path(out_dir)
    
    # Find and load config/weights
    config_path, weights_path, issues = find_peft_files(peft_dir)
    if config_path is None or weights_path is None:
        raise FileNotFoundError(f"Could not find PEFT files in {peft_dir}. Issues: {issues}")
    
    config = load_peft_adapter_config(config_path)
    state_dict = load_adapter_state_dict(weights_path, map_location="cpu")
    
    if config.r is None:
        raise ValueError("Original LoRA rank (r) not found in adapter config")
    
    if target_rank >= config.r:
        raise ValueError(f"Target rank {target_rank} must be < original rank {config.r}")
    
    # Parse and pair LoRA matrices robustly
    complete_pairs = _parse_and_pair_lora_matrices(state_dict, adapter_name)
    
    if not complete_pairs:
        raise ValueError("No complete LoRA A/B pairs found in adapter weights")
    
    # Process each pair
    new_state_dict = {}
    per_module_energy = []
    total_original_params = 0
    total_new_params = 0
    
    # Copy ALL non-LoRA tensors for compatibility
    # The original adapter may contain task-specific weights that need to be preserved
    # for PEFT to load the adapter successfully
    for key, tensor in state_dict.items():
        if "lora_A" not in key and "lora_B" not in key:
            # Preserve all non-LoRA weights to maintain compatibility
            new_state_dict[key] = tensor.contiguous()
    
    for base_key, pair in complete_pairs.items():
        A_key, lora_A = pair["A"] 
        B_key, lora_B = pair["B"]
        
        # Compute SVD truncation
        A_new, B_new, energy_retained = _compute_svd_truncation(lora_A, lora_B, target_rank)
        
        # Store new tensors (ensure contiguous for safetensors)
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        target_dtype = dtype_map[save_dtype]
        
        new_state_dict[A_key] = A_new.to(target_dtype).contiguous()
        new_state_dict[B_key] = B_new.to(target_dtype).contiguous()
        
        # Track statistics
        module_name = base_key.replace(f".lora_PAIR.{adapter_name}.weight", "").replace(".lora_PAIR.weight", "")
        original_params = lora_A.numel() + lora_B.numel()
        new_params = A_new.numel() + B_new.numel()
        
        total_original_params += original_params
        total_new_params += new_params
        
        per_module_energy.append({
            "module_name": module_name,
            "original_rank": lora_A.shape[0],
            "target_rank": target_rank,
            "energy_retained": energy_retained,
            "original_params": original_params,
            "new_params": new_params,
        })
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute overall statistics for the report
    avg_energy = np.mean([m["energy_retained"] for m in per_module_energy])
    compression_ratio = total_original_params / total_new_params if total_new_params > 0 else 1.0
    
    # Create comprehensive truncation report with future-proofing
    import datetime
    truncation_report = SVDTruncationReport(
        original_rank=config.r,
        target_rank=target_rank,
        total_modules=len(complete_pairs),
        energy_retained=float(avg_energy),
        per_module_energy=per_module_energy,
        alpha_mode=alpha_mode,
        compression_ratio=float(compression_ratio),
        # Future-proofing: add metadata to prevent audit mixups
        source_directory=str(peft_dir.name),
        output_directory=str(out_dir.name), 
        timestamp=datetime.datetime.now().isoformat(),
        total_original_params=total_original_params,
        total_new_params=total_new_params
    )
    
    # 1. Save truncated adapter config (REQUIRED by PEFT)
    new_config = _update_adapter_config(config, target_rank, alpha_mode)
    config_out_path = out_dir / "adapter_config.json"
    with open(config_out_path, 'w') as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    
    # 2. Save truncated weights (REQUIRED by PEFT)
    weights_out_path = out_dir / "adapter_model.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(new_state_dict, weights_out_path)
    except ImportError:
        # Fallback to torch.save if safetensors not available
        weights_out_path = out_dir / "adapter_model.pt"
        torch.save(new_state_dict, weights_out_path)
        print("Warning: safetensors not available, saved as .pt file")
    
    # 3. Save truncation report (OPTIONAL but highly useful for debugging)
    report_path = out_dir / "truncation_report.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(truncation_report), f, indent=2, ensure_ascii=False)
    
    # 4. Copy other supporting files to maintain adapter completeness
    # This ensures the output directory contains all files from the original adapter
    skip_files = {
        "adapter_config.json", "adapter_config.yaml", "adapter_config.yml",
        "adapter_model.safetensors", "adapter_model.pt", "adapter_model.bin", 
        "adapter_model.pth", "pytorch_model.bin",
        "truncation_report.json"  # Don't copy over our own report
    }
    
    copied_files = []
    for item in peft_dir.iterdir():
        if item.is_file() and item.name not in skip_files:
            try:
                shutil.copy2(item, out_dir / item.name)
                copied_files.append(item.name)
            except Exception as e:
                print(f"Warning: Failed to copy {item.name}: {e}")
    
    # 5. Generate a simple README for the truncated adapter
    readme_path = out_dir / "README.md"
    if not readme_path.exists():  # Don't overwrite existing README
        with open(readme_path, 'w') as f:
            f.write(_generate_truncation_readme(
                truncation_report, 
                peft_dir, 
                copied_files, 
                weights_out_path.name
            ))
    
    return truncation_report


def _generate_truncation_readme(
    report: SVDTruncationReport, 
    source_dir: Path, 
    copied_files: List[str],
    weights_filename: str
) -> str:
    """Generate a README.md for the truncated adapter."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""# SVD Truncated LoRA Adapter

This adapter was created via SVD truncation using Gradience.

## Truncation Summary

- **Original rank**: {report.original_rank}
- **Target rank**: {report.target_rank}  
- **Compression ratio**: {report.compression_ratio:.1f}x
- **Energy retained**: {report.energy_retained:.1%}
- **Alpha mode**: {report.alpha_mode}
- **Modules processed**: {report.total_modules}

## Source

- **Original adapter**: `{source_dir.name}`
- **Truncated on**: {timestamp}
- **Method**: Fast SVD via QR decomposition

## Files

### Required PEFT Files
- `adapter_config.json` - Updated configuration with rank={report.target_rank}
- `{weights_filename}` - Truncated LoRA weights

### Metadata
- `truncation_report.json` - Detailed per-module energy retention
- `README.md` - This file

### Copied from Original
{chr(10).join(f"- `{f}`" for f in sorted(copied_files)) if copied_files else "- (none)"}

## Usage

Load this adapter with PEFT/Transformers as usual:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("base_model_name")
model = PeftModel.from_pretrained(model, "path/to/this/adapter")
```

## Energy Retention by Layer

Top 5 layers by energy retention:

{chr(10).join(
    f"- {module['module_name'].split('.')[-1]}: {module['energy_retained']:.1%}" 
    for module in sorted(report.per_module_energy, key=lambda x: x['energy_retained'], reverse=True)[:5]
)}

For complete per-module statistics, see `truncation_report.json`.

---
*Generated by Gradience SVD Truncation*
"""


def save_truncation_report(report: SVDTruncationReport, output_path: Path) -> None:
    """Save truncation report as JSON."""
    with open(output_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)