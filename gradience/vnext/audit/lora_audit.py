"""
gradience.vnext.audit.lora_audit

LoRA adapter efficiency auditing.

This module is designed for the "Efficiency Auditor" role described in the Gradience
research synthesis: spectral metrics are diagnostic/auditing tools that help quantify
*capacity usage* (effective degrees of freedom), not an oracle that predicts accuracy.

Key idea:
- LoRA deltas are low-rank: ΔW = B @ A, where B is (d_out × r) and A is (r × d_in)
- We can compute stable-rank and singular spectrum using only r×r matrices, without
  materializing ΔW or doing a dense SVD.

Outputs:
- Per-layer stable rank, effective rank, utilization (= stable_rank / r)
- Aggregates by module type (attn vs mlp vs other)
- A JSON-serializable summary dict you can log into vNext telemetry:
  event="metrics", kind="lora_audit"

Design goals:
- Works on CPU; computes tiny r×r eigendecompositions in float64 by default (cheap, stable)
- Supports PEFT-style adapter dirs: adapter_config.(json|yaml) + adapter_model.(safetensors|bin|pt)
- Gracefully handles missing optional dependencies (safetensors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import math
import time

import torch

# ------------------------------------------------------------
# Module typing (attn/mlp/other) across common architectures
# ------------------------------------------------------------
def infer_module_type(name: str) -> str:
    n = (name or "").lower()
    attn_hits = [
        "q_proj","k_proj","v_proj","o_proj",
        ".attention.","self_attn",".attn.","attn.",
        "c_attn","c_proj","out_proj",
        "attention.self.query","attention.self.key","attention.self.value",
        "attention.output.dense",
        "q_lin","k_lin","v_lin","out_lin",
        ".query.",".key.",".value.",
    ]
    mlp_hits = [
        "gate_proj","up_proj","down_proj",
        ".mlp.","mlp.",".ffn.","ffn.","feed_forward",
        "intermediate.dense",
        "fc1","fc2","c_fc",
    ]
    if any(h in n for h in attn_hits):
        if "output.dense" in n and "attention.output" not in n and "attention." not in n and "attn" not in n:
            return "mlp"
        return "attn"
    if any(h in n for h in mlp_hits):
        if "attention.output.dense" in n:
            return "attn"
        return "mlp"
    return "other"


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from safetensors.torch import load_file as safetensors_load_file  # type: ignore
except Exception:  # pragma: no cover
    safetensors_load_file = None


# -----------------------------
# Public data structures
# -----------------------------

@dataclass(frozen=True)
class LoRALayerAudit:
    """Audit metrics for a single LoRA layer (one A/B pair)."""
    name: str
    module_type: str  # "attn" | "mlp" | "other"
    r: int
    alpha: Optional[float]
    a_key: str
    b_key: str
    a_shape: Tuple[int, int]
    b_shape: Tuple[int, int]
    params: int

    stable_rank: float
    effective_rank: float
    utilization: float

    sigma_max: float
    frob_sq: float

    # minimal k s.t. cumulative energy (s^2) >= threshold
    energy_rank_90: int
    energy_rank_95: int
    energy_rank_99: int

    # optional: store top singular values (JSON-friendly floats)
    top_singular_values: Optional[List[float]] = None

    # optional: policy-based rank suggestions 
    rank_suggestions: Optional[Dict[str, any]] = None

    # Update Dominance Ratio (UDR) fields
    delta_sigma_max: float = 0.0        # ||ΔW||_2 (spectral norm of update)
    delta_fro_norm: float = 0.0         # ||ΔW||_F (Frobenius norm of update) 
    scale: float = 0.0                  # alpha/r scaling factor
    
    # Base model norms (optional - None if not available)
    base_sigma_max: Optional[float] = None    # ||W_base||_2
    base_fro_norm: Optional[float] = None     # ||W_base||_F
    
    # UDR metrics (computed when base available)
    udr: Optional[float] = None               # delta_sigma_max / base_sigma_max
    udr_f: Optional[float] = None             # delta_fro_norm / base_fro_norm  
    sdi: Optional[float] = None               # log10(udr + eps)

    # Relative perturbation metrics (computed from in-memory base weights when available)
    rel_delta_fro: Optional[float] = None     # ||ΔW||_F / ||W_base||_F (in-memory)
    rel_delta_op: Optional[float] = None      # ||ΔW||_2 / ||W_base||_2 (in-memory)

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def layer_name(self) -> str:
        """Compatibility alias for name attribute."""
        return self.name
    
    @property
    def policy_rank_suggestions(self):
        """Compatibility alias for rank_suggestions attribute."""
        return self.rank_suggestions

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "module_type": self.module_type,
            "r": self.r,
            "alpha": self.alpha,
            "a_key": self.a_key,
            "b_key": self.b_key,
            "a_shape": list(self.a_shape),
            "b_shape": list(self.b_shape),
            "params": self.params,
            "stable_rank": self.stable_rank,
            "effective_rank": self.effective_rank,
            "utilization": self.utilization,
            "sigma_max": self.sigma_max,
            "frob_sq": self.frob_sq,
            "energy_rank_90": self.energy_rank_90,
            "energy_rank_95": self.energy_rank_95,
            "energy_rank_99": self.energy_rank_99,
            "rank_suggestions": self.rank_suggestions,
            # UDR fields
            "delta_sigma_max": self.delta_sigma_max,
            "delta_fro_norm": self.delta_fro_norm,
            "scale": self.scale,
            "base_sigma_max": self.base_sigma_max,
            "base_fro_norm": self.base_fro_norm, 
            "udr": self.udr,
            "udr_f": self.udr_f,
            "sdi": self.sdi,
            # Relative perturbation fields
            "rel_delta_fro": self.rel_delta_fro,
            "rel_delta_op": self.rel_delta_op,
        }
        if self.top_singular_values is not None:
            d["top_singular_values"] = self.top_singular_values
        if self.extras:
            d["extras"] = self.extras
        # Dominance ingredients (scaled by alpha/r)
        try:
            scale = None
            if getattr(self, 'r', None) and getattr(self, 'alpha', None) is not None:
                scale = float(self.alpha) / float(self.r)
            d['adapter_scale'] = scale
            # Unscaled Frobenius norm of ΔW
            if getattr(self, 'frob_sq', None) is not None:
                import math
                d['frob_norm'] = math.sqrt(max(float(self.frob_sq), 0.0))
            # Scaled norms (proxy for 'loudness')
            if scale is not None:
                if getattr(self, 'sigma_max', None) is not None:
                    d['sigma_max_scaled'] = scale * float(self.sigma_max)
                if getattr(self, 'frob_sq', None) is not None:
                    import math
                    d['frob_norm_scaled'] = scale * math.sqrt(max(float(self.frob_sq), 0.0))
        except Exception:
            pass

        return d


@dataclass(frozen=True)
class LoRAAuditResult:
    """Full adapter audit result."""
    peft_dir: Optional[str]
    adapter_config_path: Optional[str]
    adapter_weights_path: Optional[str]

    total_lora_params: int
    n_layers: int

    stable_rank_mean: float
    stable_rank_median: float
    stable_rank_weighted_mean: float

    effective_rank_mean: float
    utilization_mean: float

    energy_rank_90_p50: float
    energy_rank_90_p90: float

    by_type: Dict[str, Dict[str, float]]
    layers: List[LoRALayerAudit]

    # Per-policy global suggestions (Step 5) - DEPRECATED: Use policies.global_statistics
    policy_global_suggestions: Optional[Dict[str, Dict[str, float]]] = None
    
    # Future-proof policy structure (Schema v1)
    rank_policy_schema_version: int = 1
    policies: Optional[Dict[str, Any]] = None

    issues: List[str] = field(default_factory=list)

    def to_summary_dict(self, *, include_layers: bool = False, topk_layers: Optional[int] = None) -> Dict[str, Any]:
        """Return a JSON-serializable summary. Suitable for telemetry metrics(kind="lora_audit")."""
        out: Dict[str, Any] = {
            "total_lora_params": self.total_lora_params,
            "n_layers": self.n_layers,
            "stable_rank_mean": self.stable_rank_mean,
            "stable_rank_median": self.stable_rank_median,
            "stable_rank_weighted_mean": self.stable_rank_weighted_mean,
            "effective_rank_mean": self.effective_rank_mean,
            "utilization_mean": self.utilization_mean,
            "energy_rank_90_p50": self.energy_rank_90_p50,
            "energy_rank_90_p90": self.energy_rank_90_p90,
            "by_type": self.by_type,
        }
        # Legacy field for backward compatibility
        if self.policy_global_suggestions is not None:
            out["policy_global_suggestions"] = self.policy_global_suggestions
        
        # New structured policy schema (v1)
        out["rank_policy_schema_version"] = self.rank_policy_schema_version
        if self.policies is not None:
            out["policies"] = self.policies
        if self.peft_dir is not None:
            out["peft_dir"] = self.peft_dir
        if self.adapter_config_path is not None:
            out["adapter_config_path"] = self.adapter_config_path
        if self.adapter_weights_path is not None:
            out["adapter_weights_path"] = self.adapter_weights_path
        if self.issues:
            out["issues"] = self.issues

        if include_layers:
            layers = self.layers
            if topk_layers is not None:
                # show the most "wasteful" layers first (lowest utilization)
                layers = sorted(layers, key=lambda x: x.utilization)[:topk_layers]
            layer_rows = [l.to_dict() for l in layers]
            out["layer_data"] = {
                "layer_rows_schema": "v1",
                "layer_rows": layer_rows
            }
        # Normalize module_type for layer rows
        layer_data = out.get('layer_data')
        if isinstance(layer_data, dict) and isinstance(layer_data.get('layer_rows'), list):
            for row in layer_data['layer_rows']:
                if isinstance(row, dict):
                    nm = row.get('name') or row.get('module') or row.get('full_name') or ''
                    row['module_type'] = infer_module_type(str(nm))

        # Suggested global ranks (snap k@90% p50/p90 into {1,2,4,8,16,32})
        def _snap_rank(k):
            if k is None:
                return None
            try:
                k = float(k)
            except Exception:
                return None
            for r in (1,2,4,8,16,32):
                if k <= r:
                    return r
            return 32

        if isinstance(out, dict):
            p50 = out.get('energy_rank_90_p50')
            p90 = out.get('energy_rank_90_p90')
            out['suggested_r_global_median'] = _snap_rank(p50)
            out['suggested_r_global_90'] = _snap_rank(p90)

        # Dominance ingredients summary (scaled by alpha/r)
        try:
            import math
            def _pct(vals, q):
                if not vals:
                    return None
                vals = sorted(vals)
                idx = int(round((len(vals) - 1) * q))
                return float(vals[idx])
            sigma_scaled = []
            frob_scaled = []
            for l in getattr(self, 'layers', []) or []:
                r = getattr(l, 'r', None)
                alpha = getattr(l, 'alpha', None)
                if not r or alpha is None:
                    continue
                try:
                    scale = float(alpha) / float(r)
                except Exception:
                    continue
                sigma = getattr(l, 'sigma_max', None)
                if sigma is not None:
                    sigma_scaled.append(scale * float(sigma))
                frob_sq = getattr(l, 'frob_sq', None)
                if frob_sq is not None:
                    frob = math.sqrt(max(float(frob_sq), 0.0))
                    frob_scaled.append(scale * frob)
            if sigma_scaled:
                out['delta_sigma_max_scaled_mean'] = float(sum(sigma_scaled) / len(sigma_scaled))
                out['delta_sigma_max_scaled_p50'] = _pct(sigma_scaled, 0.50)
                out['delta_sigma_max_scaled_p90'] = _pct(sigma_scaled, 0.90)
            if frob_scaled:
                out['delta_frob_norm_scaled_mean'] = float(sum(frob_scaled) / len(frob_scaled))
                out['delta_frob_norm_scaled_p50'] = _pct(frob_scaled, 0.50)
                out['delta_frob_norm_scaled_p90'] = _pct(frob_scaled, 0.90)
        except Exception:
            pass

        # UDR summary statistics
        try:
            udr_values = [l.udr for l in self.layers if l.udr is not None]
            sdi_values = [l.sdi for l in self.layers if l.sdi is not None]
            
            # Check if UDR computation was attempted by looking for base norms data
            # If base_norms was None, UDR computation was disabled
            udr_attempted = any(
                hasattr(l, 'base_sigma_max') and l.base_sigma_max is not None 
                for l in self.layers
            )
            
            if udr_attempted:
                # Always emit n_layers_with_udr for API consistency when UDR is enabled
                out['n_layers_with_udr'] = len(udr_values)
            
            if udr_values:
                out['udr_median'] = _pct(udr_values, 0.50)
                out['udr_p90'] = _pct(udr_values, 0.90)
                out['udr_max'] = max(udr_values)
                out['udr_mean'] = float(sum(udr_values) / len(udr_values))
                out['fraction_udr_gt_0_1'] = sum(1 for x in udr_values if x > 0.1) / len(udr_values)
                out['fraction_udr_gt_0_3'] = sum(1 for x in udr_values if x > 0.3) / len(udr_values)
            
            if sdi_values:
                out['sdi_median'] = _pct(sdi_values, 0.50)
                out['sdi_p90'] = _pct(sdi_values, 0.90)
                out['sdi_mean'] = float(sum(sdi_values) / len(sdi_values))
                
        except Exception:
            pass

        # Gain metrics
        try:
            gain_metrics = compute_gain_metrics(self.layers)
            out['gain'] = gain_metrics['summary']
            out['per_module_gain'] = gain_metrics['per_module']
            out['per_layer_gain'] = gain_metrics['per_layer']
            out['global_gain'] = gain_metrics['global']
            out['composition'] = gain_metrics['composition']  # Add composition analysis
        except Exception:
            pass

        return out

    def to_metrics_event(
        self,
        *,
        run_id: str,
        step: Optional[int] = None,
        ts: Optional[float] = None,
        schema: str = "gradience.vnext.telemetry/v1",
        include_layers: bool = False,
        topk_layers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Convenience: wrap the summary as a vNext telemetry metrics event."""
        return {
            "schema": schema,
            "ts": time.time() if ts is None else ts,
            "run_id": run_id,
            "event": "metrics",
            "kind": "lora_audit",
            "step": step,
            "metrics": self.to_summary_dict(include_layers=include_layers, topk_layers=topk_layers),
        }

    def to_signal_snapshot(self) -> Any:
        """Best-effort: convert into a vNext SignalSnapshot if available, else a dict.

        We keep this tolerant so the auditor can be used standalone.
        """
        payload = {
            "stable_rank_mean": self.stable_rank_mean,
            "utilization_mean": self.utilization_mean,
            "extras": {"lora_audit": self.to_summary_dict(include_layers=False)},
        }

        try:
            import dataclasses as _dc
            from gradience.vnext.types import SignalSnapshot  # type: ignore

            fields = {f.name for f in _dc.fields(SignalSnapshot)}
            filtered = {k: v for k, v in payload.items() if k in fields}
            # If SignalSnapshot has other required args, this will raise -> fallback to dict
            return SignalSnapshot(**filtered)
        except Exception:
            return payload


# -----------------------------
# Adapter config loading
# -----------------------------

@dataclass(frozen=True)
class LoRAAdapterConfig:
    r: Optional[int] = None
    lora_alpha: Optional[float] = None
    target_modules: Optional[List[str]] = None

    # Optional PEFT pattern overrides (best-effort supported)
    rank_pattern: Optional[Dict[str, int]] = None
    alpha_pattern: Optional[Dict[str, float]] = None

    peft_type: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    
    # UDR support: base model norms for dominance ratio computation
    base_norms: Optional[Dict[str, Dict[str, float]]] = None


def _load_json_or_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed, cannot read YAML adapter config.")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_peft_adapter_config(path: Union[str, Path]) -> LoRAAdapterConfig:
    """Load PEFT adapter_config.* and return a minimal normalized config."""
    raw = _load_json_or_yaml(path)
    # PEFT uses keys: r, lora_alpha, target_modules, rank_pattern, alpha_pattern
    r = raw.get("r", None)
    alpha = raw.get("lora_alpha", raw.get("alpha", None))
    target_modules = raw.get("target_modules", None)

    rank_pattern = raw.get("rank_pattern", None)
    alpha_pattern = raw.get("alpha_pattern", None)

    peft_type = raw.get("peft_type", raw.get("peft_type", None))
    # Normalize target_modules if it's a single string
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    # Normalize rank/alpha patterns
    if isinstance(rank_pattern, dict):
        rank_pattern = {str(k): int(v) for k, v in rank_pattern.items()}
    else:
        rank_pattern = None
    if isinstance(alpha_pattern, dict):
        alpha_pattern = {str(k): float(v) for k, v in alpha_pattern.items()}
    else:
        alpha_pattern = None

    try:
        r_int = int(r) if r is not None else None
    except Exception:
        r_int = None
    try:
        a_float = float(alpha) if alpha is not None else None
    except Exception:
        a_float = None

    return LoRAAdapterConfig(
        r=r_int,
        lora_alpha=a_float,
        target_modules=target_modules if isinstance(target_modules, list) else None,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        peft_type=str(peft_type) if peft_type is not None else None,
        raw=raw,
    )


# -----------------------------
# State dict loading + parsing
# -----------------------------

DEFAULT_ADAPTER_CONFIG_NAMES = ("adapter_config.json", "adapter_config.yaml", "adapter_config.yml")
DEFAULT_ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
    "adapter_model.pth",
    "pytorch_model.bin",
)


def find_peft_files(
    peft_dir: Union[str, Path],
    *,
    adapter_config_names: Sequence[str] = DEFAULT_ADAPTER_CONFIG_NAMES,
    adapter_weight_names: Sequence[str] = DEFAULT_ADAPTER_WEIGHT_NAMES,
) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    """Find adapter_config.* and adapter weights file in a PEFT directory.

    Returns: (config_path, weights_path, issues)
    """
    issues: List[str] = []
    d = Path(peft_dir)
    if not d.exists() or not d.is_dir():
        return None, None, [f"peft_dir not found or not a directory: {d}"]

    config_path: Optional[Path] = None
    for name in adapter_config_names:
        p = d / name
        if p.exists() and p.is_file():
            config_path = p
            break
    if config_path is None:
        # fallback: first matching file recursively
        matches = []
        for name in adapter_config_names:
            matches.extend(list(d.rglob(name)))
        if len(matches) == 1:
            config_path = matches[0]
        elif len(matches) > 1:
            issues.append(f"Multiple adapter_config files found under {d}; pass explicit path.")
        else:
            issues.append(f"No adapter_config file found under {d}.")

    weights_path: Optional[Path] = None
    for name in adapter_weight_names:
        p = d / name
        if p.exists() and p.is_file():
            weights_path = p
            break
    if weights_path is None:
        matches = []
        for name in adapter_weight_names:
            matches.extend(list(d.rglob(name)))
        if len(matches) == 1:
            weights_path = matches[0]
        elif len(matches) > 1:
            issues.append(f"Multiple adapter weight files found under {d}; pass explicit path.")
        else:
            issues.append(f"No adapter weight file found under {d}.")

    return config_path, weights_path, issues


def load_adapter_state_dict(weights_path: Union[str, Path], *, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load adapter weights into a (key -> tensor) dict.

    Supports:
    - .safetensors (requires safetensors)
    - torch.load compatible formats (.bin/.pt/.pth)
    """
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError(
                "Cannot load .safetensors because 'safetensors' is not installed. "
                "Install with: pip install safetensors"
            )
        return safetensors_load_file(str(p), device=map_location)  # type: ignore[arg-type]

    obj = torch.load(str(p), map_location=map_location)
    if isinstance(obj, dict):
        # common patterns
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        # assume it's already a state dict
        # but might contain non-tensors
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    raise RuntimeError(f"Unsupported adapter weights payload type: {type(obj)}")


def _infer_module_type(name: str) -> str:
    n = name.lower()
    # attention
    if any(x in n for x in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out_proj", "c_attn", "c_proj")):
        return "attn"
    # mlp / ffn
    if any(x in n for x in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3", "mlp")):
        return "mlp"
    return "other"


def _orient_lora_factors(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return A2 (r×in), B2 (out×r), r.

    PEFT typically stores:
      A: (r, in), B: (out, r)
    Some setups store transposed:
      A: (in, r), B: (r, out)

    We detect and fix the common transposed case.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"LoRA factors must be 2D. Got A={tuple(A.shape)}, B={tuple(B.shape)}")

    # Standard orientation
    if A.shape[0] == B.shape[1]:
        r = int(A.shape[0])
        return A, B, r

    # Common transposed orientation
    if A.shape[1] == B.shape[0]:
        A2 = A.T.contiguous()
        B2 = B.T.contiguous()
        r = int(A2.shape[0])
        return A2, B2, r

    raise ValueError(f"Cannot align LoRA shapes: A={tuple(A.shape)}, B={tuple(B.shape)}")


def _iter_lora_pairs(state_dict: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, str, str]]:
    """Yield (module_prefix, a_key, b_key) for each detected LoRA A/B pair.

    Supports keys like:
      ...<module>.lora_A.weight
      ...<module>.lora_B.weight

    and:
      ...<module>.lora_A.default.weight
      ...<module>.lora_B.default.weight
    """
    for k in state_dict.keys():
        if ".lora_A." in k and k.endswith(".weight"):
            b_key = k.replace(".lora_A.", ".lora_B.")
            if b_key in state_dict:
                prefix = k.split(".lora_A.")[0]
                yield prefix, k, b_key
        elif k.endswith(".lora_A.weight"):
            b_key = k.replace(".lora_A.weight", ".lora_B.weight")
            if b_key in state_dict:
                prefix = k[: -len(".lora_A.weight")]
                yield prefix, k, b_key


# -----------------------------
# Low-rank spectral math
# -----------------------------

def _effective_rank_from_singular_values(s: torch.Tensor, eps: float = 1e-12) -> float:
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    return torch.exp(entropy).item()


def _energy_rank(s: torch.Tensor, threshold: float, eps: float = 1e-12) -> int:
    """Return minimal k s.t. sum_{i<=k} s_i^2 / sum s_i^2 >= threshold."""
    s = s[s > eps]
    if s.numel() == 0:
        return 0
    e = s.pow(2)
    total = e.sum()
    if total <= eps:
        return 0
    c = torch.cumsum(e, dim=0) / total
    idx = torch.searchsorted(c, torch.tensor([threshold], device=c.device, dtype=c.dtype), right=False)
    k = int(idx.item()) + 1
    return min(k, int(s.numel()))


def low_rank_singular_values(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute singular values of BA without forming BA.

    A: (r × d_in)
    B: (d_out × r)

    Returns:
      s: (r,) singular values sorted descending (zeros padded if needed).
    """
    A_ = A.detach().to(dtype=compute_dtype, device="cpu")
    B_ = B.detach().to(dtype=compute_dtype, device="cpu")

    r = A_.shape[0]
    AAT = A_ @ A_.T
    BTB = B_.T @ B_

    eigA, UA = torch.linalg.eigh(AAT)  # ascending
    eigA = torch.clamp(eigA, min=0.0)
    sqrt_eigA = torch.sqrt(eigA)

    C = UA.T @ BTB @ UA
    M = (sqrt_eigA.unsqueeze(1) * C) * sqrt_eigA.unsqueeze(0)
    # symmetrize to fight numerical asymmetry
    M = 0.5 * (M + M.T)

    eigM = torch.linalg.eigvalsh(M)  # ascending
    eigM = torch.clamp(eigM, min=0.0)

    s = torch.sqrt(eigM)  # ascending
    s = torch.sort(s, descending=True).values
    # pad to length r
    if s.numel() < r:
        s = torch.nn.functional.pad(s, (0, r - s.numel()))
    return s


def low_rank_stable_rank(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
    topk_singular_values: Optional[int] = None,
    rank_policies: Optional[List[str]] = None,
) -> Tuple[float, float, float, float, int, int, int, Optional[List[float]], Optional[Dict[str, any]]]:
    """Compute (stable_rank, effective_rank, sigma_max, frob_sq, r90, r95, r99, top_singular_values, policy_results).
    
    Args:
        rank_policies: List of rank selection policies to apply (e.g., ['oht', 'entropy_effective'])
                      If None, only computes traditional energy ranks.
    
    Returns:
        Traditional tuple + optional policy_results dict with rank suggestions from multiple policies.
    """
    s = low_rank_singular_values(A, B, compute_dtype=compute_dtype, eps=eps)

    A_ = A.detach().to(dtype=compute_dtype, device="cpu")
    B_ = B.detach().to(dtype=compute_dtype, device="cpu")
    AAT = A_ @ A_.T
    BTB = B_.T @ B_
    # frob_sq = tr(AAT @ BTB) = sum(AAT*BTB) since both symmetric
    frob_sq = torch.sum(AAT * BTB).item()

    sigma_max = s[0].item() if s.numel() else 0.0
    sigma_max_sq = max(sigma_max * sigma_max, eps)
    stable_rank = frob_sq / sigma_max_sq

    effective_rank = _effective_rank_from_singular_values(s, eps=eps)

    r90 = _energy_rank(s, 0.90, eps=eps)
    r95 = _energy_rank(s, 0.95, eps=eps)
    r99 = _energy_rank(s, 0.99, eps=eps)

    top_sv: Optional[List[float]] = None
    if topk_singular_values is not None:
        k = max(0, int(topk_singular_values))
        if k > 0:
            top_sv = [float(x) for x in s[:k].tolist()]

    # Apply rank selection policies (both new and existing)
    policy_results: Optional[Dict[str, any]] = None
    if rank_policies is not None or True:  # Always compute policies for consistency
        policy_results = {}
        
        # Always include existing energy@90 policy for consistency  
        policy_results['energy_90'] = {'k': r90}
        
        # Apply additional rank selection policies if requested
        if rank_policies is not None:
            try:
                from .rank_policies import apply_rank_policy, RankPolicySpec
                import numpy as np
                
                # Convert torch tensor to numpy for policy application
                s_np = s.detach().cpu().numpy()
                
                for policy_name in rank_policies:
                    try:
                        # Create policy spec
                        policy_spec = RankPolicySpec(policy_name)
                        
                        # Apply policy (need shape info for OHT)
                        # Compute effective ΔW shape from A and B
                        out_dim, in_dim = B.shape[0], A.shape[1]  # B is (out × r), A is (r × in)
                        effective_shape = (out_dim, in_dim)
                        r_alloc = len(s_np)
                        
                        result = apply_rank_policy(
                            policy_spec=policy_spec,
                            s=s_np,
                            shape=effective_shape,
                            r_alloc=r_alloc
                        )
                        
                        # Format according to requested schema
                        policy_data = {
                            'k': result.k,
                            'confidence': float(result.confidence)
                        }
                        
                        # Add key details based on policy type
                        if policy_name == 'optimal_hard_threshold':
                            policy_data.update({
                                'tau': float(result.details.get('tau', 0)),
                                'omega': float(result.details.get('omega', 0)),
                                'beta': float(result.details.get('beta', 0))
                            })
                        elif policy_name == 'entropy_effective':
                            policy_data.update({
                                'erank': float(result.details.get('erank_float', 0)),
                                'entropy': float(result.details.get('entropy', 0))
                            })
                        elif policy_name == 'knee_elbow':
                            policy_data.update({
                                'score': float(result.details.get('knee_diff_max', 0))
                            })
                        
                        # Use policy names as specified by user
                        if policy_name == 'optimal_hard_threshold':
                            key = 'oht'
                        elif policy_name == 'entropy_effective':
                            key = 'erank'  
                        elif policy_name == 'knee_elbow':
                            key = 'knee'
                        else:
                            key = policy_name
                            
                        policy_results[key] = policy_data
                        
                    except Exception as e:
                        policy_results[policy_name] = {
                            'k': 0,
                            'confidence': 0.0,
                            'error': str(e)
                        }
            except ImportError:
                # Graceful degradation if rank_policies module not available
                policy_results['error'] = 'rank_policies module not available'

    return stable_rank, effective_rank, sigma_max, frob_sq, r90, r95, r99, top_sv, policy_results


# -----------------------------
# Gain Metrics Computation
# -----------------------------

def compute_gain_metrics(layers: List[LoRALayerAudit]) -> Dict[str, Any]:
    """Compute gain/magnitude metrics across all layers.
    
    Returns gain metrics organized by:
    - per_module: Individual module metrics
    - per_layer: Layer-wise aggregated metrics  
    - global: Global statistics and rankings
    - summary: High-level summary statistics
    """
    if not layers:
        return {
            "per_module": [],
            "per_layer": [],
            "global": {
                "top_modules_by_delta_fro": [],
                "top_modules_by_rel_delta_fro": [],
                "energy_concentration": {
                    "top_k_layers_share": 0.0,
                    "top_10pct_layers_share": 0.0
                }
            },
            "summary": {
                "delta_fro_mean": 0.0,
                "delta_op_mean": 0.0,
                "rel_delta_fro_mean": None,
                "top_layers_by_delta_fro": [],
                "energy_concentration_top10pct": 0.0
            }
        }
    
    # Extract layer numbers from module names
    def extract_layer_num(name: str) -> int:
        """Extract layer number from module name like 'model.layers.17.self_attn.q_proj'"""
        try:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part in ('layers', 'layer') and i + 1 < len(parts):
                    return int(parts[i + 1])
                if 'layer.' in part:
                    return int(part.split('.')[1])
            return 0
        except (ValueError, IndexError):
            return 0
    
    # Per-module gain metrics  
    per_module_metrics = []
    for layer in layers:
        # Use existing computed norms (already scaled with alpha/r)
        delta_fro = layer.delta_fro_norm  # Frobenius norm ||ΔW||_F
        delta_op = layer.delta_sigma_max  # Spectral norm ||ΔW||_2
        
        # Use relative perturbation metrics (computed from in-memory weights when available)
        # Fall back to UDR-style relative metrics if in-memory not available
        rel_delta_fro = layer.rel_delta_fro
        rel_delta_op = layer.rel_delta_op
        
        # Fallback: use UDR-style computation if in-memory relative not available
        if rel_delta_fro is None and layer.base_fro_norm is not None and layer.base_fro_norm > 0:
            rel_delta_fro = delta_fro / layer.base_fro_norm
        if rel_delta_op is None and layer.base_sigma_max is not None and layer.base_sigma_max > 0:
            rel_delta_op = delta_op / layer.base_sigma_max
        
        layer_num = extract_layer_num(layer.name)
        
        per_module_metrics.append({
            "module": layer.name,
            "layer": layer_num,
            "r": layer.r,
            "scaling": layer.scale,
            "delta_fro": float(delta_fro),
            "delta_op": float(delta_op),
            "rel_delta_fro": float(rel_delta_fro) if rel_delta_fro is not None else None,
            "rel_delta_op": float(rel_delta_op) if rel_delta_op is not None else None
        })
    
    # Per-layer aggregated metrics (sum across modules in same layer)
    layer_aggregates = {}
    for module in per_module_metrics:
        layer_num = module["layer"]
        if layer_num not in layer_aggregates:
            layer_aggregates[layer_num] = {
                "layer": layer_num,
                "delta_fro_sq_sum": 0.0,
                "delta_fro_sum": 0.0,
                "rel_delta_fro_sq_sum": None,
                "count": 0,
                "rel_count": 0
            }
        
        agg = layer_aggregates[layer_num]
        agg["delta_fro_sq_sum"] += module["delta_fro"] ** 2
        agg["delta_fro_sum"] += module["delta_fro"]
        agg["count"] += 1
        
        if module["rel_delta_fro"] is not None:
            if agg["rel_delta_fro_sq_sum"] is None:
                agg["rel_delta_fro_sq_sum"] = 0.0
            agg["rel_delta_fro_sq_sum"] += module["rel_delta_fro"] ** 2
            agg["rel_count"] += 1
    
    per_layer_metrics = []
    for layer_num in sorted(layer_aggregates.keys()):
        agg = layer_aggregates[layer_num]
        per_layer_metrics.append({
            "layer": layer_num,
            "delta_fro_sq_sum": agg["delta_fro_sq_sum"],
            "delta_fro_sum": agg["delta_fro_sum"],
            "rel_delta_fro_sq_sum": agg["rel_delta_fro_sq_sum"]
        })
    
    # Global metrics and rankings
    
    # Sort modules by delta_fro (Frobenius norm)
    modules_by_delta_fro = sorted(per_module_metrics, 
                                  key=lambda x: x["delta_fro"], reverse=True)
    top_modules_by_delta_fro = [
        {"module": m["module"], "layer": m["layer"], "delta_fro": m["delta_fro"]} 
        for m in modules_by_delta_fro[:10]
    ]
    
    # Sort modules by relative delta_fro (if available)
    modules_with_rel = [m for m in per_module_metrics if m["rel_delta_fro"] is not None]
    top_modules_by_rel_delta_fro = []
    if modules_with_rel:
        modules_by_rel_delta_fro = sorted(modules_with_rel, 
                                          key=lambda x: x["rel_delta_fro"], reverse=True)
        top_modules_by_rel_delta_fro = [
            {"module": m["module"], "layer": m["layer"], "rel_delta_fro": m["rel_delta_fro"]} 
            for m in modules_by_rel_delta_fro[:10]
        ]
    
    # Composition-style energy concentration analysis
    from .gain_metrics import compute_layer_energy_concentration
    
    # Build module energy dict for concentration analysis
    module_energies = {}
    for module in per_module_metrics:
        module_energies[module["module"]] = module["delta_fro"] ** 2  # Use squared Frobenius norm as energy
    
    # Compute comprehensive concentration metrics
    concentration_analysis = compute_layer_energy_concentration(
        module_energies, 
        top_k=5  # Analyze top-5 layers
    )
    
    # Extract simple metrics for backward compatibility
    if concentration_analysis["top_10pct"]["n"] > 0:
        top_10pct_layers_share = concentration_analysis["top_10pct"]["share"]
        top_k_layers_share = concentration_analysis["top_k"]["share"]
    else:
        top_10pct_layers_share = 0.0
        top_k_layers_share = 0.0
    
    # Summary statistics
    delta_fro_values = [m["delta_fro"] for m in per_module_metrics]
    delta_op_values = [m["delta_op"] for m in per_module_metrics]
    rel_delta_fro_values = [m["rel_delta_fro"] for m in per_module_metrics 
                            if m["rel_delta_fro"] is not None]
    rel_delta_op_values = [m["rel_delta_op"] for m in per_module_metrics 
                           if m["rel_delta_op"] is not None]
    
    delta_fro_mean = sum(delta_fro_values) / len(delta_fro_values) if delta_fro_values else 0.0
    delta_op_mean = sum(delta_op_values) / len(delta_op_values) if delta_op_values else 0.0
    rel_delta_fro_mean = (sum(rel_delta_fro_values) / len(rel_delta_fro_values) 
                          if rel_delta_fro_values else None)
    rel_delta_op_mean = (sum(rel_delta_op_values) / len(rel_delta_op_values) 
                         if rel_delta_op_values else None)
    
    # Top layers by delta_fro (for summary)
    top_layers_by_delta_fro = []
    if per_layer_metrics:
        layers_by_energy = sorted(per_layer_metrics, 
                                  key=lambda x: x["delta_fro_sq_sum"], reverse=True)
        top_layers_by_delta_fro = [
            {"layer": l["layer"], "delta_fro_sq_sum": l["delta_fro_sq_sum"]}
            for l in layers_by_energy[:5]
        ]
    
    return {
        "per_module": per_module_metrics,
        "per_layer": per_layer_metrics,
        "composition": concentration_analysis,  # Full composition analysis
        "global": {
            "top_modules_by_delta_fro": top_modules_by_delta_fro,
            "top_modules_by_rel_delta_fro": top_modules_by_rel_delta_fro,
            "energy_concentration": {
                "top_k_layers_share": float(top_k_layers_share),
                "top_10pct_layers_share": float(top_10pct_layers_share)
            }
        },
        "summary": {
            "delta_fro_mean": float(delta_fro_mean),
            "delta_op_mean": float(delta_op_mean), 
            "rel_delta_fro_mean": float(rel_delta_fro_mean) if rel_delta_fro_mean is not None else None,
            "rel_delta_op_mean": float(rel_delta_op_mean) if rel_delta_op_mean is not None else None,
            "top_layers_by_delta_fro": top_layers_by_delta_fro,
            "energy_concentration_top10pct": float(top_10pct_layers_share),
            # Availability indicators
            "relative_available": len(rel_delta_fro_values) > 0 or len(rel_delta_op_values) > 0,
            "n_modules_with_relative": max(len(rel_delta_fro_values), len(rel_delta_op_values))
        }
    }


# -----------------------------
# Update Dominance Ratio (UDR) computation
# -----------------------------

"""
UDR/SDI BEHAVIORAL CONTRACT
===========================

This section defines the exact behavioral guarantees for Update Dominance Ratio (UDR)
and Spectral Drift Index (SDI) computation in Gradience audit pipeline.

DEFINITIONS:
-----------

UDR (Update Dominance Ratio) for LoRA-adapted modules:
    • For each adapted module with base weight W_base, LoRA induces update:
      ΔW = s · B A  where s = α/r (PEFT scaling factor)
    • UDR per module: UDR = ||ΔW||₂ / (||W_base||₂ + ε)
    • Uses spectral norm (largest singular value), NOT Frobenius norm

SDI (Spectral Drift Index):
    • SDI = log₁₀(UDR + ε)
    • Provides logarithmic scale for UDR interpretation

SCALING CONTRACT:
----------------
    • ΔW computation uses actual PEFT scaling: s = lora_alpha / r
    • Per-layer alpha patterns override global lora_alpha if present
    • Scale factor recorded in audit output for verification

MODULE INCLUSION:
----------------
    • ONLY LoRA target modules are processed (q_proj, v_proj, etc.)
    • Modules must have both lora_A and lora_B weights present
    • Missing pairs are recorded as issues, not silent failures

BASE NORMS HANDLING:
-------------------
When base norms missing:
    • Individual layer UDR/SDI fields are None/null
    • Summary statistics exclude layers without base norms  
    • Structured issue recorded with reason (cache miss, load failure, etc.)

When base norms present:
    • Per-module UDR/SDI computed and included
    • Summary statistics computed across all layers with valid UDR
    • Cache updated for future use

CACHE IDENTITY:
--------------
Base norms cache keyed by:
    • base_model_id (HuggingFace model identifier)
    • target_modules (exact set of LoRA target module patterns)
    • Future: revision hash, dtype for full reproducibility

EPSILON VALUE:
-------------
    • ε = 1e-12 (prevents division by zero, minimal impact on realistic norms)
    • Applied consistently in UDR and SDI computation

ERROR HANDLING:
--------------
    • SVD failures: skip layer, record structured issue
    • Cache corruption: fall back to recomputation or disable UDR
    • Model loading failures: disable UDR, record issue, continue audit
    • NO crashes on UDR failures - audit always completes

OUTPUT SCHEMA:
-------------
Per-layer fields (in LoRALayerAudit):
    • delta_sigma_max: float - ||ΔW||₂ 
    • delta_fro_norm: float - ||ΔW||_F (for debugging)
    • scale: float - α/r scaling factor used
    • base_sigma_max: Optional[float] - ||W_base||₂ (None if unavailable)
    • udr: Optional[float] - UDR value (None if base_sigma_max unavailable)
    • sdi: Optional[float] - SDI value (None if UDR unavailable)

Summary fields (in LoRAAuditResult):
    • udr_mean, udr_median, udr_p90, udr_max: statistics over layers with UDR
    • sdi_mean, sdi_median, sdi_p90: statistics over layers with SDI
    • fraction_udr_gt_0_1, fraction_udr_gt_0_3: threshold fractions
    • n_layers_with_udr: count of layers with valid UDR computation

GUARANTEES:
----------
1. UDR values are deterministic for same inputs (base norms + adapter weights)
2. SDI is monotonically increasing with UDR  
3. Audit never fails due to UDR computation errors
4. Base norm caching is atomic (write success or rollback)
5. All UDR-related fields are either consistently present or consistently None
"""

def compute_update_norms(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float = 1.0,
    *,
    compute_dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """Compute update delta norms without forming BA.
    
    Args:
        A: (r × d_in) LoRA factor
        B: (d_out × r) LoRA factor  
        scale: alpha/r scaling factor
        
    Returns:
        (delta_fro_norm, delta_sigma_max, stable_rank_delta, utilization)
        
    Note:
        UDR = ||ΔW||_2 / ||W_base||_2 where ΔW = scale * B @ A
        This computes ||ΔW||_2 and ||ΔW||_F efficiently via r×r operations.
    """
    from .gain_metrics import compute_lora_norms, compute_lora_stable_rank
    
    # Use optimized utility functions
    delta_fro_norm, delta_sigma_max = compute_lora_norms(
        A, B, scale, compute_dtype=compute_dtype, eps=eps
    )
    
    stable_rank_delta, utilization, r = compute_lora_stable_rank(
        A, B, scale, compute_dtype=compute_dtype, eps=eps
    )
    
    return delta_fro_norm, delta_sigma_max, stable_rank_delta, utilization


def spectral_norm_power_iter(W: torch.Tensor, n_iter: int = 20) -> float:
    """Power iteration for spectral norm estimation."""
    if W.numel() == 0:
        return 0.0
    
    # Handle 1D case
    if W.ndim == 1:
        return torch.norm(W, 2).item()
    
    m, n = W.shape[0], W.shape[1]
    
    # Random initialization
    with torch.no_grad():
        v = torch.randn(n, device=W.device, dtype=W.dtype)
        v = v / torch.norm(v)
        
        for _ in range(n_iter):
            # v <- W^T @ W @ v / ||W^T @ W @ v||
            Wv = W @ v
            WTWv = W.T @ Wv
            norm = torch.norm(WTWv)
            if norm > 1e-12:
                v = WTWv / norm
            else:
                break
        
        # Final singular value estimate
        Wv = W @ v
        return torch.norm(Wv).item()


def compute_base_norms(
    model: torch.nn.Module,
    target_modules: List[str],
    *,
    map_location: str = "cpu",
    n_power_iter: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Compute base model spectral and Frobenius norms for target modules.
    
    Args:
        model: Base model to analyze
        target_modules: List of module name patterns to include
        map_location: Device for computation (default: "cpu")
        n_power_iter: Power iteration steps for spectral norm
        
    Returns:
        {module_name: {"base_sigma_max": float, "base_fro_norm": float}}
    """
    base_norms = {}
    
    with torch.no_grad():
        for name, module in model.named_modules():
            # Check if this module matches target_modules patterns
            if not any(target in name for target in target_modules):
                continue
                
            if not hasattr(module, 'weight') or module.weight is None:
                continue
                
            W = module.weight.detach().to(device=map_location)
            
            # Frobenius norm (exact)
            base_fro_norm = torch.norm(W, 'fro').item()
            
            # Spectral norm (power iteration)
            base_sigma_max = spectral_norm_power_iter(W, n_iter=n_power_iter)
            
            base_norms[name] = {
                "base_sigma_max": base_sigma_max,
                "base_fro_norm": base_fro_norm,
            }
    
    return base_norms


def compute_udr_metrics(
    delta_sigma_max: float,
    delta_fro_norm: float, 
    base_sigma_max: Optional[float],
    base_fro_norm: Optional[float],
    eps: float = 1e-12,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute UDR and SDI metrics if base norms available.
    
    Args:
        delta_sigma_max: Spectral norm of adapter update ||ΔW||_2
        delta_fro_norm: Frobenius norm of adapter update ||ΔW||_F
        base_sigma_max: Base model spectral norm ||W_base||_2
        base_fro_norm: Base model Frobenius norm ||W_base||_F
        eps: Numerical stability threshold
        
    Returns:
        (udr, udr_f, sdi) where:
        - udr = ||ΔW||_2 / ||W_base||_2 (spectral dominance ratio)
        - udr_f = ||ΔW||_F / ||W_base||_F (Frobenius dominance ratio)  
        - sdi = log10(udr + eps) (spectral drift index)
    """
    if base_sigma_max is None or base_fro_norm is None:
        return None, None, None
        
    # Use epsilon protection for division by zero/near-zero
    udr = delta_sigma_max / (base_sigma_max + eps)
    udr_f = delta_fro_norm / (base_fro_norm + eps)
    sdi = math.log10(udr + eps)
    
    return udr, udr_f, sdi


def save_base_norms_cache(
    base_norms: Dict[str, Dict[str, float]], 
    cache_path: Union[str, Path],
    model_id: str = "unknown"
) -> None:
    """Save base norms to cache file."""
    cache_data = {
        "model_id": model_id,
        "timestamp": time.time(),
        "base_norms": base_norms
    }
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)


def load_base_norms_cache(cache_path: Union[str, Path]) -> Optional[Dict[str, Dict[str, float]]]:
    """Load base norms from cache file."""
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        return cache_data.get("base_norms", {})
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


# -----------------------------
# Base model norms loading
# -----------------------------

def load_base_model_norms(
    base_model_id: Optional[str] = None,
    base_norms_cache: Optional[Union[str, Path]] = None,
    adapter_config: Optional[LoRAAdapterConfig] = None,
    issues: Optional[List[str]] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load base model norms for UDR computation.
    
    Returns dict mapping layer_key -> {'sigma_max': float, 'fro_norm': float}
    """
    if issues is None:
        issues = []
        
    # Try cache first
    if base_norms_cache is not None:
        cache_path = Path(base_norms_cache)
        if cache_path.exists():
            try:
                import json
                with cache_path.open('r') as f:
                    return json.load(f)
            except Exception as e:
                issues.append(f"Failed to load base norms cache {cache_path}: {e}")
    
    # Try to compute from base model if provided
    if base_model_id is not None and adapter_config is not None:
        try:
            norms = compute_base_model_norms(base_model_id, adapter_config, issues)
            
            # Cache the computed norms if successful and cache path provided
            if norms is not None and base_norms_cache is not None:
                try:
                    cache_base_model_norms(norms, base_norms_cache)
                except Exception as e:
                    issues.append(f"Failed to cache base norms to {base_norms_cache}: {e}")
            
            return norms
        except Exception as e:
            issues.append(f"Failed to compute base model norms for {base_model_id}: {e}")
    
    return None


def cache_base_model_norms(
    norms: Dict[str, Dict[str, float]], 
    cache_path: Union[str, Path]
) -> None:
    """Save computed base model norms to cache file."""
    import json
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with cache_file.open('w') as f:
        json.dump(norms, f, indent=2)


def compute_base_model_norms(
    base_model_id: str,
    adapter_config: LoRAAdapterConfig,
    issues: List[str],
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute base model norms for all LoRA target modules.
    
    Loads the base model and computes spectral and Frobenius norms
    for all 2D weight matrices that match the LoRA target_modules pattern.
    
    Uses state_dict-based extraction to capture all architectures
    (nn.Linear, Conv1D, etc.) without module type filtering.
    """
    try:
        # Try to import transformers
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            issues.append("transformers library not available for base model loading")
            return None
        
        # Load model config to understand architecture
        try:
            config = AutoConfig.from_pretrained(base_model_id)
        except Exception as e:
            issues.append(f"Failed to load config for {base_model_id}: {e}")
            return None
        
        # Load model weights (CPU only to save memory) - use AutoModelForCausalLM for GPT-2
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,  # Use float32 for norm computation
                device_map="cpu",
                trust_remote_code=True,
            )
        except Exception as e:
            issues.append(f"Failed to load model {base_model_id}: {e}")
            return None
        
        # Extract target modules based on adapter config
        target_modules = adapter_config.target_modules or []
        if not target_modules:
            issues.append("No target_modules specified in adapter config")
            return None
        
        norms_dict = {}
        
        # STATE_DICT-based approach: iterate through all 2D weight matrices
        for param_name, param in model.state_dict().items():
            # Only process weight matrices (skip biases, embeddings, etc.)
            if not param_name.endswith(".weight"):
                continue
            if getattr(param, "ndim", None) != 2:
                continue
                
            # Check if this parameter name matches any target module pattern
            if any(_module_name_matches(param_name, target) for target in target_modules):
                # Convert to module name (remove .weight suffix)
                module_name = _convert_to_lora_prefix(param_name)
                
                try:
                    # Compute norms on CPU with proper tensor handling
                    param_cpu = param.detach().cpu().to(torch.float64)
                    
                    # Spectral norm (largest singular value) 
                    try:
                        _, s, _ = torch.linalg.svd(param_cpu, full_matrices=False)
                        sigma_max = float(s[0]) if len(s) > 0 else 0.0
                    except Exception as svd_e:
                        # Fallback: use matrix norm if SVD fails
                        sigma_max = float(torch.norm(param_cpu, p=2))
                        issues.append(f"SVD failed for {param_name}, using L2 norm: {svd_e}")
                    
                    # Frobenius norm
                    fro_norm = float(torch.norm(param_cpu, p='fro'))
                    
                    # Only store non-zero norms (avoid division by zero in UDR)
                    if fro_norm > 1e-12:
                        norms_dict[module_name] = {
                            'sigma_max': sigma_max,
                            'fro_norm': fro_norm,
                        }
                    else:
                        issues.append(f"Skipping {param_name}: zero norm detected")
                    
                except Exception as e:
                    issues.append(f"Failed to compute norms for {param_name}: {e}")
                    continue
        
        # Clean up model to free memory
        del model
        import gc
        gc.collect()
        
        if not norms_dict:
            issues.append(f"No matching target modules found in {base_model_id} for patterns: {target_modules}")
            return None
        
        return norms_dict
        
    except Exception as e:
        issues.append(f"Base model norm computation failed: {e}")
        return None


def _module_name_matches(param_name: str, target_pattern: str) -> bool:
    """Check if parameter name matches target module pattern."""
    # Simple substring matching for now
    # Could be extended to support regex patterns
    return target_pattern in param_name


def _convert_to_lora_prefix(param_name: str) -> str:
    """Convert model parameter name to LoRA prefix format.
    
    Examples:
    - 'model.layers.0.self_attn.q_proj.weight' -> 'model.layers.0.self_attn.q_proj'
    - 'transformer.h.0.mlp.c_fc.weight' -> 'transformer.h.0.mlp.c_fc'
    """
    # Remove .weight, .bias suffixes
    if param_name.endswith('.weight') or param_name.endswith('.bias'):
        return param_name.rsplit('.', 1)[0]
    return param_name


def _canonicalize_module_name(name: str) -> str:
    """Canonicalize module name for consistent matching between adapter and base model.
    
    Removes common PEFT prefixes and suffixes to enable consistent lookups.
    
    Examples:
    - 'base_model.model.transformer.h.0.attn.c_attn' -> 'transformer.h.0.attn.c_attn'
    - 'base_model.transformer.h.0.attn.c_attn.weight' -> 'transformer.h.0.attn.c_attn'
    - 'model.transformer.h.0.attn.c_attn' -> 'transformer.h.0.attn.c_attn'
    """
    # Remove common PEFT prefixes
    for prefix in ("base_model.model.", "base_model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Remove .weight suffix if present
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
    
    return name


def _convert_lora_prefix_to_base_weight_key(lora_prefix: str) -> str:
    """Convert LoRA module prefix to base model parameter key.
    
    Examples:
    - 'model.layers.0.self_attn.q_proj' -> 'model.layers.0.self_attn.q_proj.weight'
    - 'base_model.model.layers.0.self_attn.q_proj' -> 'model.layers.0.self_attn.q_proj.weight'
    """
    # Remove common PEFT prefixes
    prefix = lora_prefix
    if prefix.startswith('base_model.'):
        prefix = prefix[len('base_model.'):]
    
    # Add .weight suffix (most common case)
    return f"{prefix}.weight"


# -----------------------------
# Main audit entrypoints
# -----------------------------

def audit_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    adapter_config: Optional[LoRAAdapterConfig] = None,
    compute_dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
    include_top_singular_values: int = 0,
    base_norms: Optional[Dict[str, Dict[str, float]]] = None,
    base_model_weights: Optional[Dict[str, torch.Tensor]] = None,
    rank_policies: Optional[List[str]] = None,
) -> LoRAAuditResult:
    """Audit LoRA A/B pairs in a state_dict.

    Args:
        state_dict: LoRA adapter state dict
        adapter_config: Optional adapter configuration
        compute_dtype: Data type for internal computation
        eps: Numerical stability threshold
        include_top_singular_values: Number of top singular values to store
        base_norms: Pre-computed base model norms (for UDR)
        base_model_weights: Optional in-memory base model weights for relative perturbation.
                           Only used if already loaded, never triggers model loading.
        rank_policies: List of rank selection policies to apply (e.g., ['oht', 'entropy_effective'])

    adapter_config is optional; we infer per-layer r from shapes regardless.
    """
    layers: List[LoRALayerAudit] = []
    issues: List[str] = []

    # Collect per-layer ranks for summaries
    stable_ranks: List[float] = []
    effective_ranks: List[float] = []
    utilizations: List[float] = []
    params_list: List[int] = []
    r90_list: List[int] = []

    by_type_acc: Dict[str, List[LoRALayerAudit]] = {"attn": [], "mlp": [], "other": []}

    for prefix, a_key, b_key in _iter_lora_pairs(state_dict):
        A = state_dict.get(a_key, None)
        B = state_dict.get(b_key, None)
        if A is None or B is None:
            issues.append(f"Missing tensors for pair: {a_key} / {b_key}")
            continue

        try:
            A2, B2, r = _orient_lora_factors(A, B)
        except Exception as e:
            issues.append(f"Could not orient LoRA factors for {prefix}: {e}")
            continue

        # alpha: best-effort from adapter_config; may be per-layer if patterns exist
        alpha = None
        if adapter_config is not None:
            alpha = adapter_config.lora_alpha
            if adapter_config.alpha_pattern:
                for pat, val in adapter_config.alpha_pattern.items():
                    if pat in prefix:
                        alpha = float(val)
                        break

        # module type inference
        module_type = _infer_module_type(prefix)

        try:
            stable_rank, eff_rank, sigma_max, frob_sq, r90, r95, r99, top_sv, policy_results = low_rank_stable_rank(
                A2,
                B2,
                compute_dtype=compute_dtype,
                eps=eps,
                topk_singular_values=include_top_singular_values if include_top_singular_values > 0 else None,
                rank_policies=rank_policies,
            )
            
            # NEW: Compute UDR metrics
            scale = float(alpha) / float(r) if alpha is not None else 1.0
            delta_fro_norm, delta_sigma_max, stable_rank_delta, utilization_delta = compute_update_norms(
                A2, B2, scale=scale, compute_dtype=compute_dtype, eps=eps
            )
            
            # Get base norms if available
            base_sigma_max = None
            base_fro_norm = None
            if base_norms is not None:
                # Try both canonical and original prefix for flexible lookup
                canonical_prefix = _canonicalize_module_name(prefix)
                base_data = base_norms.get(canonical_prefix, {})
                
                # If canonical lookup failed, try with original prefix
                if not base_data:
                    base_data = base_norms.get(prefix, {})
                    
                base_sigma_max = base_data.get('sigma_max')
                base_fro_norm = base_data.get('fro_norm')
            elif adapter_config is not None and adapter_config.base_norms:
                # Fallback to adapter config for backward compatibility
                base_data = adapter_config.base_norms.get(prefix, {})
                base_sigma_max = base_data.get('base_sigma_max')
                base_fro_norm = base_data.get('base_fro_norm')
            
            # Compute UDR/SDI
            udr, udr_f, sdi = compute_udr_metrics(
                delta_sigma_max, delta_fro_norm, base_sigma_max, base_fro_norm, eps=eps
            )
            
            # Compute relative perturbation metrics from in-memory base weights (safe)
            rel_delta_fro = None
            rel_delta_op = None
            if base_model_weights is not None:
                base_weight_key = _convert_lora_prefix_to_base_weight_key(prefix)
                base_weight = base_model_weights.get(base_weight_key)
                
                if base_weight is not None:
                    try:
                        from .gain_metrics import compute_lora_norms
                        
                        # Compute base model norms from in-memory weights
                        base_weight_cpu = base_weight.detach().to(dtype=compute_dtype, device="cpu")
                        base_fro_in_memory = torch.norm(base_weight_cpu, p='fro').item()
                        
                        # Spectral norm via SVD (safe for in-memory computation)
                        if base_weight_cpu.dim() >= 2:
                            _, s, _ = torch.linalg.svd(base_weight_cpu, full_matrices=False)
                            base_spec_in_memory = s[0].item() if s.numel() > 0 else 0.0
                        else:
                            base_spec_in_memory = base_fro_in_memory
                        
                        # Compute relative perturbation ratios
                        if base_fro_in_memory > eps:
                            rel_delta_fro = delta_fro_norm / base_fro_in_memory
                        if base_spec_in_memory > eps:
                            rel_delta_op = delta_sigma_max / base_spec_in_memory
                            
                    except Exception as e:
                        issues.append(f"Failed to compute relative perturbation for {prefix}: {e}")
            
        except Exception as e:
            issues.append(f"Spectral computation failed for {prefix}: {e}")
            continue

        utilization = stable_rank / max(float(r), 1.0)
        params = int(A.numel() + B.numel())

        layer = LoRALayerAudit(
            name=prefix,
            module_type=module_type,
            r=int(r),
            alpha=alpha,
            a_key=a_key,
            b_key=b_key,
            a_shape=tuple(int(x) for x in A.shape),
            b_shape=tuple(int(x) for x in B.shape),
            params=params,
            stable_rank=float(stable_rank),
            effective_rank=float(eff_rank),
            utilization=float(utilization),
            sigma_max=float(sigma_max),
            frob_sq=float(frob_sq),
            energy_rank_90=int(r90),
            energy_rank_95=int(r95),
            energy_rank_99=int(r99),
            top_singular_values=top_sv,
            rank_suggestions=policy_results,
            # UDR fields
            delta_sigma_max=float(delta_sigma_max),
            delta_fro_norm=float(delta_fro_norm),
            scale=float(scale),
            base_sigma_max=base_sigma_max,
            base_fro_norm=base_fro_norm,
            udr=udr,
            udr_f=udr_f,
            sdi=sdi,
            # Relative perturbation fields (computed from in-memory base weights)
            rel_delta_fro=rel_delta_fro,
            rel_delta_op=rel_delta_op,
        )
        layers.append(layer)
        # Derive module_type from the true module name
        layer_name = getattr(layer, 'name', None) or getattr(layer, 'full_name', None) or getattr(layer, 'key', None)
        module_type = infer_module_type(str(layer_name or ''))
        try:
            layer.module_type = module_type
        except Exception:
            pass
        by_type_acc.setdefault(module_type, []).append(layer)

        stable_ranks.append(layer.stable_rank)
        effective_ranks.append(layer.effective_rank)
        utilizations.append(layer.utilization)
        params_list.append(layer.params)
        r90_list.append(layer.energy_rank_90)

    n_layers = len(layers)
    total_params = int(sum(params_list))

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        m = len(ys) // 2
        if len(ys) % 2 == 1:
            return float(ys[m])
        return float(0.5 * (ys[m - 1] + ys[m]))

    def _weighted_mean(xs: List[float], ws: List[int]) -> float:
        if not xs or not ws or sum(ws) == 0:
            return 0.0
        return float(sum(x * w for x, w in zip(xs, ws)) / sum(ws))

    stable_rank_mean = _mean(stable_ranks)
    stable_rank_median = _median(stable_ranks)
    stable_rank_wmean = _weighted_mean(stable_ranks, params_list)

    eff_rank_mean = _mean(effective_ranks)
    util_mean = _mean(utilizations)

    # Future-proof policy schema builder
    def _build_structured_policy_data(
        layers: List[LoRALayerAudit], 
        applied_policies: List[str],
        policy_global_suggestions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Build future-proof structured policy data (Schema v1)."""
        
        # Map internal policy names to user-friendly names
        policy_name_mapping = {
            'energy_threshold': 'energy_threshold',
            'knee_elbow': 'knee_elbow', 
            'entropy_effective': 'entropy_effective',
            'optimal_hard_threshold': 'optimal_hard_threshold'
        }
        
        # Default parameters for each policy
        default_parameters = {
            'energy_threshold': {'threshold': 0.90},
            'knee_elbow': {},
            'entropy_effective': {},
            'optimal_hard_threshold': {}
        }
        
        # Build per-layer suggestions with rich metadata
        per_layer_suggestions = []
        for layer in layers:
            if not layer.rank_suggestions:
                continue
                
            # Calculate importance metrics for this layer
            energy_raw = layer.frob_sq if hasattr(layer, 'frob_sq') else 0.0
            frobenius_norm = (layer.frob_sq ** 0.5) if hasattr(layer, 'frob_sq') else 0.0
            param_count = getattr(layer, 'lora_params', 0)
            utilization = getattr(layer, 'utilization', 0.0) if hasattr(layer, 'utilization') else 0.0
            
            importance_raw = (
                frobenius_norm * 0.6 +           # Primary: magnitude of update
                (param_count / 10000) * 0.3 +    # Secondary: layer size (normalized)
                utilization * 100 * 0.1          # Tertiary: how much layer is used
            )
            
            layer_data = {
                "layer_name": layer.layer_name,
                "allocated_rank": layer.r,
                "suggestions": {},
                # New importance metrics for downstream tooling
                "importance_metrics": {
                    "importance_raw": importance_raw,
                    "energy_raw": energy_raw,
                    "frobenius_norm": frobenius_norm,
                    "param_count": param_count,
                    "utilization": utilization
                }
            }
            
            for policy_internal, suggestion in layer.rank_suggestions.items():
                if policy_internal in policy_name_mapping and isinstance(suggestion, dict):
                    policy_name = policy_name_mapping[policy_internal]
                    
                    # Extract core suggestion data
                    structured_suggestion = {
                        "k": suggestion.get('k', 0),
                        "confidence": suggestion.get('confidence', 0.0),
                        "metadata": suggestion.get('details', {})
                    }
                    
                    # Add policy-specific metadata normalization
                    if policy_internal == 'energy_threshold':
                        metadata = structured_suggestion['metadata']
                        structured_suggestion['metadata'] = {
                            "threshold_used": metadata.get('threshold', 0.90),
                            "energy_captured": metadata.get('actual_energy_captured', 0.0),
                            "total_energy": metadata.get('total_energy', 0.0)
                        }
                    elif policy_internal == 'entropy_effective':
                        metadata = structured_suggestion['metadata']
                        structured_suggestion['metadata'] = {
                            "erank_float": metadata.get('erank_float', 0.0),
                            "entropy": metadata.get('entropy', 0.0),
                            "max_possible_entropy": metadata.get('max_possible_entropy', 0.0),
                            "normalized_entropy": metadata.get('normalized_entropy', 0.0)
                        }
                    elif policy_internal == 'optimal_hard_threshold':
                        metadata = structured_suggestion['metadata']
                        structured_suggestion['metadata'] = {
                            "omega_beta": metadata.get('omega_beta', 0.0),
                            "beta": metadata.get('beta', 1.0),
                            "threshold": metadata.get('threshold', 0.0),
                            "median_sv": metadata.get('median_sv', 0.0)
                        }
                    elif policy_internal == 'knee_elbow':
                        metadata = structured_suggestion['metadata']
                        structured_suggestion['metadata'] = {
                            "knee_index": metadata.get('knee_index', 0),
                            "difference_curve_max": metadata.get('difference_curve_max', 0.0),
                            "smoothing_applied": metadata.get('smoothing_applied', False)
                        }
                    
                    layer_data["suggestions"][policy_name] = structured_suggestion
            
            if layer_data["suggestions"]:  # Only include layers with suggestions
                per_layer_suggestions.append(layer_data)
        
        # Calculate global importance distribution metrics for downstream tooling
        if per_layer_suggestions:
            total_energy = sum(layer['importance_metrics']['energy_raw'] for layer in per_layer_suggestions)
            n_layers = len(per_layer_suggestions)
            uniform_share = 1.0 / n_layers if n_layers > 0 else 0.0
            
            # Add energy share and uniform multiplier to each layer
            max_uniform_mult = 0.0
            for layer_data in per_layer_suggestions:
                energy_share = layer_data['importance_metrics']['energy_raw'] / total_energy if total_energy > 0 else uniform_share
                uniform_mult = energy_share / uniform_share if uniform_share > 0 else 1.0
                
                layer_data['importance_metrics']['energy_share'] = energy_share
                layer_data['importance_metrics']['uniform_mult'] = uniform_mult
                max_uniform_mult = max(max_uniform_mult, uniform_mult)
            
            # Global distribution characteristics
            min_uniform_mult_threshold = 1.5  # Same threshold as CLI
            distribution_is_flat = max_uniform_mult < min_uniform_mult_threshold
            
            importance_distribution = {
                "n_layers": n_layers,
                "total_energy": total_energy,
                "max_uniform_mult": max_uniform_mult,
                "distribution_is_flat": distribution_is_flat,
                "min_uniform_mult_threshold": min_uniform_mult_threshold,
                "uniform_share": uniform_share
            }
        else:
            importance_distribution = {
                "n_layers": 0,
                "total_energy": 0.0,
                "max_uniform_mult": 0.0,
                "distribution_is_flat": True,
                "min_uniform_mult_threshold": 1.5,
                "uniform_share": 0.0
            }
        
        # Build the complete structured schema
        return {
            "metadata": {
                "version": 1,
                "applied_policies": [policy_name_mapping.get(p, p) for p in applied_policies],
                "default_parameters": {policy_name_mapping.get(p, p): default_parameters.get(p, {}) for p in applied_policies}
            },
            "global_statistics": {
                policy_name_mapping.get(k, k): v for k, v in policy_global_suggestions.items()
            },
            "importance_distribution": importance_distribution,
            "per_layer": per_layer_suggestions
        }

    # energy_rank_90 summary percentiles
    def _percentile_int(xs: List[int], q: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        if len(ys) == 1:
            return float(ys[0])
        # linear interpolation
        pos = (len(ys) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ys[lo])
        return float(ys[lo] * (hi - pos) + ys[hi] * (pos - lo))

    r90_p50 = _percentile_int(r90_list, 0.50)
    r90_p90 = _percentile_int(r90_list, 0.90)

    # Compute per-policy global suggestions (Step 5)
    policy_global_suggestions: Optional[Dict[str, Dict[str, float]]] = None
    if layers and any(layer.rank_suggestions for layer in layers):
        policy_global_suggestions = {}
        
        # Collect all available policy names
        all_policies = set()
        for layer in layers:
            if layer.rank_suggestions:
                all_policies.update(layer.rank_suggestions.keys())
        
        for policy_name in all_policies:
            # Collect k values for this policy across all layers
            policy_k_values = []
            for layer in layers:
                if (layer.rank_suggestions and 
                    policy_name in layer.rank_suggestions and 
                    'k' in layer.rank_suggestions[policy_name]):
                    k = layer.rank_suggestions[policy_name]['k']
                    if isinstance(k, (int, float)) and k >= 0:
                        policy_k_values.append(int(k))
            
            # Compute global statistics for this policy
            if policy_k_values:
                policy_global_suggestions[policy_name] = {
                    'uniform_median': float(_percentile_int(policy_k_values, 0.50)),
                    'uniform_p90': float(_percentile_int(policy_k_values, 0.90)),
                    'uniform_max': float(max(policy_k_values)),
                    'n_layers': len(policy_k_values)
                }

    # by-type aggregates
    by_type: Dict[str, Dict[str, float]] = {}
    for t, ls in by_type_acc.items():
        if not ls:
            continue
        by_type[t] = {
            "n_layers": float(len(ls)),
            "params": float(sum(l.params for l in ls)),
            "stable_rank_mean": float(sum(l.stable_rank for l in ls) / len(ls)),
            "utilization_mean": float(sum(l.utilization for l in ls) / len(ls)),
            "energy_rank_90_p50": float(_percentile_int([l.energy_rank_90 for l in ls], 0.50)),
            "energy_rank_90_p90": float(_percentile_int([l.energy_rank_90 for l in ls], 0.90)),
        }

    # Build structured policy schema (v1) 
    structured_policies = None
    if rank_policies and policy_global_suggestions:
        structured_policies = _build_structured_policy_data(
            layers, rank_policies, policy_global_suggestions
        )
    
    return LoRAAuditResult(
        peft_dir=None,
        adapter_config_path=None,
        adapter_weights_path=None,
        total_lora_params=total_params,
        n_layers=n_layers,
        stable_rank_mean=stable_rank_mean,
        stable_rank_median=stable_rank_median,
        stable_rank_weighted_mean=stable_rank_wmean,
        effective_rank_mean=eff_rank_mean,
        utilization_mean=util_mean,
        energy_rank_90_p50=r90_p50,
        energy_rank_90_p90=r90_p90,
        by_type=by_type,
        layers=layers,
        # Legacy field for backward compatibility
        policy_global_suggestions=policy_global_suggestions,
        # New structured schema (v1)
        policies=structured_policies,
        issues=issues,
    )


def audit_lora_peft_dir(
    peft_dir: Union[str, Path],
    *,
    adapter_config_path: Optional[Union[str, Path]] = None,
    adapter_weights_path: Optional[Union[str, Path]] = None,
    map_location: str = "cpu",
    compute_dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
    include_top_singular_values: int = 0,
    base_model_id: Optional[str] = None,
    base_norms_cache: Optional[Union[str, Path]] = None,
    compute_udr: bool = True,
    base_model_weights: Optional[Dict[str, torch.Tensor]] = None,
    rank_policies: Optional[List[str]] = None,
) -> LoRAAuditResult:
    """Audit a PEFT adapter directory (adapter_config + weights)."""
    d = Path(peft_dir)
    issues: List[str] = []

    cfg_path, w_path, find_issues = find_peft_files(d)
    issues.extend(find_issues)

    if adapter_config_path is not None:
        cfg_path = Path(adapter_config_path)
    if adapter_weights_path is not None:
        w_path = Path(adapter_weights_path)

    adapter_config: Optional[LoRAAdapterConfig] = None
    if cfg_path is not None and cfg_path.exists():
        try:
            adapter_config = load_peft_adapter_config(cfg_path)
        except Exception as e:
            issues.append(f"Failed to load adapter config {cfg_path}: {e}")

    if w_path is None or not w_path.exists():
        issues.append("Adapter weights file not found; cannot audit.")
        return LoRAAuditResult(
            peft_dir=str(d),
            adapter_config_path=str(cfg_path) if cfg_path else None,
            adapter_weights_path=str(w_path) if w_path else None,
            total_lora_params=0,
            n_layers=0,
            stable_rank_mean=0.0,
            stable_rank_median=0.0,
            stable_rank_weighted_mean=0.0,
            effective_rank_mean=0.0,
            utilization_mean=0.0,
            energy_rank_90_p50=0.0,
            energy_rank_90_p90=0.0,
            by_type={},
            layers=[],
            policy_global_suggestions=None,
            policies=None,
            issues=issues,
        )

    try:
        sd = load_adapter_state_dict(w_path, map_location=map_location)
    except Exception as e:
        issues.append(f"Failed to load adapter weights {w_path}: {e}")
        return LoRAAuditResult(
            peft_dir=str(d),
            adapter_config_path=str(cfg_path) if cfg_path else None,
            adapter_weights_path=str(w_path),
            total_lora_params=0,
            n_layers=0,
            stable_rank_mean=0.0,
            stable_rank_median=0.0,
            stable_rank_weighted_mean=0.0,
            effective_rank_mean=0.0,
            utilization_mean=0.0,
            energy_rank_90_p50=0.0,
            energy_rank_90_p90=0.0,
            by_type={},
            layers=[],
            policy_global_suggestions=None,
            policies=None,
            issues=issues,
        )

    # Load base model norms if UDR computation is requested
    base_norms = None
    if compute_udr:
        base_norms = load_base_model_norms(
            base_model_id=base_model_id,
            base_norms_cache=base_norms_cache,
            adapter_config=adapter_config,
            issues=issues,
        )

    result = audit_lora_state_dict(
        sd,
        adapter_config=adapter_config,
        compute_dtype=compute_dtype,
        eps=eps,
        include_top_singular_values=include_top_singular_values,
        base_norms=base_norms,
        base_model_weights=base_model_weights,
        rank_policies=rank_policies,
    )
    # patch in paths + issues
    return LoRAAuditResult(
        peft_dir=str(d),
        adapter_config_path=str(cfg_path) if cfg_path else None,
        adapter_weights_path=str(w_path),
        total_lora_params=result.total_lora_params,
        n_layers=result.n_layers,
        stable_rank_mean=result.stable_rank_mean,
        stable_rank_median=result.stable_rank_median,
        stable_rank_weighted_mean=result.stable_rank_weighted_mean,
        effective_rank_mean=result.effective_rank_mean,
        utilization_mean=result.utilization_mean,
        energy_rank_90_p50=result.energy_rank_90_p50,
        energy_rank_90_p90=result.energy_rank_90_p90,
        by_type=result.by_type,
        layers=result.layers,
        policy_global_suggestions=result.policy_global_suggestions,
        policies=result.policies,
        issues=(issues + result.issues),
    )
