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

    extras: Dict[str, Any] = field(default_factory=dict)

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
) -> Tuple[float, float, float, float, int, int, int, Optional[List[float]]]:
    """Compute (stable_rank, effective_rank, sigma_max, frob_sq, r90, r95, r99, top_singular_values)."""
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

    return stable_rank, effective_rank, sigma_max, frob_sq, r90, r95, r99, top_sv


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
) -> LoRAAuditResult:
    """Audit LoRA A/B pairs in a state_dict.

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
            stable_rank, eff_rank, sigma_max, frob_sq, r90, r95, r99, top_sv = low_rank_stable_rank(
                A2,
                B2,
                compute_dtype=compute_dtype,
                eps=eps,
                topk_singular_values=include_top_singular_values if include_top_singular_values > 0 else None,
            )
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
            issues=issues,
        )

    result = audit_lora_state_dict(
        sd,
        adapter_config=adapter_config,
        compute_dtype=compute_dtype,
        eps=eps,
        include_top_singular_values=include_top_singular_values,
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
        issues=(issues + result.issues),
    )
