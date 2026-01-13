"""
gradience.vnext.rank_suggestion

Pure functions that turn LoRA audit summaries into conservative global rank
compression suggestions.

Design goals:
- No side effects (no file I/O, no logging)
- Conservative defaults (bucket to allowed ranks, never suggest increasing)
- Works even if audit JSON contains only summary stats (no per-layer rows yet)

This module is intentionally "small and boring" so it can be depended on by
CLI, monitor, notebooks, etc. without pulling in heavy deps.

## Public API

The following are considered public API with stability guarantees:
- suggest_global_ranks_from_audit()
- suggest_per_layer_ranks()
- GlobalRankSuggestion class
- PerLayerRankSuggestion class
- PerLayerRankSuggestionReport class
- DEFAULT_ALLOWED_RANKS constant

All other functions (prefixed with _) are internal and may change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Keep this small and PEFT-ish. You can extend later (128, 256) if you want.
DEFAULT_ALLOWED_RANKS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def _round_up_to_allowed_rank(x: float, allowed: Sequence[int] = DEFAULT_ALLOWED_RANKS) -> int:
    """
    Round x up to the nearest allowed rank bucket.

    Examples:
      2.0 -> 2
      3.0 -> 4
      6.0 -> 8
      0.0 -> 0
    """
    try:
        xf = float(x)
    except Exception:
        return 0

    if xf <= 0:
        return 0

    for r in allowed:
        if r >= xf:
            return int(r)
    return int(allowed[-1])


def _infer_current_r_from_means(stable_rank_mean: Any, utilization_mean: Any) -> Optional[int]:
    """
    Infer current LoRA rank r using:
      utilization_mean ~= stable_rank_mean / r  => r ~= stable_rank_mean / utilization_mean

    This is a *fallback* for cases where we don't have explicit r in the JSON.
    """
    try:
        sr = float(stable_rank_mean)
        util = float(utilization_mean)
    except Exception:
        return None

    if util <= 0:
        return None

    r_hat = sr / util
    if r_hat <= 0:
        return None

    # Round to nearest int; guard against pathological values.
    r_int = int(round(r_hat))
    return max(1, r_int)


def _get_first_present(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


@dataclass(frozen=True)
class GlobalRankSuggestion:
    """
    Conservative global rank compression suggestion derived from audit summary stats.
    """

    current_r: int

    suggested_r_median: int
    suggested_r_p90: int

    total_lora_params: int
    params_at_r_median: int
    params_at_r_p90: int

    reduction_ratio_median: float
    reduction_ratio_p90: float

    # Evidence we used to make the suggestion (useful for monitor/verbose).
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_r": self.current_r,
            "suggested_r_median": self.suggested_r_median,
            "suggested_r_p90": self.suggested_r_p90,
            "total_lora_params": self.total_lora_params,
            "params_at_r_median": self.params_at_r_median,
            "params_at_r_p90": self.params_at_r_p90,
            "reduction_ratio_median": self.reduction_ratio_median,
            "reduction_ratio_p90": self.reduction_ratio_p90,
            "evidence": dict(self.evidence),
        }


def suggest_global_ranks_from_audit(
    audit: Dict[str, Any],
    *,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
) -> GlobalRankSuggestion:
    """
    Pure function: audit summary dict -> conservative global rank suggestions.

    Inputs expected (from gradience audit --json):
      - total_lora_params (int)
      - stable_rank_mean (float)
      - utilization_mean (float)
      - energy_rank_90_p50 (float)
      - energy_rank_90_p90 (float)
      - optionally suggested_r_global_median / suggested_r_global_90 (already bucketed)

    Behavior:
      - Prefer explicit suggested ranks if audit already provides them.
      - Otherwise derive from energy_rank_90_p50/p90 and bucket to allowed ranks.
      - Infer current_r (prefer explicit; else infer from stable_rank_mean/utilization_mean).
      - Never suggest increasing rank above current_r.
      - Estimate params at suggested ranks using linear scaling:
            params_new ~= total_lora_params * (r_new / current_r)
    """
    if not isinstance(audit, dict):
        raise TypeError("audit must be a dict")

    # ---- total params (required for savings math; default to 0 if missing) ----
    total_params_raw = _get_first_present(audit, ("total_lora_params", "lora_params", "total_params"))
    try:
        total_params = int(total_params_raw) if total_params_raw is not None else 0
    except Exception:
        total_params = 0

    # ---- current_r (prefer explicit, else infer) ----
    current_r_raw = _get_first_present(audit, ("current_r", "r", "lora_r"))
    current_r: Optional[int] = None
    if isinstance(current_r_raw, int) and current_r_raw > 0:
        current_r = current_r_raw

    if current_r is None:
        current_r = _infer_current_r_from_means(audit.get("stable_rank_mean"), audit.get("utilization_mean"))

    if current_r is None:
        raise ValueError(
            "Could not infer current_r. Provide audit['r']/'current_r' or stable_rank_mean+utilization_mean."
        )

    # ---- suggested ranks (prefer explicit; else derive) ----
    s_med_raw = _get_first_present(audit, ("suggested_r_global_median", "suggested_r_median"))
    s_p90_raw = _get_first_present(audit, ("suggested_r_global_90", "suggested_r_p90"))

    if s_med_raw is None:
        s_med_raw = audit.get("energy_rank_90_p50", 0.0)
    if s_p90_raw is None:
        s_p90_raw = audit.get("energy_rank_90_p90", 0.0)

    s_med = _round_up_to_allowed_rank(float(s_med_raw), allowed=allowed_ranks)
    s_p90 = _round_up_to_allowed_rank(float(s_p90_raw), allowed=allowed_ranks)

    # Never suggest increasing
    s_med = min(s_med, current_r)
    s_p90 = min(s_p90, current_r)

    # ---- savings math (linear in r, for uniform-r adapters) ----
    def est_params(r_new: int) -> int:
        if total_params <= 0 or current_r <= 0:
            return 0
        return int(round(total_params * (r_new / current_r)))

    params_med = est_params(s_med)
    params_p90 = est_params(s_p90)

    red_med = 0.0 if total_params <= 0 else max(0.0, 1.0 - (params_med / total_params))
    red_p90 = 0.0 if total_params <= 0 else max(0.0, 1.0 - (params_p90 / total_params))

    evidence = {
        "stable_rank_mean": audit.get("stable_rank_mean"),
        "utilization_mean": audit.get("utilization_mean"),
        "energy_rank_90_p50": audit.get("energy_rank_90_p50"),
        "energy_rank_90_p90": audit.get("energy_rank_90_p90"),
        "suggested_r_global_median_input": audit.get("suggested_r_global_median"),
        "suggested_r_global_90_input": audit.get("suggested_r_global_90"),
        "allowed_ranks": list(allowed_ranks),
    }

    return GlobalRankSuggestion(
        current_r=current_r,
        suggested_r_median=s_med,
        suggested_r_p90=s_p90,
        total_lora_params=total_params,
        params_at_r_median=params_med,
        params_at_r_p90=params_p90,
        reduction_ratio_median=red_med,
        reduction_ratio_p90=red_p90,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Per-layer rank suggestions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerLayerRankSuggestion:
    name: str
    current_r: int
    energy_rank_90: float
    suggested_r: int
    reduction_ratio: float
    stable_rank: Optional[float] = None
    utilization: Optional[float] = None
    module_type: Optional[str] = None  # optional convenience


@dataclass(frozen=True)
class PerLayerRankSuggestionReport:
    layers: Tuple[PerLayerRankSuggestion, ...]
    default_r: int
    rank_pattern: Dict[str, int]   # only entries where suggested != default_r
    by_module_type_p90: Dict[str, int]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_r": self.default_r,
            "rank_pattern": dict(self.rank_pattern),
            "by_module_type_p90": dict(self.by_module_type_p90),
            "notes": self.notes,
            "layers": [
                {
                    "name": s.name,
                    "module_type": s.module_type,
                    "current_r": s.current_r,
                    "energy_rank_90": s.energy_rank_90,
                    "stable_rank": s.stable_rank,
                    "utilization": s.utilization,
                    "suggested_r": s.suggested_r,
                    "reduction_ratio": s.reduction_ratio,
                }
                for s in self.layers
            ],
        }


def _get_layer_name(row: Dict[str, Any]) -> Optional[str]:
    for k in ("name", "module", "path", "layer", "layer_name"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _get_float(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def _get_int(row: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            iv = int(v)
            return iv if iv > 0 else None
        except Exception:
            pass
    return None


def suggest_per_layer_ranks(
    audit: Dict[str, Any],
    *,
    margin: float = 1.0,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
) -> PerLayerRankSuggestionReport:
    """
    Pure function: consume audit dict (must include `layers`) and suggest rank per layer.

    - Uses per-layer energy_rank_90 (k@90%) as the base.
    - Applies a multiplicative margin (default 1.0 = no extra headroom).
    - Buckets to allowed ranks.
    - Never suggests increasing above current_r.

    Returns:
      PerLayerRankSuggestionReport with:
        - per-layer suggested ranks
        - a suggested default_r (mode of suggested ranks)
        - rank_pattern overrides for layers != default_r
        - by_module_type_p90 aggregation
    """
    if not isinstance(audit, dict):
        raise TypeError("audit must be a dict")

    # Support both old and new layer data structures
    rows = audit.get("layers", None)
    if rows is None:
        layer_data = audit.get("layer_data", {})
        if isinstance(layer_data, dict):
            rows = layer_data.get("layer_rows", None)
    
    if not isinstance(rows, list):
        raise ValueError("audit must include layer data in `layers` or `layer_data.layer_rows` (run audit with --layers).")

    suggestions: List[PerLayerRankSuggestion] = []
    skipped = 0

    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue

        name = _get_layer_name(row)
        if not name:
            skipped += 1
            continue

        current_r = _get_int(row, ("r", "rank", "current_r"))  # per-row current rank
        if current_r is None:
            skipped += 1
            continue

        k90 = _get_float(row, ("energy_rank_90", "k_90", "k90", "energy_rank_k90"))
        if k90 is None:
            skipped += 1
            continue

        stable_rank = _get_float(row, ("stable_rank", "stable_rank_est"))
        util = _get_float(row, ("utilization", "util"))

        module_type = row.get("module_type") or row.get("type")
        if not isinstance(module_type, str):
            # fallback: last token in module path
            module_type = name.split(".")[-1] if "." in name else None

        target = k90 * float(margin)
        suggested = _round_up_to_allowed_rank(target, allowed=allowed_ranks)
        suggested = min(suggested, current_r)

        reduction_ratio = 1.0 - (suggested / current_r) if current_r > 0 else 0.0

        suggestions.append(
            PerLayerRankSuggestion(
                name=name,
                module_type=module_type,
                current_r=current_r,
                energy_rank_90=float(k90),
                stable_rank=stable_rank,
                utilization=util,
                suggested_r=int(suggested),
                reduction_ratio=float(reduction_ratio),
            )
        )

    if not suggestions:
        return PerLayerRankSuggestionReport(
            layers=tuple(),
            default_r=0,
            rank_pattern={},
            by_module_type_p90={},
            notes="No valid layer rows found in audit['layers'].",
        )

    # default_r = mode of suggested ranks (keeps rank_pattern small)
    counts: Dict[int, int] = {}
    for s in suggestions:
        counts[s.suggested_r] = counts.get(s.suggested_r, 0) + 1
    default_r = max(counts.items(), key=lambda kv: kv[1])[0]

    rank_pattern = {s.name: s.suggested_r for s in suggestions if s.suggested_r != default_r}

    # by_module_type_p90 (conservative)
    by_type: Dict[str, List[int]] = {}
    for s in suggestions:
        if not s.module_type:
            continue
        by_type.setdefault(s.module_type, []).append(s.suggested_r)

    by_module_type_p90: Dict[str, int] = {}
    for t, rs in by_type.items():
        rs_sorted = sorted(rs)
        idx = int(0.9 * (len(rs_sorted) - 1)) if len(rs_sorted) > 1 else 0
        by_module_type_p90[t] = rs_sorted[idx]

    notes = f"ok (rows={len(rows)}, used={len(suggestions)}, skipped={skipped}, margin={margin})"

    return PerLayerRankSuggestionReport(
        layers=tuple(suggestions),
        default_r=int(default_r),
        rank_pattern=rank_pattern,
        by_module_type_p90=by_module_type_p90,
        notes=notes,
    )


# Public API exports (stability guaranteed)
__all__ = [
    "DEFAULT_ALLOWED_RANKS",
    "GlobalRankSuggestion", 
    "PerLayerRankSuggestion",
    "PerLayerRankSuggestionReport",
    "suggest_global_ranks_from_audit",
    "suggest_per_layer_ranks",
]