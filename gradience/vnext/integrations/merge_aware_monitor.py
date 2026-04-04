"""
Diagnostic-only merge-aware compatibility monitoring during training.

This module reuses existing merge-audit substrate (overlap/conflict/imbalance)
to compute bounded compatibility snapshots between:
1) the adapter currently under training, and
2) one fixed reference adapter.

It does not modify optimization behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gradience.vnext.audit import iter_lora_pairs, orient_lora_factors
from gradience.vnext.merge.io import get_module_type, load_adapter
from gradience.vnext.merge.spectral_compat import compute_subspace_metrics
from gradience.vnext.merge.verdicts import CompatibilityVerdict, VerdictThresholds, assess_layer, assess_overall


@dataclass(frozen=True)
class MergeAwareSnapshot:
    """Single training-step compatibility snapshot against one reference adapter."""

    reference_adapter: str
    status: str
    n_current_layers: int
    n_reference_layers: int
    n_shared_layers: int
    mean_overlap: float
    mean_agreement: float
    conflict_fraction: float
    imbalance_fraction: float
    compatibility_score: float
    overall_verdict: str
    overlap_trend_target: str = "up"
    conflict_trend_target: str = "down"
    imbalance_trend_target: str = "down"
    score_trend_target: str = "up"

    def to_metrics_payload(self) -> dict[str, Any]:
        return {
            "reference_adapter": self.reference_adapter,
            "status": self.status,
            "n_current_layers": self.n_current_layers,
            "n_reference_layers": self.n_reference_layers,
            "n_shared_layers": self.n_shared_layers,
            "mean_overlap": self.mean_overlap,
            "mean_agreement": self.mean_agreement,
            "conflict_fraction": self.conflict_fraction,
            "imbalance_fraction": self.imbalance_fraction,
            "compatibility_score": self.compatibility_score,
            "overall_verdict": self.overall_verdict,
            "overlap_trend_target": self.overlap_trend_target,
            "conflict_trend_target": self.conflict_trend_target,
            "imbalance_trend_target": self.imbalance_trend_target,
            "score_trend_target": self.score_trend_target,
        }


def _collect_lora_factors(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
    """Extract `{module_prefix: (A, B, r)}` from an adapter-like state dict."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for prefix, a_key, b_key in iter_lora_pairs(state_dict):
        A_raw = state_dict[a_key]
        B_raw = state_dict[b_key]
        A, B, r = orient_lora_factors(A_raw, B_raw)
        out[prefix] = (A.detach().cpu(), B.detach().cpu(), int(r))
    return out


def _extract_lora_state_dict_from_model(model: Any) -> dict[str, torch.Tensor]:
    """
    Build a small state dict containing only LoRA A/B weights from a model.

    Works for PEFT-like keys:
      ...lora_A.weight
      ...lora_B.weight
      ...lora_A.default.weight
      ...lora_B.default.weight
    """
    if model is None or not hasattr(model, "state_dict"):
        return {}
    try:
        full_state = model.state_dict()
    except (RuntimeError, AttributeError, TypeError):
        return {}

    out: dict[str, torch.Tensor] = {}
    for k, v in full_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if (".lora_A." in k or ".lora_B." in k or k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")) and (
            k.endswith(".weight")
        ):
            out[k] = v.detach().cpu()
    return out


def _infer_model_alpha_and_rank(model: Any) -> tuple[float | None, int | None]:
    """Best-effort LoRA alpha/r extraction from a PEFT model."""
    if model is None:
        return None, None
    try:
        peft_cfg = getattr(model, "peft_config", None)
    except (AttributeError, TypeError):
        peft_cfg = None

    if isinstance(peft_cfg, dict) and peft_cfg:
        cfg = peft_cfg.get("default") or next(iter(peft_cfg.values()))
    else:
        cfg = peft_cfg

    if cfg is None:
        return None, None

    alpha = getattr(cfg, "lora_alpha", None)
    rank = getattr(cfg, "r", None)
    alpha_f = float(alpha) if alpha is not None else None
    rank_i = int(rank) if rank is not None else None
    return alpha_f, rank_i


def _trend_label(values: list[float], *, flat_tol: float = 1e-6) -> str:
    """
    Return one of: increasing, decreasing, mixed_unstable, flat, inconclusive.
    """
    if len(values) < 3:
        return "inconclusive"

    start = float(values[0])
    end = float(values[-1])
    span = max(values) - min(values)
    delta = end - start

    if span <= flat_tol:
        return "flat"

    # Ignore tiny fluctuations relative to span.
    sig = max(0.05 * span, flat_tol)
    signs: list[int] = []
    for a, b in zip(values[:-1], values[1:]):
        d = b - a
        if abs(d) < sig:
            continue
        signs.append(1 if d > 0 else -1)

    if not signs:
        return "flat"
    if any(s != signs[0] for s in signs):
        return "mixed_unstable"

    # If total movement is tiny compared to span, treat as mixed/flat.
    if abs(delta) < 0.1 * span:
        return "mixed_unstable"
    return "increasing" if delta > 0 else "decreasing"


def summarize_merge_aware_trends(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-metric and run-level compatibility trend labels."""
    if not snapshots:
        return {
            "n_snapshots": 0,
            "run_level_trend": "inconclusive",
            "per_metric_trend": {},
            "reason": "no_snapshots",
        }

    # Keep snapshots step-ordered.
    ordered = sorted(snapshots, key=lambda s: int(s.get("step", 0)))

    def _series(key: str) -> list[float]:
        vals = []
        for s in ordered:
            v = s.get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        return vals

    overlap = _series("mean_overlap")
    conflict = _series("conflict_fraction")
    imbalance = _series("imbalance_fraction")
    score = _series("compatibility_score")

    trends = {
        "mean_overlap": _trend_label(overlap),
        "conflict_fraction": _trend_label(conflict),
        "imbalance_fraction": _trend_label(imbalance),
        "compatibility_score": _trend_label(score),
    }

    toward = 0
    away = 0
    mixed = 0
    for metric, label in trends.items():
        if label in {"flat", "inconclusive"}:
            continue
        if label == "mixed_unstable":
            mixed += 1
            continue

        # Desired directions:
        # overlap up, conflict down, imbalance down, score up.
        desired_up = metric in {"mean_overlap", "compatibility_score"}
        is_up = label == "increasing"
        if (desired_up and is_up) or ((not desired_up) and (not is_up)):
            toward += 1
        else:
            away += 1

    if toward >= 2 and away == 0 and mixed <= 1:
        run_trend = "toward_compatibility"
    elif away >= 2 and toward == 0 and mixed <= 1:
        run_trend = "away_from_compatibility"
    elif toward >= 1 and away >= 1 or mixed >= 2:
        run_trend = "mixed"
    else:
        run_trend = "inconclusive"

    return {
        "n_snapshots": len(ordered),
        "run_level_trend": run_trend,
        "per_metric_trend": trends,
        "start_step": int(ordered[0].get("step", 0)),
        "end_step": int(ordered[-1].get("step", 0)),
    }


class MergeAwareTrainingMonitor:
    """
    Diagnostic monitor for compatibility drift against one fixed reference.

    This monitor is measurement-only: it never mutates model/optimizer state.
    """

    def __init__(
        self,
        merge_target: str | Path,
        *,
        thresholds: VerdictThresholds | None = None,
        energy_threshold: float = 0.90,
        compute_dtype: str = "float32",
    ):
        self.merge_target = Path(merge_target)
        self.thresholds = thresholds or VerdictThresholds()
        self.energy_threshold = float(energy_threshold)
        self.compute_dtype = torch.float32 if str(compute_dtype).lower() in {"float32", "fp32"} else torch.float64

        # Load and cache fixed reference adapter + factors once.
        self.reference = load_adapter(self.merge_target)
        self._reference_factors = _collect_lora_factors(self.reference.state_dict)

    def snapshot_from_state_dict(
        self,
        current_state_dict: dict[str, torch.Tensor],
        *,
        current_alpha: float | None = None,
        current_rank: int | None = None,
    ) -> MergeAwareSnapshot:
        current_factors = _collect_lora_factors(current_state_dict)
        shared = sorted(set(current_factors).intersection(self._reference_factors))

        n_current = len(current_factors)
        n_ref = len(self._reference_factors)
        n_shared = len(shared)
        if n_shared == 0:
            return MergeAwareSnapshot(
                reference_adapter=str(self.merge_target),
                status="no_shared_layers",
                n_current_layers=n_current,
                n_reference_layers=n_ref,
                n_shared_layers=0,
                mean_overlap=0.0,
                mean_agreement=0.0,
                conflict_fraction=0.0,
                imbalance_fraction=0.0,
                compatibility_score=0.0,
                overall_verdict="inconclusive",
            )

        # Conservative fallback when current alpha/r are unavailable.
        layer_verdicts = []
        for prefix in shared:
            A_cur, B_cur, r_cur = current_factors[prefix]
            A_ref, B_ref, r_ref = self._reference_factors[prefix]
            alpha_cur = float(current_alpha) if current_alpha is not None else float(current_rank or r_cur)
            alpha_ref = float(self.reference.alpha)
            module_type = get_module_type(prefix)
            metrics = compute_subspace_metrics(
                A_cur,
                B_cur,
                alpha_cur,
                r_cur,
                A_ref,
                B_ref,
                alpha_ref,
                r_ref,
                energy_threshold=self.energy_threshold,
                compute_dtype=self.compute_dtype,
            )
            lv = assess_layer(prefix, module_type, metrics, self.thresholds)
            layer_verdicts.append(lv)

        overall, score, _recommendations = assess_overall(layer_verdicts)
        overlaps = [lv.metrics.mean_overlap for lv in layer_verdicts]
        agreements = [lv.metrics.directional_agreement for lv in layer_verdicts]
        n_conflict = sum(1 for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.CONFLICTING)
        n_imb = sum(1 for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.IMBALANCED)

        return MergeAwareSnapshot(
            reference_adapter=str(self.merge_target),
            status="ok",
            n_current_layers=n_current,
            n_reference_layers=n_ref,
            n_shared_layers=n_shared,
            mean_overlap=float(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
            mean_agreement=float(sum(agreements) / len(agreements)) if agreements else 0.0,
            conflict_fraction=float(n_conflict / n_shared),
            imbalance_fraction=float(n_imb / n_shared),
            compatibility_score=float(score),
            overall_verdict=str(overall.value),
        )

    def snapshot_from_model(self, model: Any) -> MergeAwareSnapshot:
        state = _extract_lora_state_dict_from_model(model)
        alpha, rank = _infer_model_alpha_and_rank(model)
        return self.snapshot_from_state_dict(state, current_alpha=alpha, current_rank=rank)
