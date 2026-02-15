"""
Merge plan generation from audit reports.

A MergePlan specifies exactly how to merge two adapters: which strategy to
use per layer, what coefficients, target rank, and trim fractions.  Plans
are generated from a MergeAuditReport and can be serialized to/from JSON
for inspection, modification, and reproducible execution.

Three planning strategies are provided:

- **uniform_linear**: Equal-weight linear merge on all shared layers.
- **audit_aware**: Uses per-layer verdicts to choose strategy and coefficients.
- **overlap_ties**: TIES on all layers with overlap-proportional trim density.

Usage::

    from gradience.vnext.merge import merge_audit, plan_from_audit

    report = merge_audit("./adapter_a", "./adapter_b")
    plan = plan_from_audit("audit_aware", report, "./adapter_a", "./adapter_b")
    plan.to_json(Path("merge_plan.json"))
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from gradience.vnext.merge.report import MergeAuditReport
from gradience.vnext.merge.strategies import LayerMergeConfig


# ---------------------------------------------------------------------------
# MergePlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergePlan:
    """Complete merge execution plan.

    Frozen dataclass — once generated, a plan is immutable.  To modify,
    create a new plan or deserialize from edited JSON.

    Attributes
    ----------
    plan_id : UUID identifying this plan
    strategy_name : planning strategy used to generate this plan
    adapter_a_dir : path to adapter A directory
    adapter_b_dir : path to adapter B directory
    output_rank : target rank for the merged LoRA adapter
    output_alpha : LoRA alpha for the merged adapter
    layer_configs : per-layer merge configuration
    metadata : additional info (source audit, timestamps, etc.)
    """

    plan_id: str
    strategy_name: str
    adapter_a_dir: str
    adapter_b_dir: str
    output_rank: int
    output_alpha: float
    layer_configs: Tuple[LayerMergeConfig, ...]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "schema_version": "gradience.merge_plan/v1",
            "plan_id": self.plan_id,
            "strategy_name": self.strategy_name,
            "adapter_a_dir": self.adapter_a_dir,
            "adapter_b_dir": self.adapter_b_dir,
            "output_rank": self.output_rank,
            "output_alpha": self.output_alpha,
            "layer_configs": [lc.to_dict() for lc in self.layer_configs],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MergePlan:
        """Deserialize from a dict."""
        return cls(
            plan_id=d["plan_id"],
            strategy_name=d["strategy_name"],
            adapter_a_dir=d["adapter_a_dir"],
            adapter_b_dir=d["adapter_b_dir"],
            output_rank=d["output_rank"],
            output_alpha=d["output_alpha"],
            layer_configs=tuple(
                LayerMergeConfig.from_dict(lc) for lc in d["layer_configs"]
            ),
            metadata=d.get("metadata", {}),
        )

    def to_json(self, path: Path) -> None:
        """Write plan to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> MergePlan:
        """Load plan from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    report: MergeAuditReport,
    strategy_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build metadata dict from an audit report."""
    meta: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_name,
        "source_audit": {
            "overall_verdict": report.aggregate.get("overall_verdict", "unknown"),
            "compatibility_score": report.aggregate.get("compatibility_score", 0.0),
            "n_shared_layers": report.matching.get("n_shared", 0),
        },
    }
    if extra:
        meta.update(extra)
    return meta


def _shared_layer_names(report: MergeAuditReport) -> list[str]:
    """Extract shared layer names from a report's per-layer verdicts."""
    return [lv["layer_name"] for lv in report.layer_verdicts]


# ---------------------------------------------------------------------------
# Planning strategies
# ---------------------------------------------------------------------------


def plan_uniform_linear(
    report: MergeAuditReport,
    adapter_a_dir: str,
    adapter_b_dir: str,
    coefficients: Tuple[float, float] = (0.5, 0.5),
    output_rank: int = 8,
    output_alpha: float = 16.0,
) -> MergePlan:
    """Equal-weight linear merge on all shared layers.

    Ignores verdicts — applies the same strategy and coefficients everywhere.
    This is the simplest baseline: a weighted average of the two task vectors.
    """
    layer_names = _shared_layer_names(report)

    layer_configs = tuple(
        LayerMergeConfig(
            module_prefix=name,
            strategy="linear",
            coefficients=coefficients,
            target_rank=output_rank,
            trim_fraction=0.0,
        )
        for name in layer_names
    )

    return MergePlan(
        plan_id=str(uuid.uuid4()),
        strategy_name="uniform_linear",
        adapter_a_dir=adapter_a_dir,
        adapter_b_dir=adapter_b_dir,
        output_rank=output_rank,
        output_alpha=output_alpha,
        layer_configs=layer_configs,
        metadata=_make_metadata(
            report, "uniform_linear",
            {"coefficients": list(coefficients)},
        ),
    )


def plan_audit_aware(
    report: MergeAuditReport,
    adapter_a_dir: str,
    adapter_b_dir: str,
    output_rank: int = 8,
    output_alpha: float = 16.0,
) -> MergePlan:
    """Uses per-layer verdicts to choose strategy and coefficients.

    Decision mapping:
    - SAFE → linear (0.5, 0.5)
    - REDUNDANT → ties (0.5, 0.5) to deduplicate
    - CONFLICTING → ties (0.5, 0.5) with 0.2 trim to dampen conflicts
    - IMBALANCED → linear with suggested rebalanced coefficients
    """
    layer_configs = []

    for lv_dict in report.layer_verdicts:
        verdict = lv_dict["verdict"]
        name = lv_dict["layer_name"]
        suggested_coeffs = lv_dict.get("suggested_coefficients")

        if verdict == "safe":
            layer_configs.append(LayerMergeConfig(
                module_prefix=name,
                strategy="linear",
                coefficients=(0.5, 0.5),
                target_rank=output_rank,
                trim_fraction=0.0,
            ))
        elif verdict == "redundant":
            layer_configs.append(LayerMergeConfig(
                module_prefix=name,
                strategy="ties",
                coefficients=(0.5, 0.5),
                target_rank=output_rank,
                trim_fraction=0.0,
            ))
        elif verdict == "conflicting":
            layer_configs.append(LayerMergeConfig(
                module_prefix=name,
                strategy="ties",
                coefficients=(0.5, 0.5),
                target_rank=output_rank,
                trim_fraction=0.2,
            ))
        elif verdict == "imbalanced":
            coeffs = tuple(suggested_coeffs) if suggested_coeffs else (0.5, 0.5)
            layer_configs.append(LayerMergeConfig(
                module_prefix=name,
                strategy="linear",
                coefficients=coeffs,
                target_rank=output_rank,
                trim_fraction=0.0,
            ))
        else:
            # Fallback: treat as safe
            layer_configs.append(LayerMergeConfig(
                module_prefix=name,
                strategy="linear",
                coefficients=(0.5, 0.5),
                target_rank=output_rank,
                trim_fraction=0.0,
            ))

    return MergePlan(
        plan_id=str(uuid.uuid4()),
        strategy_name="audit_aware",
        adapter_a_dir=adapter_a_dir,
        adapter_b_dir=adapter_b_dir,
        output_rank=output_rank,
        output_alpha=output_alpha,
        layer_configs=tuple(layer_configs),
        metadata=_make_metadata(report, "audit_aware"),
    )


def plan_overlap_ties(
    report: MergeAuditReport,
    adapter_a_dir: str,
    adapter_b_dir: str,
    output_rank: int = 8,
    output_alpha: float = 16.0,
    trim_fraction: float = 0.2,
) -> MergePlan:
    """TIES on all layers with overlap-proportional trim density.

    Trim fraction scales with the layer's mean_overlap: higher overlap
    gets higher trim (more aggressive deduplication).  The base
    ``trim_fraction`` is multiplied by the layer's mean_overlap.

    For a layer with overlap=0.1 and base trim=0.2: effective trim = 0.02
    For a layer with overlap=0.9 and base trim=0.2: effective trim = 0.18
    """
    layer_configs = []

    for lv_dict in report.layer_verdicts:
        name = lv_dict["layer_name"]
        metrics = lv_dict.get("metrics", {})
        overlap = metrics.get("mean_overlap", 0.0)

        # Scale trim by overlap: more overlap → more trimming
        effective_trim = trim_fraction * overlap

        layer_configs.append(LayerMergeConfig(
            module_prefix=name,
            strategy="ties",
            coefficients=(0.5, 0.5),
            target_rank=output_rank,
            trim_fraction=round(effective_trim, 4),
        ))

    return MergePlan(
        plan_id=str(uuid.uuid4()),
        strategy_name="overlap_ties",
        adapter_a_dir=adapter_a_dir,
        adapter_b_dir=adapter_b_dir,
        output_rank=output_rank,
        output_alpha=output_alpha,
        layer_configs=tuple(layer_configs),
        metadata=_make_metadata(
            report, "overlap_ties",
            {"base_trim_fraction": trim_fraction},
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PLAN_STRATEGIES: Dict[str, Callable[..., MergePlan]] = {
    "uniform_linear": plan_uniform_linear,
    "audit_aware": plan_audit_aware,
    "overlap_ties": plan_overlap_ties,
}


def plan_from_audit(
    strategy_name: str,
    report: MergeAuditReport,
    adapter_a_dir: str,
    adapter_b_dir: str,
    **kwargs: Any,
) -> MergePlan:
    """Dispatch to the named planning strategy.

    Parameters
    ----------
    strategy_name : one of ``"uniform_linear"``, ``"audit_aware"``, ``"overlap_ties"``
    report : MergeAuditReport from a prior merge_audit() call
    adapter_a_dir : path to adapter A
    adapter_b_dir : path to adapter B
    **kwargs : forwarded to the strategy function (e.g., ``output_rank``, ``trim_fraction``)

    Returns
    -------
    MergePlan

    Raises
    ------
    ValueError
        If ``strategy_name`` is not recognized.
    """
    fn = PLAN_STRATEGIES.get(strategy_name)
    if fn is None:
        raise ValueError(
            f"Unknown plan strategy '{strategy_name}'. "
            f"Available: {sorted(PLAN_STRATEGIES.keys())}"
        )
    return fn(report, adapter_a_dir, adapter_b_dir, **kwargs)
