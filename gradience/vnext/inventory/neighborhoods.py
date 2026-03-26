"""Rule-based merge neighborhood suggestions from existing preflight artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

from gradience.vnext.inventory.merge_neighborhood_report import (
    BoundaryWarning,
    ExcludedAdapter,
    MergeNeighborhoodReport,
    NeighborhoodGroup,
)

COMPATIBILITY_BUCKETS = frozenset(
    {
        "high_compatibility",
        "moderate_compatibility",
        "conditional_compatibility",
        "incompatible",
    }
)

_STRICT_BLOCK_STATUSES = frozenset({"flagged_weak", "unknown_no_behavioral_eval", None})
_EXCLUDE_BY_DEFAULT = frozenset({"flagged_weak"})
_SAFE_STRATEGY_DEFAULT = "linear"

_BUCKET_SEVERITY = {
    "high_compatibility": 0,
    "moderate_compatibility": 1,
    "conditional_compatibility": 2,
    "incompatible": 3,
}

_REASON_PRIORITY = {
    "flagged_weak": 4,
    "missing_qa_artifact": 3,
    "unknown_no_behavioral_eval": 2,
    "unknown_adapter": 1,
}


@dataclass(frozen=True)
class CompatibilityEdge:
    adapter_a: str
    adapter_b: str
    compatibility_bucket: str
    compatibility_score: float
    recommended_strategy: str
    dominant_issue: str
    strict_block_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_a": self.adapter_a,
            "adapter_b": self.adapter_b,
            "compatibility_bucket": self.compatibility_bucket,
            "compatibility_score": float(self.compatibility_score),
            "recommended_strategy": self.recommended_strategy,
            "dominant_issue": self.dominant_issue,
            "strict_block_candidate": bool(self.strict_block_candidate),
        }


def _adapter_id_from_qa(artifact: Any) -> str:
    return str(getattr(artifact, "adapter_path", "") or getattr(artifact, "adapter_name", "unknown_adapter"))


def _select_reason(existing: str | None, new_reason: str) -> str:
    if existing is None:
        return new_reason
    existing_priority = _REASON_PRIORITY.get(existing, 0)
    new_priority = _REASON_PRIORITY.get(new_reason, 0)
    return new_reason if new_priority > existing_priority else existing


def _compatibility_bucket_for_report(
    report: Any,
    *,
    min_compatibility: float,
) -> tuple[str, bool]:
    pair_risk = str(getattr(report, "pair_risk", "high"))
    strategy = str(getattr(report, "recommended_strategy", "audit_aware"))
    score = float(getattr(report, "compatibility_score", 0.0))

    a_status = getattr(getattr(report, "adapter_a", None), "eligibility_status", None)
    b_status = getattr(getattr(report, "adapter_b", None), "eligibility_status", None)
    strict_block_candidate = a_status in _STRICT_BLOCK_STATUSES or b_status in _STRICT_BLOCK_STATUSES

    if strict_block_candidate:
        return "incompatible", True
    if score < min_compatibility:
        return "incompatible", False
    if pair_risk == "low" and strategy == "linear":
        return "high_compatibility", False
    if pair_risk == "medium" and strategy == "norm_equalized":
        return "moderate_compatibility", False
    if pair_risk == "low":
        return "moderate_compatibility", False
    if pair_risk == "high" and strategy == "audit_aware":
        return "conditional_compatibility", False
    if pair_risk == "medium":
        return "conditional_compatibility", False
    return "incompatible", False


def build_compatibility_matrix(
    merge_reports: list[Any],
    *,
    min_compatibility: float = 0.0,
) -> list[CompatibilityEdge]:
    """Build a conservative pairwise compatibility edge list."""
    edges_by_pair: dict[tuple[str, str], CompatibilityEdge] = {}

    for report in merge_reports:
        adapter_a = str(getattr(getattr(report, "adapter_a", None), "path", ""))
        adapter_b = str(getattr(getattr(report, "adapter_b", None), "path", ""))
        if not adapter_a or not adapter_b or adapter_a == adapter_b:
            continue

        bucket, strict_block = _compatibility_bucket_for_report(
            report,
            min_compatibility=min_compatibility,
        )
        if bucket not in COMPATIBILITY_BUCKETS:
            bucket = "incompatible"

        edge = CompatibilityEdge(
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            compatibility_bucket=bucket,
            compatibility_score=float(getattr(report, "compatibility_score", 0.0)),
            recommended_strategy=str(getattr(report, "recommended_strategy", "unknown")),
            dominant_issue=str(getattr(report, "dominant_issue", "unknown")),
            strict_block_candidate=strict_block,
        )

        key = tuple(sorted((adapter_a, adapter_b)))
        existing = edges_by_pair.get(key)
        if existing is None:
            edges_by_pair[key] = edge
            continue

        existing_severity = _BUCKET_SEVERITY.get(existing.compatibility_bucket, 99)
        new_severity = _BUCKET_SEVERITY.get(edge.compatibility_bucket, 99)
        if new_severity > existing_severity or (
            new_severity == existing_severity and edge.compatibility_score < existing.compatibility_score
        ):
            edges_by_pair[key] = edge

    return sorted(
        edges_by_pair.values(),
        key=lambda e: (min(e.adapter_a, e.adapter_b), max(e.adapter_a, e.adapter_b)),
    )


def _collect_exclusions(
    qa_artifacts: list[Any],
    merge_reports: list[Any],
    *,
    strict_qa: bool,
    exclude_unknown: bool,
) -> dict[str, str]:
    excluded: dict[str, str] = {}

    for qa in qa_artifacts:
        adapter_id = _adapter_id_from_qa(qa)
        status = getattr(getattr(qa, "status", None), "value", None)
        if status in _EXCLUDE_BY_DEFAULT:
            excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "flagged_weak")
        elif (strict_qa or exclude_unknown) and status == "unknown_no_behavioral_eval":
            excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "unknown_no_behavioral_eval")

    for report in merge_reports:
        for adapter in (getattr(report, "adapter_a", None), getattr(report, "adapter_b", None)):
            if adapter is None:
                continue
            adapter_id = str(getattr(adapter, "path", ""))
            if not adapter_id:
                continue
            status = getattr(adapter, "eligibility_status", None)
            if status == "flagged_weak":
                excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "flagged_weak")
            if strict_qa and status is None:
                excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "missing_qa_artifact")
            if strict_qa and status == "unknown_no_behavioral_eval":
                excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "unknown_no_behavioral_eval")
            if exclude_unknown and status is None:
                excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "unknown_adapter")
            if exclude_unknown and status == "unknown_no_behavioral_eval":
                excluded[adapter_id] = _select_reason(excluded.get(adapter_id), "unknown_no_behavioral_eval")

    return excluded


def _build_high_compatibility_groups(candidates: list[str], edges: list[CompatibilityEdge]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {adapter: set() for adapter in candidates}
    for edge in edges:
        if edge.compatibility_bucket != "high_compatibility":
            continue
        if edge.adapter_a in adjacency and edge.adapter_b in adjacency:
            adjacency[edge.adapter_a].add(edge.adapter_b)
            adjacency[edge.adapter_b].add(edge.adapter_a)

    groups: list[set[str]] = []
    seen: set[str] = set()
    for adapter in sorted(candidates):
        if adapter in seen:
            continue
        stack = [adapter]
        component: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            stack.extend(sorted(adjacency[cur] - component))
        seen.update(component)
        groups.append(component)
    return groups


def _can_merge_groups(
    group_a: set[str],
    group_b: set[str],
    edge_lookup: dict[tuple[str, str], CompatibilityEdge],
) -> bool:
    for left, right in product(group_a, group_b):
        key = tuple(sorted((left, right)))
        edge = edge_lookup.get(key)
        if edge is None:
            continue
        if edge.compatibility_bucket in {"incompatible", "conditional_compatibility"}:
            return False
    return True


def _mode_or_default(values: list[str], default: str) -> str:
    if not values:
        return default
    counter = Counter(values)
    top = max(counter.values())
    tied = sorted([k for k, v in counter.items() if v == top])
    return tied[0]


def _characterize_group(group: set[str], intra_edges: list[CompatibilityEdge]) -> str:
    buckets = {e.compatibility_bucket for e in intra_edges}
    if "conditional_compatibility" in buckets:
        return "audit-aware neighborhood"
    if "moderate_compatibility" in buckets:
        return "caution neighborhood"
    if len(group) == 1:
        return "caution neighborhood"
    return "likely-safe neighborhood"


def _boundary_reason(cross_edges: list[CompatibilityEdge]) -> str | None:
    if any(e.compatibility_bucket == "incompatible" for e in cross_edges):
        return "high cross-group merge risk"
    if any(e.compatibility_bucket == "conditional_compatibility" for e in cross_edges):
        return "conditional cross-group merge risk (audit-aware checks recommended)"
    return None


def suggest_merge_neighborhoods(
    qa_artifacts: list[Any],
    merge_reports: list[Any],
    *,
    strict_qa: bool = False,
    min_compatibility: float = 0.0,
    exclude_unknown: bool = False,
) -> MergeNeighborhoodReport:
    """Suggest conservative merge neighborhoods from existing artifacts."""
    edges = build_compatibility_matrix(
        merge_reports,
        min_compatibility=min_compatibility,
    )
    edge_lookup: dict[tuple[str, str], CompatibilityEdge] = {
        tuple(sorted((e.adapter_a, e.adapter_b))): e for e in edges
    }

    adapter_ids: set[str] = set()
    for qa in qa_artifacts:
        adapter_ids.add(_adapter_id_from_qa(qa))
    for report in merge_reports:
        for adapter in (getattr(report, "adapter_a", None), getattr(report, "adapter_b", None)):
            if adapter is not None and getattr(adapter, "path", None):
                adapter_ids.add(str(adapter.path))

    excluded_map = _collect_exclusions(
        qa_artifacts,
        merge_reports,
        strict_qa=strict_qa,
        exclude_unknown=exclude_unknown,
    )
    candidates = sorted(adapter_ids - set(excluded_map.keys()))

    # 1) High-compatibility components.
    groups = _build_high_compatibility_groups(candidates, edges)

    # 2) Attach moderate neighbors only when no contradictory high-risk boundary is introduced.
    moderate_edges = [e for e in edges if e.compatibility_bucket == "moderate_compatibility"]
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for edge in moderate_edges:
            idx_a = next((i for i, g in enumerate(groups) if edge.adapter_a in g), None)
            idx_b = next((i for i, g in enumerate(groups) if edge.adapter_b in g), None)
            if idx_a is None or idx_b is None or idx_a == idx_b:
                continue
            ga, gb = groups[idx_a], groups[idx_b]
            if not _can_merge_groups(ga, gb, edge_lookup):
                continue
            merged = ga | gb
            groups = [g for i, g in enumerate(groups) if i not in {idx_a, idx_b}]
            groups.append(merged)
            changed = True
            break

    # Deterministic ordering
    sorted_groups = sorted((sorted(g) for g in groups if g), key=lambda g: g[0])
    typed_groups: list[NeighborhoodGroup] = []
    for i, members in enumerate(sorted_groups, start=1):
        gid = f"cluster_{i:02d}"

        intra_edges = []
        for left, right in combinations(members, 2):
            edge = edge_lookup.get(tuple(sorted((left, right))))
            if edge is not None:
                intra_edges.append(edge)

        typed_groups.append(
            NeighborhoodGroup(
                group_id=gid,
                members=tuple(members),
                characterization=_characterize_group(set(members), intra_edges),
                common_strategy=_mode_or_default([e.recommended_strategy for e in intra_edges], _SAFE_STRATEGY_DEFAULT),
                dominant_issue=_mode_or_default(
                    [e.dominant_issue for e in intra_edges if e.dominant_issue not in {"none", "unknown", ""}],
                    "none",
                ),
            )
        )

    excluded = tuple(
        ExcludedAdapter(adapter=adapter, reason=reason)
        for adapter, reason in sorted(excluded_map.items(), key=lambda x: x[0])
    )

    boundary_warnings: list[BoundaryWarning] = []
    for g1, g2 in combinations(typed_groups, 2):
        cross_edges: list[CompatibilityEdge] = []
        for left, right in product(g1.members, g2.members):
            edge = edge_lookup.get(tuple(sorted((left, right))))
            if edge is not None:
                cross_edges.append(edge)
        reason = _boundary_reason(cross_edges)
        if reason is None:
            continue
        boundary_warnings.append(BoundaryWarning(between=(g1.group_id, g2.group_id), reason=reason))

    return MergeNeighborhoodReport(
        groups=tuple(typed_groups),
        excluded=excluded,
        boundary_warnings=tuple(boundary_warnings),
    )


def explain_edges_for_group(group: NeighborhoodGroup, edges: list[CompatibilityEdge]) -> list[CompatibilityEdge]:
    """Return edges whose endpoints are fully contained in the given group."""
    members = set(group.members)
    return [e for e in edges if e.adapter_a in members and e.adapter_b in members]
