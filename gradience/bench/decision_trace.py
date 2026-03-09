"""
Decision Trace for Bench Candidate Control

Tracks audit-driven decisions for transparency and reproducibility.
Keeps the "why" behind compression candidate selection explicit.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from gradience.bench._util import round_to_allowed_ranks


@dataclass
class RuleRecord:
    """Record of a single decision rule firing (or not)."""

    rule_id: str
    trigger: str  # Human-readable trigger condition
    evidence: dict[str, Any]  # Supporting metrics/data
    action: str  # What was done as a result


@dataclass
class DecisionTrace:
    """
    Complete trace of audit-driven compression decisions.

    JSON-serializable record for bench.json artifacts.
    """

    probe_rank: int

    # Core audit metrics that drive decisions
    audit_metrics: dict[str, float] = field(default_factory=dict)

    # Rules that fired (resulted in actions)
    rules_fired: list[RuleRecord] = field(default_factory=list)

    # Rules that were considered but didn't trigger
    rules_considered: list[RuleRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "probe_rank": self.probe_rank,
            "audit_metrics": self.audit_metrics,
            "rules_fired": [asdict(rule) for rule in self.rules_fired],
            "rules_considered": [asdict(rule) for rule in self.rules_considered],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionTrace":
        """Create from JSON dict."""
        return cls(
            probe_rank=data["probe_rank"],
            audit_metrics=data.get("audit_metrics", {}),
            rules_fired=[RuleRecord(**rule) for rule in data.get("rules_fired", [])],
            rules_considered=[RuleRecord(**rule) for rule in data.get("rules_considered", [])],
        )

    def add_fired_rule(self, rule_id: str, trigger: str, evidence: dict[str, Any], action: str):
        """Record a rule that triggered and caused an action."""
        self.rules_fired.append(RuleRecord(rule_id=rule_id, trigger=trigger, evidence=evidence, action=action))

    def add_considered_rule(self, rule_id: str, trigger: str, evidence: dict[str, Any], reason: str):
        """Record a rule that was evaluated but didn't trigger."""
        self.rules_considered.append(
            RuleRecord(rule_id=rule_id, trigger=trigger, evidence=evidence, action=f"not triggered: {reason}")
        )


def create_decision_trace(probe_rank: int, audit_data: dict[str, Any]) -> DecisionTrace:
    """
    Create decision trace from probe rank and audit data.

    Extracts key audit metrics for decision making.
    """
    # Extract audit summary metrics (handle various schema versions)
    audit_summary = audit_data.get("audit_summary") or audit_data.get("summary") or {}

    audit_metrics = {
        "utilization_mean": audit_summary.get("utilization_mean", 0.0),
        "stable_rank_mean": audit_summary.get("stable_rank_mean", 0.0),
        "current_r": audit_summary.get("current_r", probe_rank),
    }

    # Add additional useful metrics if available
    if "suggested_r_global_median" in audit_data:
        audit_metrics["suggested_r_global_median"] = audit_data["suggested_r_global_median"]
    if "suggested_r_global_90" in audit_data:
        audit_metrics["suggested_r_global_90"] = audit_data["suggested_r_global_90"]

    return DecisionTrace(probe_rank=probe_rank, audit_metrics=audit_metrics)


def maybe_add_second_rung_candidates(
    probe_rank: int,
    audit_metrics: dict[str, float],
    allowed_ranks: list[int],
    existing_candidates: list[str],
    decision_trace: DecisionTrace,
) -> list[dict[str, Any]]:
    """
    Pure function to compute second rung candidates based on 2-tier logic.

    Tier A (moderate slack): utilization_mean <= 0.55 OR suggested_rank <= 0.75 * probe_rank
    Tier B (strong over-provisioning): utilization_mean < 0.30 AND stable_rank_mean <= probe_rank/4

    Args:
        probe_rank: Current probe rank
        audit_metrics: Dict with utilization_mean, stable_rank_mean, etc.
        allowed_ranks: List of allowed ranks for rounding
        existing_candidates: List of existing candidate names to avoid duplicates
        decision_trace: DecisionTrace to record rule evaluations

    Returns:
        List of new candidate dicts to add
    """
    new_candidates = []

    utilization_mean = audit_metrics.get("utilization_mean", 0.0)
    stable_rank_mean = audit_metrics.get("stable_rank_mean", 0.0)
    suggested_r_median = audit_metrics.get("suggested_r_global_median", probe_rank)

    # Tier A: Moderate slack (moderate compression)
    tier_a_trigger_util = utilization_mean <= 0.55
    tier_a_trigger_suggested = suggested_r_median <= 0.75 * probe_rank
    tier_a_triggered = tier_a_trigger_util or tier_a_trigger_suggested

    r1 = round_to_allowed_ranks(int(0.75 * probe_rank), allowed_ranks)
    r1_name = f"uniform_r{r1}"

    if tier_a_triggered and r1 < probe_rank and r1_name not in existing_candidates:
        decision_trace.add_fired_rule(
            rule_id="tier_a_moderate_compression",
            trigger="utilization_mean <= 0.55 OR suggested_rank <= 0.75 * probe_rank",
            evidence={
                "utilization_mean": utilization_mean,
                "threshold_util": 0.55,
                "suggested_r_median": suggested_r_median,
                "threshold_suggested": 0.75 * probe_rank,
                "probe_rank": probe_rank,
                "triggered_by_util": tier_a_trigger_util,
                "triggered_by_suggested": tier_a_trigger_suggested,
            },
            action=f"Added moderate compression candidate: {r1_name} (r={r1})",
        )

        new_candidates.append(
            {
                "name": r1_name,
                "policy_type": "second_rung_tier_a",
                "suggested_r": 0.75 * probe_rank,
                "actual_r": r1,
                "conservatism_score": 2.5,  # Between knee and energy
                "priority": 1,  # Fast mode priority
            }
        )

    else:
        # Record why Tier A didn't trigger
        reason_parts = []
        if not tier_a_trigger_util:
            reason_parts.append(f"utilization {utilization_mean:.3f} > 0.55")
        if not tier_a_trigger_suggested:
            reason_parts.append(f"suggested_r {suggested_r_median:.1f} > {0.75 * probe_rank:.1f}")
        if r1 >= probe_rank:
            reason_parts.append(f"computed_r {r1} >= probe_rank {probe_rank}")
        if r1_name in existing_candidates:
            reason_parts.append(f"candidate {r1_name} already exists")

        decision_trace.add_considered_rule(
            rule_id="tier_a_moderate_compression",
            trigger="utilization_mean <= 0.55 OR suggested_rank <= 0.75 * probe_rank",
            evidence={
                "utilization_mean": utilization_mean,
                "stable_rank_mean": stable_rank_mean,
                "suggested_r_median": suggested_r_median,
                "probe_rank": probe_rank,
            },
            reason=" AND ".join(reason_parts),
        )

    # Tier B: Strong over-provisioning (aggressive compression)
    tier_b_trigger_util = utilization_mean < 0.30
    tier_b_trigger_stable = stable_rank_mean <= probe_rank / 4
    tier_b_triggered = tier_b_trigger_util and tier_b_trigger_stable

    r2 = round_to_allowed_ranks(max(probe_rank // 2, int(0.50 * probe_rank)), allowed_ranks)
    r2_name = f"uniform_r{r2}"

    if tier_b_triggered and r2 < probe_rank and r2_name not in existing_candidates:
        decision_trace.add_fired_rule(
            rule_id="tier_b_aggressive_compression",
            trigger="utilization_mean < 0.30 AND stable_rank_mean <= probe_rank/4",
            evidence={
                "utilization_mean": utilization_mean,
                "threshold_util": 0.30,
                "stable_rank_mean": stable_rank_mean,
                "threshold_stable": probe_rank / 4,
                "probe_rank": probe_rank,
            },
            action=f"Added aggressive compression candidate: {r2_name} (r={r2})",
        )

        new_candidates.append(
            {
                "name": r2_name,
                "policy_type": "second_rung_tier_b",
                "suggested_r": 0.50 * probe_rank,
                "actual_r": r2,
                "conservatism_score": 1.0,  # Most aggressive
                "priority": 1,  # Fast mode priority
            }
        )

    else:
        # Record why Tier B didn't trigger
        reason_parts = []
        if not tier_b_trigger_util:
            reason_parts.append(f"utilization {utilization_mean:.3f} >= 0.30")
        if not tier_b_trigger_stable:
            reason_parts.append(f"stable_rank {stable_rank_mean:.1f} > {probe_rank / 4:.1f}")
        if r2 >= probe_rank:
            reason_parts.append(f"computed_r {r2} >= probe_rank {probe_rank}")
        if r2_name in existing_candidates:
            reason_parts.append(f"candidate {r2_name} already exists")

        decision_trace.add_considered_rule(
            rule_id="tier_b_aggressive_compression",
            trigger="utilization_mean < 0.30 AND stable_rank_mean <= probe_rank/4",
            evidence={
                "utilization_mean": utilization_mean,
                "stable_rank_mean": stable_rank_mean,
                "probe_rank": probe_rank,
            },
            reason=" AND ".join(reason_parts),
        )

    return new_candidates
