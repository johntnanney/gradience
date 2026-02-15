"""
Escalation (safety fallback) for the bench protocol.

When an audit-driven compression candidate fails catastrophically, the bench
automatically escalates to a safer (higher) rank and re-evaluates.  This turns
"FAIL" into a credible story: tested aggressive → observed instability →
self-corrected → delivered validated compression.

All functions are pure (no side effects) except the dataclass mutators.
Training is handled by the caller via the existing
``run_compressed_variant_training`` / ``run_all_compressed_variants`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

from gradience.bench.constants import (
    DEFAULT_CATASTROPHIC_MARGIN_MIN,
    DEFAULT_CATASTROPHIC_MARGIN_RATIO,
    DEFAULT_ACCURACY_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

def classify_failure(
    worst_delta: float,
    acc_tolerance: float,
    catastrophic_margin: Optional[float] = None,
) -> str:
    """Classify a variant's failure severity.

    Args:
        worst_delta: The worst delta_vs_probe (single seed or min across seeds).
        acc_tolerance: The configured accuracy tolerance (e.g. 0.005).
        catastrophic_margin: Override for the catastrophic margin.  When *None*,
            auto-computed as ``max(MIN, RATIO * acc_tolerance)``.

    Returns:
        ``"none"`` – within tolerance (PASS territory).
        ``"near_miss"`` – failed, but not by a wide margin.
        ``"catastrophic"`` – fell off a cliff; escalation warranted.
    """
    if catastrophic_margin is None:
        catastrophic_margin = max(
            DEFAULT_CATASTROPHIC_MARGIN_MIN,
            DEFAULT_CATASTROPHIC_MARGIN_RATIO * acc_tolerance,
        )

    catastrophic_threshold = -(acc_tolerance + catastrophic_margin)

    if worst_delta >= -acc_tolerance:
        return "none"
    elif worst_delta < catastrophic_threshold:
        return "catastrophic"
    else:
        return "near_miss"


# ---------------------------------------------------------------------------
# Escalation candidate computation
# ---------------------------------------------------------------------------

def compute_escalation_candidate(
    failed_variant: str,
    failed_rank: int,
    probe_rank: int,
    allowed_ranks: List[int],
    already_tested_ranks: set,
) -> Optional[Dict[str, Any]]:
    """Return the next-higher-rank escalation candidate, or *None*.

    Walks *allowed_ranks* upward from *failed_rank*, skipping ranks that have
    already been tested and stopping before *probe_rank* (no point testing the
    full probe width).

    The returned dict is shaped for ``run_compressed_variant_training``'s
    *compression_config* parameter.  The caller must fill in the ``"config"``
    key with the full LoRA configuration.
    """
    sorted_ranks = sorted(allowed_ranks)
    higher_ranks = [r for r in sorted_ranks if r > failed_rank and r < probe_rank]

    for candidate_rank in higher_ranks:
        if candidate_rank not in already_tested_ranks:
            esc_name = f"{failed_variant}_esc_r{candidate_rank}"
            return {
                "variant": esc_name,
                "suggested_r": candidate_rank,
                "actual_r": candidate_rank,
                "policy_type": "escalation",
                "conservatism_score": 5.0,  # Most conservative tier
                "original_variant": failed_variant,
                "original_rank": failed_rank,
                "escalation_rank": candidate_rank,
                "status": "ready",
                # "config" populated by the caller
            }

    return None


# ---------------------------------------------------------------------------
# Trace data structures
# ---------------------------------------------------------------------------

@dataclass
class EscalationTraceEntry:
    """Record of a single escalation attempt."""

    original_variant: str
    original_rank: int
    failure_mode: str          # "catastrophic"
    worst_delta: float
    catastrophic_threshold: float
    escalated_to: Optional[str] = None       # e.g. "energy_p90_esc_r12"
    escalation_rank: Optional[int] = None
    escalation_result: Optional[str] = None  # "PASS" | "FAIL" | None


@dataclass
class EscalationTrace:
    """Top-level escalation trace for the bench report."""

    triggered: bool = False
    candidates_tested: List[str] = field(default_factory=list)
    escalation_trace: List[EscalationTraceEntry] = field(default_factory=list)
    final_recommendation: Optional[str] = None
    recommendation_rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "triggered": self.triggered,
            "candidates_tested": self.candidates_tested,
            "escalation_trace": [asdict(e) for e in self.escalation_trace],
            "final_recommendation": self.final_recommendation,
            "recommendation_rationale": self.recommendation_rationale,
        }


# ---------------------------------------------------------------------------
# Orchestration helper (does NOT train — returns configs for the caller)
# ---------------------------------------------------------------------------

def run_escalation_round(
    variant_results: Dict[str, Dict[str, Any]],
    verdicts: Dict[str, Any],
    compression_configs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    probe_rank: int,
    acc_tolerance: float,
    escalation_config: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], EscalationTrace]:
    """Analyse verdicts for catastrophic failures and produce escalation candidates.

    This function does **not** train anything.  It returns escalation
    compression configs that the caller passes to
    ``run_all_compressed_variants`` / ``run_compressed_variant_training``.

    Returns:
        ``(escalation_compression_configs, escalation_trace)``
    """
    enabled = escalation_config.get("enabled", True)
    catastrophic_margin = escalation_config.get("catastrophic_margin")
    allowed_ranks = config.get("compression", {}).get("allowed_ranks", [])
    lora_config = config.get("lora", {})

    trace = EscalationTrace()
    escalation_configs: Dict[str, Dict[str, Any]] = {}

    if not enabled:
        return escalation_configs, trace

    # Collect ranks already tested (from original candidates + variant results)
    already_tested_ranks: set[int] = set()
    for _vname, vconfig in compression_configs.items():
        if vconfig.get("status") == "ready" and "actual_r" in vconfig:
            already_tested_ranks.add(vconfig["actual_r"])
    for _vname, vresult in variant_results.items():
        if vresult.get("status") == "completed" and "rank" in vresult:
            already_tested_ranks.add(vresult["rank"])

    # Compute the effective margin for threshold reporting
    effective_margin = catastrophic_margin
    if effective_margin is None:
        effective_margin = max(
            DEFAULT_CATASTROPHIC_MARGIN_MIN,
            DEFAULT_CATASTROPHIC_MARGIN_RATIO * acc_tolerance,
        )
    catastrophic_threshold = -(acc_tolerance + effective_margin)

    for variant_name, verdict in verdicts.items():
        if verdict.get("verdict") != "FAIL":
            continue

        delta = verdict.get("delta_vs_probe")
        if delta is None:
            continue

        failure_mode = classify_failure(delta, acc_tolerance, catastrophic_margin)
        if failure_mode != "catastrophic":
            continue

        # Look up the failed variant's rank
        vconfig = compression_configs.get(variant_name, {})
        failed_rank = vconfig.get("actual_r")
        if failed_rank is None:
            failed_rank = variant_results.get(variant_name, {}).get("rank")
        if failed_rank is None:
            continue

        # Compute the escalation candidate
        esc_candidate = compute_escalation_candidate(
            failed_variant=variant_name,
            failed_rank=failed_rank,
            probe_rank=probe_rank,
            allowed_ranks=allowed_ranks,
            already_tested_ranks=already_tested_ranks,
        )

        trace_entry = EscalationTraceEntry(
            original_variant=variant_name,
            original_rank=failed_rank,
            failure_mode=failure_mode,
            worst_delta=delta,
            catastrophic_threshold=catastrophic_threshold,
            escalated_to=esc_candidate["variant"] if esc_candidate else None,
            escalation_rank=esc_candidate["escalation_rank"] if esc_candidate else None,
            escalation_result=None,  # filled after training
        )
        trace.escalation_trace.append(trace_entry)

        if esc_candidate is not None:
            trace.triggered = True
            esc_name = esc_candidate["variant"]
            esc_rank = esc_candidate["actual_r"]

            # Build the full LoRA config for training
            esc_candidate["config"] = {
                **lora_config,
                "probe_r": esc_rank,
                "alpha": esc_rank,
            }
            escalation_configs[esc_name] = esc_candidate
            trace.candidates_tested.append(esc_name)

            # Prevent re-use of this rank in later iterations
            already_tested_ranks.add(esc_rank)

    return escalation_configs, trace


# ---------------------------------------------------------------------------
# Post-training trace update
# ---------------------------------------------------------------------------

def update_escalation_trace_with_results(
    trace: EscalationTrace,
    escalation_verdicts: Dict[str, Any],
    original_best_compression: Optional[Dict[str, Any]],
) -> EscalationTrace:
    """Fill in escalation results and derive the final recommendation.

    Called after escalation candidates have been trained and verdicted.
    """
    for entry in trace.escalation_trace:
        if entry.escalated_to and entry.escalated_to in escalation_verdicts:
            v = escalation_verdicts[entry.escalated_to]
            entry.escalation_result = v.get("verdict", "UNKNOWN")

    # Derive final recommendation
    passing_escalations = [
        e for e in trace.escalation_trace
        if e.escalation_result == "PASS"
    ]

    if passing_escalations:
        # Pick the one with the highest compression (lowest rank)
        best = min(passing_escalations, key=lambda e: e.escalation_rank or float("inf"))
        trace.final_recommendation = best.escalated_to
        trace.recommendation_rationale.append(
            f"Escalated from {best.original_variant} (r={best.original_rank}) "
            f"to r={best.escalation_rank}: passed."
        )
    elif original_best_compression:
        trace.final_recommendation = original_best_compression.get("variant")
        trace.recommendation_rationale.append(
            "Escalation candidates did not pass; retaining original best compression."
        )
    else:
        trace.final_recommendation = None
        trace.recommendation_rationale.append(
            "No passing variants found, including escalation attempts."
        )

    return trace


# ---------------------------------------------------------------------------
# Verdict enrichment
# ---------------------------------------------------------------------------

def enrich_verdicts_with_stability(
    verdicts: Dict[str, Any],
    acc_tolerance: float,
    escalation_trace: EscalationTrace,
    catastrophic_margin: Optional[float] = None,
) -> Dict[str, Any]:
    """Add stability metadata to each verdict dict (mutates in place).

    Fields added per verdict:
        ``stability_status``: ``"stable"`` | ``"near_miss"`` | ``"unstable"``
        ``failure_mode``: ``"none"`` | ``"near_miss"`` | ``"catastrophic"``
        ``escalated_to``: variant name or *None*
        ``escalation_reason``: human-readable explanation or *None*
    """
    # Build lookup: original_variant → trace entry
    escalation_map: Dict[str, EscalationTraceEntry] = {}
    for entry in escalation_trace.escalation_trace:
        escalation_map[entry.original_variant] = entry

    for variant_name, verdict in verdicts.items():
        delta = verdict.get("delta_vs_probe")

        if delta is None:
            verdict["stability_status"] = "unknown"
            verdict["failure_mode"] = "unknown"
            verdict["escalated_to"] = None
            verdict["escalation_reason"] = None
            continue

        fm = classify_failure(delta, acc_tolerance, catastrophic_margin)
        verdict["failure_mode"] = fm

        if fm == "none":
            verdict["stability_status"] = "stable"
        elif fm == "near_miss":
            verdict["stability_status"] = "near_miss"
        else:
            verdict["stability_status"] = "unstable"

        # Link to escalation trace if this variant triggered escalation
        if variant_name in escalation_map:
            entry = escalation_map[variant_name]
            verdict["escalated_to"] = entry.escalated_to
            verdict["escalation_reason"] = (
                f"Catastrophic failure: worst_delta={entry.worst_delta:+.4f}, "
                f"threshold={entry.catastrophic_threshold:+.4f}"
            )
        else:
            verdict["escalated_to"] = None
            verdict["escalation_reason"] = None

    return verdicts
