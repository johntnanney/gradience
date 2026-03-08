"""
Merge Audit & Execution — Public API.

Phase 1 — Analysis: Given two PEFT LoRA adapters, analyse per-layer spectral
compatibility and produce ``merge_audit.json`` / ``merge_audit.md``.

Phase 2 — Execution: Generate a merge plan from the audit report, then
execute the plan to produce a PEFT-compatible merged adapter.

Quick-start::

    from gradience.vnext.merge import merge_audit, plan_from_audit, execute_merge

    # Phase 1: Audit
    report = merge_audit("./adapter_a", "./adapter_b")

    # Phase 2: Plan + Execute
    plan = plan_from_audit("audit_aware", report, "./adapter_a", "./adapter_b")
    result = execute_merge(plan, "./merged_adapter")
    print(f"Reconstruction error: {result.mean_reconstruction_error:.4f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import logging

import torch

from gradience.exceptions import MergeError
from gradience.vnext.merge.io import (
    AdapterInfo,
    extract_factors,
    get_module_type,
    load_adapter,
    match_layers,
)
from gradience.vnext.merge.report import (
    MergeAuditReport,
    build_report,
    to_json,
    to_markdown,
    write_reports,
)
from gradience.vnext.merge.spectral_compat import (
    SubspaceMetrics,
    compute_subspace_metrics,
)
from gradience.vnext.merge.verdicts import (
    CompatibilityVerdict,
    LayerVerdict,
    VerdictThresholds,
    assess_layer,
    assess_overall,
)

# Phase 2 — merge execution
from gradience.vnext.merge.strategies import (
    LayerMergeConfig,
    MergeStrategy,
    LinearMerge,
    TIESMerge,
    DARELinearMerge,
    DARETIESMerge,
    NormEqualizedMerge,
    get_strategy,
)
from gradience.vnext.merge.refactor import refactor_to_lora
from gradience.vnext.merge.plan import (
    MergePlan,
    plan_from_audit,
    PLAN_STRATEGIES,
)
from gradience.vnext.merge.executor import (
    MergeResult,
    LayerMergeResult,
    execute_merge,
)

# Phase 2 — Recommendations
from gradience.vnext.merge.recommend import (
    MergeRecommendation,
    LayerRecommendation,
    recommend_merge,
    format_recommendation,
    rebalance_coefficients,
    norm_equalized_coefficients,
)

# Source eligibility screening
from gradience.vnext.merge.eligibility import (
    EligibilityStatus,
    AdapterQAResult,
    classify_eligibility,
    screen_adapters,
)

# Phase 2 — M1 protocol modules
from gradience.vnext.merge.outcomes import compute_merge_outcomes, is_bad_merge
from gradience.vnext.merge.scale import symmetric_scale_metrics, symmetric_frobenius_metrics
from gradience.vnext.merge.evaluation import merge_prediction_evaluation
from gradience.vnext.merge.null_controls import randomized_subspace_control, layer_shuffle_control
from gradience.vnext.merge.norm_equalized import norm_equalized_merge

logger = logging.getLogger(__name__)

__all__ = [
    # Phase 1 — Audit orchestrator
    "merge_audit",
    # Phase 1 — Data structures
    "AdapterInfo",
    "SubspaceMetrics",
    "CompatibilityVerdict",
    "LayerVerdict",
    "VerdictThresholds",
    "MergeAuditReport",
    # Phase 1 — Helpers
    "load_adapter",
    "match_layers",
    "extract_factors",
    "compute_subspace_metrics",
    "assess_layer",
    "assess_overall",
    "build_report",
    "to_json",
    "to_markdown",
    "write_reports",
    # Phase 2 — Merge strategies
    "LayerMergeConfig",
    "MergeStrategy",
    "LinearMerge",
    "TIESMerge",
    "DARELinearMerge",
    "DARETIESMerge",
    "NormEqualizedMerge",
    "get_strategy",
    # Phase 2 — SVD refactoring
    "refactor_to_lora",
    # Phase 2 — Planning
    "MergePlan",
    "plan_from_audit",
    "PLAN_STRATEGIES",
    # Phase 2 — Execution
    "MergeResult",
    "LayerMergeResult",
    "execute_merge",
    # Phase 2 — Recommendations
    "MergeRecommendation",
    "LayerRecommendation",
    "recommend_merge",
    "format_recommendation",
    "rebalance_coefficients",
    "norm_equalized_coefficients",
    # M1 protocol — Outcome metrics
    "compute_merge_outcomes",
    "is_bad_merge",
    # M1 protocol — Symmetric scale
    "symmetric_scale_metrics",
    "symmetric_frobenius_metrics",
    # M1 protocol — Evaluation
    "merge_prediction_evaluation",
    # M1 protocol — Null controls
    "randomized_subspace_control",
    "layer_shuffle_control",
    # M1 protocol — Norm-equalized merge (M2 stub)
    "norm_equalized_merge",
    # Source eligibility screening
    "EligibilityStatus",
    "AdapterQAResult",
    "classify_eligibility",
    "screen_adapters",
]


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "fp64": torch.float64,
    "fp32": torch.float32,
}


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def merge_audit(
    adapter_a_dir: Union[str, Path],
    adapter_b_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    energy_threshold: float = 0.90,
    thresholds: Optional[VerdictThresholds] = None,
    compute_dtype: str = "float64",
    verbose: bool = False,
    source_qa_a: Optional[AdapterQAResult] = None,
    source_qa_b: Optional[AdapterQAResult] = None,
) -> MergeAuditReport:
    """Run a merge compatibility audit on two PEFT LoRA adapters.

    Parameters
    ----------
    adapter_a_dir : path to the first PEFT adapter directory
    adapter_b_dir : path to the second PEFT adapter directory
    output_dir : optional directory to write merge_audit.json and merge_audit.md
    energy_threshold : fraction of squared-singular-value energy defining
        effective rank (default 0.90)
    thresholds : verdict decision thresholds; ``None`` uses defaults
    compute_dtype : ``"float64"`` (default) or ``"float32"`` for SVD
    verbose : if True, print progress to stdout
    source_qa_a : optional AdapterQAResult for adapter A (eligibility screening)
    source_qa_b : optional AdapterQAResult for adapter B (eligibility screening)

    Returns
    -------
    MergeAuditReport with per-layer and aggregate compatibility results.

    Raises
    ------
    FileNotFoundError
        If adapter directories or required files are missing.
    ValueError
        If adapters have no LoRA pairs or no shared layers.
    """
    if thresholds is None:
        thresholds = VerdictThresholds()

    dtype = _DTYPE_MAP.get(compute_dtype)
    if dtype is None:
        raise MergeError(
            f"Unsupported compute_dtype '{compute_dtype}'. "
            f"Choose from: {list(_DTYPE_MAP.keys())}"
        )

    # --- Step 1: Load adapters ---
    if verbose:
        print(f"Loading adapter A: {adapter_a_dir}")
    info_a = load_adapter(adapter_a_dir)

    if verbose:
        print(f"Loading adapter B: {adapter_b_dir}")
    info_b = load_adapter(adapter_b_dir)

    if verbose:
        print(
            f"  A: rank={info_a.rank}, alpha={info_a.alpha}, "
            f"{len(info_a.lora_pairs)} layers"
        )
        print(
            f"  B: rank={info_b.rank}, alpha={info_b.alpha}, "
            f"{len(info_b.lora_pairs)} layers"
        )

    logger.debug("Loaded adapters: A=%s (rank=%d, %d layers), B=%s (rank=%d, %d layers)", adapter_a_dir, info_a.rank, len(info_a.lora_pairs), adapter_b_dir, info_b.rank, len(info_b.lora_pairs))

    # --- Step 2: Match layers ---
    shared, only_a, only_b = match_layers(info_a, info_b)

    if verbose:
        print(
            f"Layer matching: {len(shared)} shared, "
            f"{len(only_a)} only-A, {len(only_b)} only-B"
        )

    logger.debug("Layer matching: %d shared, %d only-A, %d only-B", len(shared), len(only_a), len(only_b))

    if not shared:
        raise MergeError(
            f"No shared LoRA layers between adapters. "
            f"A has {len(info_a.lora_pairs)} layers, "
            f"B has {len(info_b.lora_pairs)} layers. "
            f"Check that both adapters target the same modules."
        )

    # --- Step 3: Per-layer analysis ---
    layer_verdicts = []

    for i, module_prefix in enumerate(shared):
        if verbose:
            print(
                f"  [{i + 1}/{len(shared)}] Analyzing {module_prefix}...",
                end="",
                flush=True,
            )

        A_a, B_a, r_a = extract_factors(info_a, module_prefix)
        A_b, B_b, r_b = extract_factors(info_b, module_prefix)

        metrics = compute_subspace_metrics(
            A_a, B_a, info_a.alpha, r_a,
            A_b, B_b, info_b.alpha, r_b,
            energy_threshold=energy_threshold,
            compute_dtype=dtype,
        )

        module_type = get_module_type(module_prefix)
        lv = assess_layer(module_prefix, module_type, metrics, thresholds)
        layer_verdicts.append(lv)

        if verbose:
            print(f" {lv.verdict.value} (overlap={metrics.mean_overlap:.3f})")

    # --- Step 3b: Release weight tensors to free memory ---
    # build_report only reads metadata (path, rank, alpha, config.raw,
    # len(lora_pairs)), not state_dict tensors.  Use object.__delattr__
    # to bypass frozen dataclass restriction and free the large dicts.
    try:
        object.__delattr__(info_a, "state_dict")
        object.__delattr__(info_b, "state_dict")
    except (AttributeError, TypeError):
        pass  # best-effort cleanup

    # --- Step 4: Aggregate ---
    overall_verdict, score, recommendations = assess_overall(layer_verdicts)

    if verbose:
        print(
            f"\nOverall verdict: {overall_verdict.value.upper()} "
            f"(score={score:.3f})"
        )

    logger.debug("Merge audit verdict: %s (score=%.3f)", overall_verdict.value, score)

    # --- Step 4b: Source eligibility screening ---
    eligibility_warnings = screen_adapters(source_qa_a, source_qa_b)
    if eligibility_warnings and verbose:
        print("\nSource eligibility warnings:")
        for w in eligibility_warnings:
            print(f"  - {w}")

    # --- Step 5: Build report ---
    report = build_report(
        adapter_a_info=info_a,
        adapter_b_info=info_b,
        shared=shared,
        only_a=only_a,
        only_b=only_b,
        layer_verdicts=layer_verdicts,
        overall_verdict=overall_verdict,
        score=score,
        recommendations=recommendations,
        thresholds=thresholds,
        source_qa_a=source_qa_a,
        source_qa_b=source_qa_b,
    )

    # --- Step 6: Write output files ---
    if output_dir is not None:
        write_reports(report, output_dir)
        if verbose:
            out = Path(output_dir)
            print(f"\nReports written to {out / 'merge_audit.json'}")
            print(f"                    {out / 'merge_audit.md'}")

    return report
