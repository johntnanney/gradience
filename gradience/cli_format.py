"""CLI output formatting functions.

Pure output functions that take structured data and produce terminal output.
Separated from cli_utils.py to enable independent testing and alternative
output formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _severity_rank(sev: str) -> int:
    order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    return order.get(sev.lower(), 99)



def _fmt(x: Any, *, pct: bool = False) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (ValueError, TypeError):
        return str(x)
    if pct:
        return f"{xf * 100:.1f}%"
    # Use compact scientific for very small/large
    if abs(xf) != 0 and (abs(xf) < 1e-3 or abs(xf) >= 1e4):
        return f"{xf:.2e}"
    return f"{xf:.3g}"



def _fmt_params(n) -> str:
    """Format a parameter count into human-friendly units (K/M/B)."""
    if n is None:
        return "n/a"
    try:
        x = float(n)
    except (ValueError, TypeError):
        return str(n)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.1f}K"
    if x.is_integer():
        return str(int(x))
    return f"{x:.3g}"



def _print_recommendations(config: Any, recs: list[Any], *, verbose: bool = False) -> None:
    print("=" * 72)
    print("GRADIENCE CHECK")
    print("=" * 72)

    # Config summary
    model_name = getattr(config, "model_name", None)
    dataset_name = getattr(config, "dataset_name", None)
    task_profile = getattr(getattr(config, "task_profile", None), "value", None) or str(
        getattr(config, "task_profile", "unknown")
    )

    opt = getattr(config, "optimizer", None)
    lora = getattr(config, "lora", None)

    lr = getattr(opt, "lr", None)
    wd = getattr(opt, "weight_decay", None)
    r = getattr(lora, "r", None)
    alpha = getattr(lora, "alpha", None)
    a_over_r = getattr(lora, "alpha_over_r", None)
    targets = list(getattr(lora, "target_modules", []) or [])

    print(f"Model:   {model_name or '-'}")
    print(f"Dataset: {dataset_name or '-'}")
    print(f"Profile: {task_profile}")

    print("\nKey knobs:")
    print(f"  LR:          {lr if lr is not None else '-'}")
    print(f"  Weight decay:{wd if wd is not None else '-'}")
    print(f"  Targets:     {', '.join(targets) if targets else '-'}")
    print(f"  Rank r:      {r if r is not None else '-'}")
    if alpha is None or a_over_r is None:
        print(f"  Alpha:       {alpha if alpha is not None else '-'}")
    else:
        print(f"  Alpha:       {alpha} (α/r={a_over_r:.3g})")

    if not recs:
        print("\nNo recommendations. Config looks reasonable.")
        return

    # Sort by severity then action
    recs_sorted = sorted(
        recs,
        key=lambda r: (
            _severity_rank(getattr(getattr(r, "severity", None), "value", str(getattr(r, "severity", "info")))),
            str(getattr(r, "action", "")),
        ),
    )

    print(f"\nRecommendations ({len(recs_sorted)}):")
    for i, rec in enumerate(recs_sorted, 1):
        sev = getattr(getattr(rec, "severity", None), "value", str(getattr(rec, "severity", "info"))).upper()
        action = getattr(rec, "action", "")
        msg = getattr(rec, "message", "")
        print(f"  {i:02d}. [{sev}] {action}: {msg}")

        if verbose:
            rationale = getattr(rec, "rationale", None)
            confidence = getattr(rec, "confidence", None)
            scope = getattr(rec, "scope", None)
            evidence = getattr(rec, "evidence", None)
            if rationale:
                print(f"       why: {rationale}")
            if confidence is not None:
                print(f"       confidence: {confidence:.2f}")
            if scope:
                print(f"       scope: {scope}")
            if evidence:
                try:
                    ev_json = json.dumps(evidence, sort_keys=True)
                except (TypeError, ValueError):
                    ev_json = str(evidence)
                print(f"       evidence: {ev_json}")



def _print_monitor_result(
    *,
    telemetry_path: Path,
    config: Any,
    signals: Any,
    alerts: list[dict[str, Any]],
    recs: list[Any],
    issues: list[str],
    verbose: bool = False,
    guard_activity: dict[str, Any] | None = None,
) -> None:
    print("=" * 72)
    print("GRADIENCE MONITOR")
    print("=" * 72)
    print(f"File: {telemetry_path}")

    # Config summary (best-effort)
    model_name = getattr(config, "model_name", None) if config is not None else None
    dataset_name = getattr(config, "dataset_name", None) if config is not None else None
    task_profile = None
    if config is not None:
        task_profile = getattr(getattr(config, "task_profile", None), "value", None) or str(
            getattr(config, "task_profile", "unknown")
        )
    else:
        # Fall back to summarize() extras
        try:
            task_profile = (getattr(signals, "extras", {}) or {}).get("task_profile")
            model_name = model_name or (getattr(signals, "extras", {}) or {}).get("model_name")
            dataset_name = dataset_name or (getattr(signals, "extras", {}) or {}).get("dataset_name")
        except (AttributeError, KeyError, TypeError):
            task_profile = task_profile or "unknown"

    print(f"Model:   {model_name or '-'}")
    print(f"Dataset: {dataset_name or '-'}")
    print(f"Profile: {task_profile or 'unknown'}")

    # Signals summary
    train = getattr(signals, "train", None)
    test = getattr(signals, "test", None)

    train_ppl = getattr(train, "ppl", None) if train is not None else None
    test_ppl = getattr(test, "ppl", None) if test is not None else None
    train_acc = getattr(train, "accuracy", None) if train is not None else None
    test_acc = getattr(test, "accuracy", None) if test is not None else None
    gap = getattr(signals, "gap", None)

    print("\nLatest eval signals:")
    print(f"  Train PPL: {_fmt(train_ppl)}")
    print(f"  Test  PPL: {_fmt(test_ppl)}")
    print(f"  Gap:       {_fmt(gap)}x")
    print(f"  Train Acc: {_fmt(train_acc, pct=True)}")
    print(f"  Test  Acc: {_fmt(test_acc, pct=True)}")

    # Optional structural signals
    sr = getattr(signals, "stable_rank_mean", None)
    util = getattr(signals, "utilization_mean", None)
    dom = getattr(signals, "dominance_act_mean", None)
    kap = getattr(signals, "kappa_mean", None)

    if any(v is not None for v in (sr, util, dom, kap)):
        print("\nDiagnostics:")
        if sr is not None:
            print(f"  Stable rank (mean): {_fmt(sr)}")
        if util is not None:
            print(f"  Utilization (mean): {_fmt(util, pct=True)}")
            # Dominance ingredients (scaled) — verbose only
            if verbose:
                _la = (getattr(signals, "extras", None) or {}).get("lora_audit") or {}
                _s50 = _la.get("delta_sigma_max_scaled_p50")
                _s90 = _la.get("delta_sigma_max_scaled_p90")
                _f50 = _la.get("delta_frob_norm_scaled_p50")
                _f90 = _la.get("delta_frob_norm_scaled_p90")
                if any(v is not None for v in (_s50, _s90, _f50, _f90)):
                    _s50s = "n/a" if _s50 is None else f"{float(_s50):.4g}"
                    _s90s = "n/a" if _s90 is None else f"{float(_s90):.4g}"
                    _f50s = "n/a" if _f50 is None else f"{float(_f50):.4g}"
                    _f90s = "n/a" if _f90 is None else f"{float(_f90):.4g}"
                    print("  Dominance (scaled):")
                    print(f"    sigma_max_scaled (p50/p90): {_s50s}/{_s90s}")
                    print(f"    frob_norm_scaled (p50/p90): {_f50s}/{_f90s}")

        if dom is not None:
            print(f"  Activation dominance (mean): {_fmt(dom)}")
        if kap is not None:
            print(f"  Kappa (mean): {_fmt(kap)}")

    # Optional: surface LoRA audit stats if present in telemetry summary.
    audit = None
    try:
        audit = (getattr(signals, "extras", {}) or {}).get("lora_audit")
    except (AttributeError, KeyError, TypeError):
        audit = None

    if isinstance(audit, dict) and audit:
        total_params = audit.get("total_lora_params")
        n_layers = audit.get("n_layers")
        e90_p50 = audit.get("energy_rank_90_p50")
        e90_p90 = audit.get("energy_rank_90_p90")

        def _fmt_dom_params(p: Any) -> str:
            try:
                pf = float(p)
            except (ValueError, TypeError):
                return "-"
            if pf >= 1e6:
                return f"{pf / 1e6:.1f}M"
            if pf >= 1e3:
                return f"{pf / 1e3:.1f}K"
            return f"{pf:.0f}"

        print("\nLoRA audit:")
        print(f"  LoRA params: {_fmt_params(total_params)}")
        print(f"  Layers:      {_fmt(n_layers)}")
        if e90_p50 is not None or e90_p90 is not None:
            print(f"  Energy rank k@90% (p50/p90): {_fmt(e90_p50)}/{_fmt(e90_p90)}")

            # Suggested rank printout (global)
            try:
                s_med = audit.get("suggested_r_global_median")
                s_p90 = audit.get("suggested_r_global_90")
                p50 = audit.get("energy_rank_90_p50")
                p90 = audit.get("energy_rank_90_p90")
            except (AttributeError, KeyError, TypeError):
                s_med = s_p90 = p50 = p90 = None

            if s_med:
                print(f"  Suggested rank (median): r={int(s_med)} likely sufficient for most layers (p50 k@90%={p50})")
            if s_p90:
                print(
                    f"  Suggested rank (p90):    r={int(s_p90)} covers worst-case layers at 90% energy (p90 k@90%={p90})"
                )

        by_type = audit.get("by_type")
        if isinstance(by_type, dict) and by_type:
            # Show a compact per-type breakdown.
            for t in ("attn", "mlp", "other"):
                row = by_type.get(t)
                if not isinstance(row, dict):
                    continue
                t_params = row.get("params")
                t_util = row.get("utilization_mean")
                t_sr = row.get("stable_rank_mean")
                print(f"  {t:>5}: params={_fmt_params(t_params)}  util={_fmt(t_util, pct=True)}  sr={_fmt(t_sr)}")

    # Guard activity
    if guard_activity and guard_activity.get("present"):
        # In verbose mode, always show Guard activity if present
        if verbose:
            print("\nGuard activity:")
            print(f"  Last action:    {guard_activity.get('last_action', '-')}")
            print(f"  Rollbacks:      {guard_activity.get('rollback_count', 0)}")
            if guard_activity.get("last_trigger_code"):
                print(f"  Last trigger:   {guard_activity['last_trigger_code']}")
            print(f"  Snapshots:      {guard_activity.get('snapshot_count', 0)}")
            print(f"  Memory usage:   {_fmt(guard_activity.get('memory_mb', 0))} MB")

        # In non-verbose mode, only show if rollback occurred or training aborted
        elif guard_activity.get("rollback_occurred") or guard_activity.get("aborted"):
            if guard_activity.get("rollback_occurred"):
                rollback_count = guard_activity.get("rollback_count", 1)
                print(f"\n⚠ Guard performed {rollback_count} rollback(s) during training")
            if guard_activity.get("aborted"):
                print("⚠ Guard aborted rollback attempts (anti-thrash protection)")

    # Issues
    if issues:
        print(f"\nTelemetry issues: {len(issues)}")
        if verbose:
            for line in issues[:10]:
                print(f"  - {line}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")

    # Alerts
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for i, a in enumerate(alerts, 1):
            sev = str(a.get("severity", "info")).upper()
            code = a.get("code", "")
            msg = a.get("message", "")
            print(f"  {i:02d}. [{sev}] {code}: {msg}")
            if verbose and a.get("context"):
                try:
                    ctx = json.dumps(a["context"], sort_keys=True)
                except (TypeError, ValueError):
                    ctx = str(a["context"])
                print(f"       context: {ctx}")

    # Recommendations
    if recs:
        print(f"\nRecommendations ({len(recs)}):")
        for i, rec in enumerate(recs, 1):
            sev = getattr(getattr(rec, "severity", None), "value", str(getattr(rec, "severity", "info"))).upper()
            action = getattr(rec, "action", "")
            msg = getattr(rec, "message", "")
            print(f"  {i:02d}. [{sev}] {action}: {msg}")
            if verbose:
                rationale = getattr(rec, "rationale", None)
                confidence = getattr(rec, "confidence", None)
                scope = getattr(rec, "scope", None)
                evidence = getattr(rec, "evidence", None)
                if rationale:
                    print(f"       why: {rationale}")
                if confidence is not None:
                    try:
                        print(f"       confidence: {float(confidence):.2f}")
                    except (ValueError, TypeError):
                        print(f"       confidence: {confidence}")
                if scope:
                    print(f"       scope: {scope}")
                if evidence:
                    try:
                        ev_json = json.dumps(evidence, sort_keys=True)
                    except (TypeError, ValueError):
                        ev_json = str(evidence)
                    print(f"       evidence: {ev_json}")
    else:
        print("\nNo recommendations emitted.")



def _print_policy_disagreement_summary(
    layers: list[Any],
    name_mapping: dict[str, str],
    importance_config: dict[str, Any] | None = None,
    rationale_verbosity: str = "flagged_only",
) -> None:
    """Print smart policy disagreement analysis weighted by layer importance.

    Args:
        layers: List of layer objects with rank suggestions
        name_mapping: Mapping of internal policy names to user-friendly names
        importance_config: Configuration for importance thresholds:
            - quantile_threshold: Quantile threshold for energy share filtering (default: 0.75)
            - uniform_mult_gate: Uniform multiplier gate threshold (default: 1.5)
            - metric: Energy importance metric to use (default: 'energy_share')
    """
    from gradience.policy_analysis import (
        compute_energy_distribution,
        compute_layer_importance_scores,
        filter_disagreement_layers,
    )

    if importance_config is None:
        importance_config = {}

    quantile_threshold = importance_config.get("quantile_threshold", 0.75)
    min_uniform_mult = importance_config.get("uniform_mult_gate", 1.5)

    if not layers:
        return

    # Shared computation steps 1-3
    layer_analysis, importance_scores = compute_layer_importance_scores(layers, name_mapping, importance_config)

    if not importance_scores:
        return

    total_energy, uniform_share, max_uniform_mult, distribution_is_flat = compute_energy_distribution(
        layer_analysis, min_uniform_mult
    )

    smart_disagreement_layers, all_disagreement_layers = filter_disagreement_layers(
        layer_analysis, importance_scores, quantile_threshold, min_uniform_mult, distribution_is_flat
    )

    # Step 4: Smart output - handle flat vs concentrated distributions differently
    if all_disagreement_layers:
        if distribution_is_flat:
            # FLAT DISTRIBUTION: No meaningfully high-impact layers
            print(f"\n🔍 Policy disagreement detected ({len(all_disagreement_layers)} layers):")
            print(f"  Energy distribution is flat (no layer captures ≥ {min_uniform_mult:.1f}× its uniform share)")
            print("  Treating all disagreement layers as medium impact; prioritize by spread.")
            print()
            print("  Layer                                   Spread  Range   Uniform×   Policies suggest")
            print("  -------------------------------------  ------  ------  --------   ----------------")

            # Sort by priority_score (incorporates both spread and uniform_mult)
            sorted_layers = sorted(all_disagreement_layers, key=lambda x: x["priority_score"], reverse=True)

            for layer_info in sorted_layers[:8]:  # Show top 8 by spread
                layer_name = layer_info["layer_name"]
                if len(layer_name) > 35:
                    layer_name = layer_name[:32] + "..."

                spread = layer_info["spread"]
                min_k = layer_info["min_k"]
                max_k = layer_info["max_k"]
                uniform_mult = layer_info["uniform_mult"]

                # Create a compact representation of policy suggestions
                policy_strs = []
                for policy, k in zip(layer_info["policies"], layer_info["k_values"]):
                    policy_strs.append(f"{policy}={k}")

                policies_summary = ", ".join(policy_strs[:2])  # Show first 2 policies
                if len(policy_strs) > 2:
                    policies_summary += f", +{len(policy_strs) - 2}"

                print(
                    f"  {layer_name:<35}  {spread:>6}  {min_k}-{max_k:<4}   {uniform_mult:>6.1f}×   {policies_summary}"
                )

            if len(all_disagreement_layers) > 8:
                print(f"  ... and {len(all_disagreement_layers) - 8} more layers")

            print(f"\n💡 Recommendation: Energy distribution is flat (max={max_uniform_mult:.1f}× uniform share)")
            print("  No layer captures a meaningful fraction of adapter's update energy.")
            print("  Consider Bench validation on highest spread layers or policy consensus.")

        elif smart_disagreement_layers:
            # CONCENTRATED DISTRIBUTION: Clear high-impact layers
            # Sort by combined importance and disagreement (importance * spread)
            # Sort by priority_score (spread_norm * uniform_mult) for Bench ordering
            smart_disagreement_layers.sort(key=lambda x: x["priority_score"], reverse=True)

            print(f"\n🔍 Critical policy disagreement detected ({len(smart_disagreement_layers)} high-impact layers):")
            print("  These layers capture meaningful fractions of adapter energy AND show policy ambiguity!")
            print("  Layer                                   Spread  Range   Uniform×   Policies suggest")
            print("  -------------------------------------  ------  ------  --------   ----------------")

            for layer_info in smart_disagreement_layers[:8]:  # Show top 8 critical layers
                layer_name = layer_info["layer_name"]
                if len(layer_name) > 35:
                    layer_name = layer_name[:32] + "..."

                spread = layer_info["spread"]
                min_k = layer_info["min_k"]
                max_k = layer_info["max_k"]
                uniform_mult = layer_info["uniform_mult"]

                # Create a compact representation of policy suggestions
                policy_strs = []
                for policy, k in zip(layer_info["policies"], layer_info["k_values"]):
                    policy_strs.append(f"{policy}={k}")

                policies_summary = ", ".join(policy_strs[:2])  # Show first 2 policies
                if len(policy_strs) > 2:
                    policies_summary += f", +{len(policy_strs) - 2}"

                print(
                    f"  {layer_name:<35}  {spread:>6}  {min_k}-{max_k:<4}   {uniform_mult:>6.1f}×   {policies_summary}"
                )

            if len(smart_disagreement_layers) > 8:
                print(f"  ... and {len(smart_disagreement_layers) - 8} more critical layers")

            print(
                f"\n🎯 Priority: Focus Bench validation on these {len(smart_disagreement_layers)} energy-significant layers"
            )

            # Show top focus layer with priority score for Bench ordering
            if smart_disagreement_layers:
                top_layer = smart_disagreement_layers[0]
                layer_name = top_layer["layer_name"]
                priority_score = top_layer["priority_score"]
                print(f"💡 Top focus layer: {layer_name} (priority_score={priority_score:.1f})")

            # Show summary of less important disagreements
            low_importance_count = len(all_disagreement_layers) - len(smart_disagreement_layers)
            if low_importance_count > 0:
                print(f"\n📊 Additional info: {low_importance_count} lower-importance layers also show disagreement")
                print(f"  (uniform mult < {min_uniform_mult:.1f}× or insufficient energy share; deprioritized)")

            # Overall recommendation for concentrated distribution
            total_disagreements = len(all_disagreement_layers)
            critical_disagreements = len(smart_disagreement_layers)
            print(
                f"\n💡 Smart recommendation: Prioritize Bench validation on {critical_disagreements}/{total_disagreements} energy-significant layers"
            )

        else:
            # CONCENTRATED but no high-impact disagreements (shouldn't happen but handle gracefully)
            print(
                f"\n💡 Note: {len(all_disagreement_layers)} layers show disagreement but don't meet high-impact criteria"
            )
            print(f"  (requires: meaningful energy share AND ≥ {min_uniform_mult:.1f}× uniform)")
            print("  Consider bulk validation or accept policy consensus")

    else:
        print(f"\n✅ Policy consensus: No significant disagreements detected across {len(layer_analysis)} layers")



def _print_audit_summary(
    result: Any, *, top_wasteful: int = 0, importance_config: dict[str, Any] | None = None
) -> None:
    """Pretty-print a compact LoRA audit summary."""

    def _fmt_params(p: Any) -> str:
        try:
            pf = float(p)
        except (ValueError, TypeError):
            return "-"
        if pf >= 1e6:
            return f"{pf / 1e6:.1f}M"
        if pf >= 1e3:
            return f"{pf / 1e3:.1f}K"
        return f"{pf:.0f}"

    print("=" * 72)
    print("GRADIENCE LoRA AUDIT")
    print("=" * 72)

    peft_dir = getattr(result, "peft_dir", None)
    cfg_path = getattr(result, "adapter_config_path", None)
    w_path = getattr(result, "adapter_weights_path", None)

    if peft_dir:
        print(f"PEFT dir: {peft_dir}")
    if cfg_path:
        print(f"Config:   {cfg_path}")
    if w_path:
        print(f"Weights:  {w_path}")

    print("\nSummary:")
    print(f"  LoRA params: {_fmt_params(getattr(result, 'total_lora_params', None))}")
    print(f"  Layers:      {_fmt(getattr(result, 'n_layers', None))}")
    print(f"  Stable rank (mean):    {_fmt(getattr(result, 'stable_rank_mean', None))}")
    print(f"  Stable rank (median):  {_fmt(getattr(result, 'stable_rank_median', None))}")
    print(f"  Stable rank (w-mean):  {_fmt(getattr(result, 'stable_rank_weighted_mean', None))}")
    print(f"  Effective rank (mean): {_fmt(getattr(result, 'effective_rank_mean', None))}")
    print(f"  Utilization (mean):    {_fmt(getattr(result, 'utilization_mean', None), pct=True)}")

    e90_p50 = getattr(result, "energy_rank_90_p50", None)
    e90_p90 = getattr(result, "energy_rank_90_p90", None)
    if e90_p50 is not None or e90_p90 is not None:
        print(f"  Energy rank k@90% (p50/p90): {_fmt(e90_p50)}/{_fmt(e90_p90)}")

        # Suggested rank printout (audit)
        def _snap_rank(_k):
            if _k is None:
                return None
            try:
                _k = float(_k)
            except (ValueError, TypeError):
                return None
            for _r in (1, 2, 4, 8, 16, 32):
                if _k <= _r:
                    return _r
            return 32

        _p50 = e90_p50  # Already extracted from result above
        _p90 = e90_p90
        _s_med = _snap_rank(_p50)
        _s_p90 = _snap_rank(_p90)
        if _s_med is not None:
            print(f"  Suggested rank (median): r={int(_s_med)} likely sufficient for most layers (p50 k@90%={_p50})")
        if _s_p90 is not None:
            print(
                f"  Suggested rank (p90):    r={int(_s_p90)} covers worst-case layers at 90% energy (p90 k@90%={_p90})"
            )

    # Policy-based rank suggestions table (Step 6)
    policy_suggestions = getattr(result, "policy_global_suggestions", None)
    if policy_suggestions and isinstance(policy_suggestions, dict):
        print("\nRank policy suggestions:")
        print("  Policy            Median   P90   Max   Don't Compress")
        print("  ----------------  ------  ----  ----  --------------")

        # Map internal names back to user-friendly names
        name_mapping = {
            "energy_threshold": "energy@0.90",
            "knee_elbow": "knee",
            "entropy_effective": "erank",
            "optimal_hard_threshold": "oht (exp.)",  # Mark as experimental
        }

        # Track experimental policies for footnotes
        experimental_policies = {"optimal_hard_threshold"}

        # Get layers for "don't compress" analysis
        layers = getattr(result, "layers", [])
        r_alloc_values = [layer.r for layer in layers] if layers else []
        _typical_r = max(r_alloc_values) if r_alloc_values else 8

        for policy_internal, stats in policy_suggestions.items():
            # Map to user-friendly name
            policy_name = name_mapping.get(policy_internal, policy_internal)

            median = int(stats["uniform_median"])
            p90 = int(stats["uniform_p90"])
            max_val = int(stats["uniform_max"])
            n_layers = int(stats.get("n_layers", 0))

            # Calculate "don't compress" percentage
            dont_compress_count = 0
            if layers:
                for layer in layers:
                    if (
                        layer.rank_suggestions
                        and policy_internal in layer.rank_suggestions
                        and "k" in layer.rank_suggestions[policy_internal]
                    ):
                        k = layer.rank_suggestions[policy_internal]["k"]
                        # Consider "don't compress" if suggested rank is >= 80% of allocated rank
                        if k >= 0.8 * layer.r:
                            dont_compress_count += 1

            dont_compress_pct = (dont_compress_count / n_layers * 100) if n_layers > 0 else 0

            print(f"  {policy_name:<16}  {median:>6}  {p90:>4}  {max_val:>4}       {dont_compress_pct:>4.0f}%")

        # Add footnote for experimental policies
        has_experimental = any(policy_internal in experimental_policies for policy_internal in policy_suggestions)
        if has_experimental:
            print("  (exp.) = Experimental policy based on theoretical assumptions")

        # Policy disagreement analysis (Step 10)
        _print_policy_disagreement_summary(layers, name_mapping, importance_config)

    by_type = getattr(result, "by_type", None)
    if isinstance(by_type, dict) and by_type:
        print("\nBy module type:")
        for t in ("attn", "mlp", "other"):
            row = by_type.get(t)
            if not isinstance(row, dict):
                continue
            print(
                f"  {t:>5}: params={_fmt_params(row.get('params'))}  layers={_fmt(row.get('n_layers'))}  "
                f"util={_fmt(row.get('utilization_mean'), pct=True)}  sr={_fmt(row.get('stable_rank_mean'))}"
            )

    issues = getattr(result, "issues", None)
    if isinstance(issues, list) and issues:
        print(f"\nIssues ({len(issues)}):")
        for line in issues[:10]:
            print(f"  - {line}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    if top_wasteful and int(top_wasteful) > 0:
        layers = getattr(result, "layers", None)
        if isinstance(layers, list) and layers:
            # Most wasteful = lowest utilization
            ls = sorted(layers, key=lambda x: getattr(x, "utilization", 1e9))[: int(top_wasteful)]
            print(f"\nMost wasteful layers (lowest utilization, top {len(ls)}):")
            for i, layer in enumerate(ls, 1):
                name = getattr(layer, "name", "?")
                mtype = getattr(layer, "module_type", "?")
                r = getattr(layer, "r", None)
                sr = getattr(layer, "stable_rank", None)
                util = getattr(layer, "utilization", None)
                e90 = getattr(layer, "energy_rank_90", None)
                print(
                    f"  {i:02d}. {mtype:>4} r={r:<3} util={_fmt(util, pct=True):>6}  sr={_fmt(sr):>5}  k@90%={_fmt(e90):>3}  {name}"
                )



def _print_qa_summary(artifact: Any) -> None:
    """Pretty-print a compact QA summary to the terminal."""
    sep = "\u2500" * 50

    # --- Adapter ---
    print("ADAPTER QA SUMMARY")
    print(sep)
    print(f"  Adapter:       {artifact.adapter_name}")
    print(f"  Path:          {artifact.adapter_path}")
    if artifact.base_model:
        print(f"  Base model:    {artifact.base_model}")
    print(f"  Rank:          {artifact.rank_nominal}")
    print(f"  Layers:        {artifact.n_layers}")

    # --- Structural ---
    print("\nSTRUCTURAL SUMMARY")
    print(sep)
    print(f"  Utilization (mean):    {artifact.utilization_mean:.3f}")
    print(f"  Utilization (median):  {artifact.utilization_median:.3f}")
    print(f"  Stable rank (mean):    {artifact.stable_rank_mean:.3f}")
    if artifact.energy_rank_90_p50 is not None:
        print(f"  Effective rank 90 (median):  {artifact.energy_rank_90_p50:.1f}")
    print(f"  Rank waste ratio:      {artifact.rank_waste_ratio:.3f}")
    if artifact.structural_flags:
        print(f"  Flags:                 {', '.join(artifact.structural_flags)}")
    else:
        print("  Flags:                 (none)")

    # --- Behavioral ---
    print("\nBEHAVIORAL SUMMARY")
    print(sep)
    if not artifact.eval_available:
        print("  Eval available: no")
        print("  Eligibility determined from structural evidence only")
    else:
        if artifact.eval_dataset:
            print(f"  Eval dataset:  {artifact.eval_dataset}")
        if artifact.lower_is_better is not None:
            direction = "lower is better" if artifact.lower_is_better else "higher is better"
        else:
            direction = ""
        if artifact.metric_name:
            print(f"  Metric:        {artifact.metric_name} ({direction})")
        print(f"  Adapter score: {artifact.adapter_score}")
        print(f"  Base score:    {artifact.base_score}")
        print(f"  Beats base:    {'yes' if artifact.beats_base else 'no'}")

    # --- Eligibility ---
    print("\nELIGIBILITY")
    print(sep)
    print(f"  Status:      {artifact.status.value.upper()}")
    print(f"  Confidence:  {artifact.confidence}")
    if artifact.reasons:
        print("  Reasons:")
        for reason in artifact.reasons:
            print(f"    - {reason}")
