"""Verdict computation for bench protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

from gradience.bench.task_profiles import get_task_profile_from_config


def classify_validation_level(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Classify validation level based on configuration.

    Returns validation classification and reasoning.
    """
    # Check for multi-seed
    compression = config.get("compression", {})
    seeds = compression.get("seeds", [])
    is_multiseed = len(seeds) > 1

    # Check training budget
    train_config = config.get("train", {})
    max_steps = train_config.get("max_steps", 0)

    # Determine validation level
    if is_multiseed and max_steps >= 200:
        if len(seeds) >= 3 and max_steps >= 500:
            classification = "certifiable"
            rationale = f"{len(seeds)} seeds × {max_steps} steps provides statistical rigor"
        else:
            classification = "screening_plus"
            rationale = f"{len(seeds)} seeds × {max_steps} steps (limited budget/seeds)"
    elif is_multiseed:
        classification = "screening_plus"
        rationale = f"{len(seeds)} seeds but only {max_steps} steps (limited budget)"
    elif max_steps >= 500:
        classification = "screening"
        rationale = f"Single seed, {max_steps} steps (no variance estimation)"
    else:
        classification = "screening"
        rationale = f"Single seed, {max_steps} steps (quick validation only)"

    return {
        "level": classification,
        "rationale": rationale,
        "is_multiseed": is_multiseed,
        "n_seeds": len(seeds) if is_multiseed else 1,
        "max_steps": max_steps
    }


def compute_verdicts(
    probe_results: Dict[str, Any],
    variant_results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path,
    smoke: bool = False
) -> Dict[str, Any]:
    """
    Step 3.6: Compute verdicts for compressed variants.

    Returns verdict analysis including PASS/FAIL decisions and best compression.
    """
    compression_config = config.get("compression", {})
    acc_tolerance = compression_config.get("acc_tolerance", 0.005)

    # Classify validation level
    validation_classification = classify_validation_level(config)

    # Get probe baseline
    probe_accuracy = probe_results["probe"]["accuracy"]
    probe_params = probe_results["probe"]["params"]

    # Probe quality gating using task profile
    task_profile = get_task_profile_from_config(config)

    # Load the original probe evaluation results for probe_gate
    probe_rank = config["lora"]["probe_r"]
    probe_eval_path = output_path / f"probe_r{probe_rank}" / "eval.json"
    with open(probe_eval_path, 'r') as f:
        probe_eval_results = json.load(f)

    probe_passed, gate_info = task_profile.probe_gate(probe_eval_results, config)
    probe_quality_threshold = gate_info["threshold"]

    if not probe_passed:
        print(f"\n=== PROBE QUALITY GATE FAILED ===")
        print(f"Probe accuracy: {probe_accuracy:.4f}")
        print(f"Required threshold: {probe_quality_threshold:.4f}")

        if smoke:
            print(f"Status: UNDERTRAINED_SMOKE - continuing in smoke mode")
            status_code = "UNDERTRAINED_SMOKE"
        else:
            print(f"Status: UNDERTRAINED - compression certification not valid")
            status_code = "UNDERTRAINED"

        # Return undertrained status for all variants
        verdicts = {}
        for variant_name in variant_results.keys():
            verdicts[variant_name] = {
                "status": "undertrained",
                "reason": f"Probe accuracy {probe_accuracy:.4f} < threshold {probe_quality_threshold:.4f}",
                "delta_vs_probe": None,
                "param_reduction": None,
                "verdict": status_code
            }

        return {
            "verdicts": verdicts,
            "probe_quality_status": status_code,
            "summary": {
                "probe_quality": "FAILED",
                "probe_accuracy": probe_accuracy,
                "probe_threshold": probe_quality_threshold,
                "recommendations_validated": "N/A",
                "best_compression": None,
                "notes": [f"Probe undertrained - compression results not reliable (smoke mode: {smoke})"]
            }
        }

    verdicts = {}
    pass_variants = []

    print(f"\n=== VERDICT ANALYSIS ===")
    print(f"🔬 Validation level: {validation_classification['level'].upper()}")
    print(f"   {validation_classification['rationale']}")
    print(f"✅ Probe quality: {probe_accuracy:.4f} ≥ {probe_quality_threshold:.4f}")
    print(f"Probe baseline: {probe_accuracy:.4f} accuracy, {probe_params:,} params")
    print(f"Accuracy tolerance: ±{acc_tolerance:.3f}")
    print()

    for variant_name, result in variant_results.items():
        if result["status"] != "completed":
            # Skip variants that didn't complete successfully
            status = result["status"]
            reason = result.get("reason", "Training not completed")

            verdicts[variant_name] = {
                "status": "skipped",
                "reason": reason,
                "delta_vs_probe": None,
                "param_reduction": None,
                "verdict": "FAIL" if status == "FAILED" else "SKIP"
            }

            if status == "FAILED":
                print(f"{variant_name}: FAIL - {reason}")
            else:
                print(f"{variant_name}: SKIP - {reason}")
            continue

        # Compute metrics
        compressed_accuracy = result["accuracy"]
        compressed_params = result["params"]

        delta_vs_probe = compressed_accuracy - probe_accuracy
        param_reduction = 1.0 - (compressed_params / probe_params)

        # Make verdict
        if delta_vs_probe >= -acc_tolerance:
            verdict = "PASS"
            pass_variants.append({
                "variant": variant_name,
                "param_reduction": param_reduction,
                "delta_vs_probe": delta_vs_probe,
                "compressed_params": compressed_params,
                "compressed_accuracy": compressed_accuracy
            })
        else:
            verdict = "FAIL"

        verdicts[variant_name] = {
            "status": "evaluated",
            "reason": None,
            "delta_vs_probe": delta_vs_probe,
            "param_reduction": param_reduction,
            "verdict": verdict,
            "compressed_accuracy": compressed_accuracy,
            "compressed_params": compressed_params,
            "probe_accuracy": probe_accuracy,
            "probe_params": probe_params
        }

        # Print verdict
        reduction_pct = param_reduction * 100
        print(f"{variant_name}: {verdict}")
        print(f"  Δ accuracy: {delta_vs_probe:+.4f} (threshold: {-acc_tolerance:.3f})")
        print(f"  Param reduction: {reduction_pct:.1f}% ({probe_params:,} → {compressed_params:,})")
        print(f"  Accuracy: {compressed_accuracy:.4f} vs {probe_accuracy:.4f}")
        print()

    # Find best compression among PASS variants
    best_compression = None
    if pass_variants:
        best_variant = max(pass_variants, key=lambda x: x["param_reduction"])
        best_compression = {
            "variant": best_variant["variant"],
            "param_reduction": best_variant["param_reduction"],
            "delta_vs_probe": best_variant["delta_vs_probe"],
            "compressed_params": best_variant["compressed_params"],
            "compressed_accuracy": best_variant["compressed_accuracy"]
        }

        reduction_pct = best_compression["param_reduction"] * 100
        print(f"🏆 BEST COMPRESSION: {best_compression['variant']}")
        print(f"   {reduction_pct:.1f}% parameter reduction with {best_compression['delta_vs_probe']:+.4f} accuracy delta")
    else:
        print("❌ NO PASSING VARIANTS: All compressions exceeded accuracy tolerance")

    return {
        "verdicts": verdicts,
        "best_compression": best_compression,
        "probe_baseline": {
            "accuracy": probe_accuracy,
            "params": probe_params
        },
        "acc_tolerance": acc_tolerance,
        "validation_classification": validation_classification,
        "summary": {
            "probe_quality": "PASSED",
            "probe_accuracy": probe_accuracy,
            "probe_threshold": probe_quality_threshold,
            "total_variants": len(variant_results),
            "completed": len([v for v in variant_results.values() if v["status"] == "completed"]),
            "passed": len([v for v in verdicts.values() if v["verdict"] == "PASS"]),
            "failed": len([v for v in verdicts.values() if v["verdict"] == "FAIL"]),
            "skipped": len([v for v in verdicts.values() if v["verdict"] == "SKIP"]),
            "notes": []
        }
    }
