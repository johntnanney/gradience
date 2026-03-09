#!/usr/bin/env python3
"""
run_merge_audits.py — Run merge-audit on all configured adapter pairs.

Uses the Python API directly instead of shelling out to the CLI, so we
get structured MergeAuditReport objects and can measure timing precisely.

Usage:
    python scripts/merge_audit_test/run_merge_audits.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ADAPTER_DIR = Path("/workspace/merge_test/adapters")
RESULTS_DIR = Path("/workspace/merge_test/results")

# Define pairs: (label, adapter_a_name, adapter_b_name, hypothesis)
PAIRS = [
    (
        "pair1_code_vs_support",
        "magicoder",
        "customer_support",
        "LOW overlap: code generation vs customer support are fundamentally different tasks",
    ),
    # Pair 2: Summarization variants — not available on HuggingFace
    # (
    #     "pair2_summarization_variants",
    #     "tldr",
    #     "dialogsum",
    #     "MODERATE overlap: both learn text compression but from different input domains",
    # ),
    (
        "pair3_nlu_classification",
        "glue_mnli",
        "glue_qnli",
        "HIGH overlap: both learn entailment/contradiction features",
    ),
    (
        "pair4_chat_vs_math",
        "zephyr_sft",
        "gsm8k",
        "ROBUSTNESS: non-Predibase adapters, possibly different configs/base model variants",
    ),
]

# Threshold presets to test per pair
THRESHOLD_PRESETS = ["default", "conservative", "permissive"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_single_audit(label, adapter_a_dir, adapter_b_dir, output_dir, thresholds_name="default"):
    """Run one merge-audit and return (report, elapsed, error)."""
    from gradience.vnext.merge import VerdictThresholds, merge_audit

    if thresholds_name == "conservative":
        thresholds = VerdictThresholds.conservative()
    elif thresholds_name == "permissive":
        thresholds = VerdictThresholds.permissive()
    else:
        thresholds = VerdictThresholds()

    start = time.time()
    try:
        report = merge_audit(
            adapter_a_dir=adapter_a_dir,
            adapter_b_dir=adapter_b_dir,
            output_dir=output_dir,
            energy_threshold=0.90,
            thresholds=thresholds,
            compute_dtype="float64",
            verbose=True,
        )
        elapsed = time.time() - start
        return report, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return None, elapsed, e


def run_all_pairs():
    """Run merge-audit on all configured pairs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for label, name_a, name_b, hypothesis in PAIRS:
        path_a = ADAPTER_DIR / name_a
        path_b = ADAPTER_DIR / name_b

        print("\n" + "=" * 70)
        print(f"  {label}")
        print(f"  A: {name_a}  ({path_a})")
        print(f"  B: {name_b}  ({path_b})")
        print(f"  Hypothesis: {hypothesis}")
        print("=" * 70)

        if not path_a.exists():
            print(f"  SKIPPED — adapter A not found: {path_a}")
            results.append({"label": label, "status": "skipped", "reason": f"{name_a} not downloaded"})
            continue
        if not path_b.exists():
            print(f"  SKIPPED — adapter B not found: {path_b}")
            results.append({"label": label, "status": "skipped", "reason": f"{name_b} not downloaded"})
            continue

        # Run with default thresholds (writes artifacts)
        out_dir = RESULTS_DIR / label
        report, elapsed, error = run_single_audit(label, path_a, path_b, out_dir, thresholds_name="default")

        if error:
            print(f"\n  FAILED ({elapsed:.1f}s): {error}")
            traceback.print_exc()
            results.append(
                {
                    "label": label,
                    "status": "error",
                    "error": str(error),
                    "elapsed_s": elapsed,
                }
            )
            continue

        # Extract summary
        agg = report.aggregate
        n_layers = len(report.layer_verdicts)
        per_layer_time = elapsed / max(n_layers, 1)

        print(f"\n  DONE in {elapsed:.1f}s ({per_layer_time:.3f}s/layer)")
        print(f"  Verdict:   {agg['overall_verdict'].upper()}")
        print(f"  Score:     {agg['compatibility_score']:.3f}")
        print(f"  Overlap:   mean={agg['mean_overlap']:.3f}, max={agg['max_overlap']:.3f}")
        print(f"  Agreement: {agg['mean_agreement']:.3f}")
        print(
            f"  Verdicts:  {agg['n_safe']} safe, {agg['n_redundant']} redundant, "
            f"{agg['n_conflicting']} conflicting, {agg['n_imbalanced']} imbalanced"
        )
        print(f"  Artifacts: {out_dir}/merge_audit.{{json,md}}")

        results.append(
            {
                "label": label,
                "status": "ok",
                "elapsed_s": round(elapsed, 2),
                "per_layer_s": round(per_layer_time, 3),
                "n_layers": n_layers,
                "verdict": agg["overall_verdict"],
                "score": agg["compatibility_score"],
                "mean_overlap": agg["mean_overlap"],
                "max_overlap": agg["max_overlap"],
                "mean_agreement": agg["mean_agreement"],
                "n_safe": agg["n_safe"],
                "n_redundant": agg["n_redundant"],
                "n_conflicting": agg["n_conflicting"],
                "n_imbalanced": agg["n_imbalanced"],
                "hypothesis": hypothesis,
            }
        )

        # Also run with conservative thresholds (no artifact write, just compare)
        print("\n  Re-running with conservative thresholds...")
        report_c, elapsed_c, error_c = run_single_audit(label, path_a, path_b, None, thresholds_name="conservative")
        if report_c:
            agg_c = report_c.aggregate
            print(
                f"  Conservative: {agg_c['overall_verdict'].upper()} "
                f"(score={agg_c['compatibility_score']:.3f}, "
                f"safe={agg_c['n_safe']}, redundant={agg_c['n_redundant']}, "
                f"conflicting={agg_c['n_conflicting']}, imbalanced={agg_c['n_imbalanced']})"
            )

    return results


def write_summary(results):
    """Write summary JSON for downstream analysis."""
    summary_path = RESULTS_DIR / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pairs": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Merge-Audit Test Protocol")
    print("=" * 70)

    results = run_all_pairs()
    write_summary(results)

    # Quick summary table
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  {'Pair':<35s} {'Time':>6s} {'Layers':>6s} {'Overlap':>8s} {'Agree':>8s} {'Verdict':>12s}")
    print(f"  {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 12}")

    for r in results:
        if r["status"] == "ok":
            print(
                f"  {r['label']:<35s} {r['elapsed_s']:>5.1f}s {r['n_layers']:>6d} "
                f"{r['mean_overlap']:>8.3f} {r['mean_agreement']:>+8.3f} "
                f"{r['verdict']:>12s}"
            )
        elif r["status"] == "skipped":
            print(f"  {r['label']:<35s} {'—':>6s} {'—':>6s} {'—':>8s} {'—':>8s} {'SKIPPED':>12s}")
        else:
            print(f"  {r['label']:<35s} {r.get('elapsed_s', 0):>5.1f}s {'—':>6s} {'—':>8s} {'—':>8s} {'ERROR':>12s}")

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    err_count = sum(1 for r in results if r["status"] == "error")
    print(f"\n  {ok_count} succeeded, {skip_count} skipped, {err_count} failed")

    print("\nNext: python scripts/merge_audit_test/analyze_merge_results.py")

    sys.exit(1 if err_count > 0 else 0)
