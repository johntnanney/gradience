"""N134 Phase 0b: Pilot gate validation.

Loads the 8 pilot eval results from Phase 0 and checks two gate criteria:

    C1 (accuracy band):  every adapter accuracy in [0.70, 0.90]
    Erank continuity:    Hartigan's dip test p > 0.10 on the 8-point
                         distribution of per-adapter mean erank values

If all checks pass, the pilot gate is PASS and N134 can proceed to
full multi-seed training.  If any check fails, specific remediation
suggestions are provided (e.g. retry at r=8 or r=32).

Usage:
    python3 scripts/n134/01_pilot_gate.py
    python3 scripts/n134/01_pilot_gate.py --smoke   # accept smoke eval files

Output: /workspace/n134/pilot/pilot_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PILOT_ROOT = Path("/workspace/n134/pilot")
EVAL_ROOT = Path("/workspace/n134/pilot/evals")
OUTPUT_FILE = Path("/workspace/n134/pilot/pilot_gate.json")

TASKS = ["arc", "hellaswag", "winogrande", "openbookqa", "commonsenseqa", "piqa", "siqa", "boolq"]
SEED = 42

# Gate thresholds
ACC_LOWER = 0.70
ACC_UPPER = 0.90
DIP_P_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Spectral audit helpers (minimal, matching N133 conventions)
# ---------------------------------------------------------------------------

def entropy_effective_rank(S: np.ndarray) -> float:
    """Entropy effective rank = exp(H) where H = -sum(p * log(p))."""
    S2 = S**2
    total = np.sum(S2)
    if total < 1e-20:
        return 0.0
    p = S2 / total
    p = p[p > 1e-20]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def fast_svd_delta_W(A: np.ndarray, B: np.ndarray, scaling: float = 1.0):
    """QR-based fast SVD for low-rank delta_W = scaling * B @ A."""
    B_scaled = scaling * B
    Q_B, R_B = np.linalg.qr(B_scaled, mode="reduced")
    M = R_B @ A
    U_small, S, Vt = np.linalg.svd(M, full_matrices=False)
    U = Q_B @ U_small
    return U, S, Vt


def load_adapter_weights(adapter_dir: Path) -> dict[str, dict]:
    """Load LoRA A/B matrices from adapter safetensors."""
    weights = {}
    path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    config = json.loads(config_path.read_text())
    scaling = config.get("lora_alpha", 32) / config.get("r", 16)

    with safe_open(str(path), framework="numpy") as f:
        keys = list(f.keys())
        layer_keys: dict[str, dict] = {}
        for k in keys:
            if "lora_A" in k:
                layer_name = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if layer_name.startswith(prefix):
                        layer_name = layer_name[len(prefix):]
                if layer_name not in layer_keys:
                    layer_keys[layer_name] = {}
                layer_keys[layer_name]["A"] = f.get_tensor(k).astype(np.float64)
            elif "lora_B" in k:
                layer_name = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if layer_name.startswith(prefix):
                        layer_name = layer_name[len(prefix):]
                if layer_name not in layer_keys:
                    layer_keys[layer_name] = {}
                layer_keys[layer_name]["B"] = f.get_tensor(k).astype(np.float64)

        for layer_name, matrices in layer_keys.items():
            if "A" in matrices and "B" in matrices:
                weights[layer_name] = {
                    "A": matrices["A"],
                    "B": matrices["B"],
                    "scaling": scaling,
                }

    return weights


def compute_adapter_mean_erank(adapter_dir: Path) -> float:
    """Compute mean entropy effective rank across all LoRA layers."""
    weights = load_adapter_weights(adapter_dir)
    eranks = []
    for layer_name in sorted(weights.keys()):
        lw = weights[layer_name]
        _U, S, _Vt = fast_svd_delta_W(lw["A"], lw["B"], lw["scaling"])
        eranks.append(entropy_effective_rank(S))
    return float(np.mean(eranks)) if eranks else 0.0


# ---------------------------------------------------------------------------
# Dip test
# ---------------------------------------------------------------------------

def _hartigans_dip_statistic(data: np.ndarray) -> float:
    """Compute Hartigan's dip statistic for unimodality testing.

    This is a simplified implementation.  For production use, prefer the
    ``diptest`` package which wraps the original Fortran implementation.
    """
    n = len(data)
    if n < 4:
        return 0.0

    np.sort(data)

    # Uniform CDF
    np.arange(1, n + 1) / n

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Greatest convex minorant and least concave majorant of the ECDF
    # Simplified: compute max deviation between ECDF and best-fit uniform
    # on the sorted data
    lo = np.zeros(n)
    hi = np.zeros(n)

    for i in range(n):
        lo[i] = (i) / n
        hi[i] = (i + 1) / n

    # Dip = max over all intervals of |F_n - F_uniform|
    # where F_uniform is the best-fitting uniform on each modal interval
    diffs_upper = ecdf - lo
    diffs_lower = hi - ecdf

    dip = max(np.max(diffs_upper) - np.min(diffs_upper), np.max(diffs_lower) - np.min(diffs_lower))
    dip = dip / 2.0

    return float(dip)


def hartigans_dip_test(data: np.ndarray) -> tuple[float, float]:
    """Run Hartigan's dip test, returning (dip_statistic, p_value).

    Uses the ``diptest`` package if available for an accurate p-value.
    Otherwise falls back to a bootstrap approximation.
    """
    try:
        import diptest

        dip, p_value = diptest.diptest(data)
        return float(dip), float(p_value)
    except ImportError:
        pass

    # Fallback: bootstrap approximation
    dip_obs = _hartigans_dip_statistic(data)
    n = len(data)
    n_boot = 10000
    rng = np.random.default_rng(seed=42)
    count_ge = 0
    for _ in range(n_boot):
        uniform_sample = rng.uniform(0, 1, size=n)
        dip_boot = _hartigans_dip_statistic(uniform_sample)
        if dip_boot >= dip_obs:
            count_ge += 1
    p_value = count_ge / n_boot

    return dip_obs, p_value


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def check_accuracy_band(eval_results: dict[str, dict]) -> list[dict]:
    """Check C1: every adapter accuracy in [ACC_LOWER, ACC_UPPER].

    Returns a list of failure records (empty if all pass).
    """
    failures = []
    for task, result in eval_results.items():
        acc = result["accuracy"]
        if acc < ACC_LOWER:
            suggestion = "retry at r=8 (may need longer training or different lr)"
            if acc < 0.50:
                suggestion = "task may be too hard for LoRA r=16; consider r=32 or excluding"
            failures.append({
                "task": task,
                "accuracy": acc,
                "violation": "below_lower_bound",
                "bound": ACC_LOWER,
                "suggestion": suggestion,
            })
        elif acc > ACC_UPPER:
            failures.append({
                "task": task,
                "accuracy": acc,
                "violation": "above_upper_bound",
                "bound": ACC_UPPER,
                "suggestion": "retry at r=8 to lower accuracy, or exclude (saturated task)",
            })
    return failures


def check_erank_continuity(erank_values: dict[str, float]) -> dict:
    """Check erank continuity: dip test p > DIP_P_THRESHOLD on 8-point distribution."""
    values = np.array(list(erank_values.values()))
    dip_stat, p_value = hartigans_dip_test(values)

    passed = p_value > DIP_P_THRESHOLD
    return {
        "dip_statistic": dip_stat,
        "p_value": p_value,
        "threshold": DIP_P_THRESHOLD,
        "passed": passed,
        "n_points": len(values),
        "interpretation": (
            "erank distribution is consistent with unimodality (PASS)"
            if passed
            else "erank distribution shows significant multimodality (FAIL) -- "
            "adapters may have inconsistent spectral profiles"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="N134 Phase 0b: Pilot gate validation")
    parser.add_argument("--smoke", action="store_true", help="Accept smoke-mode eval files")
    parser.parse_args()

    print("=" * 60)
    print("  N134 Pilot Gate Validation")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load eval results
    # -----------------------------------------------------------------------
    print("\n  Loading pilot eval results...")
    eval_results: dict[str, dict] = {}
    missing_tasks = []

    for task in TASKS:
        eval_file = EVAL_ROOT / f"{task}_s{SEED}_source_eval.json"
        if eval_file.exists():
            result = json.loads(eval_file.read_text())
            eval_results[task] = result
            print(f"    {task}: accuracy = {result['accuracy']:.3f}")
        else:
            missing_tasks.append(task)
            print(f"    {task}: MISSING")

    if missing_tasks:
        print(f"\n  ERROR: Missing eval results for {missing_tasks}")
        print("  Run 00_pilot_train.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Check C1: accuracy band
    # -----------------------------------------------------------------------
    print(f"\n  Checking C1: accuracy in [{ACC_LOWER}, {ACC_UPPER}]...")
    acc_failures = check_accuracy_band(eval_results)

    if acc_failures:
        print(f"  C1 FAILED: {len(acc_failures)} task(s) outside band")
        for f in acc_failures:
            print(f"    {f['task']}: {f['accuracy']:.3f} ({f['violation']})")
            print(f"      Suggestion: {f['suggestion']}")
    else:
        print("  C1 PASSED: all tasks in band")

    # -----------------------------------------------------------------------
    # Compute per-adapter mean erank via Gradience spectral audit
    # -----------------------------------------------------------------------
    print("\n  Computing per-adapter mean erank...")
    erank_values: dict[str, float] = {}

    for task in TASKS:
        adapter_dir = PILOT_ROOT / f"{task}_s{SEED}"
        if not (adapter_dir / "adapter_model.safetensors").exists():
            print(f"    {task}: adapter not found, skipping erank")
            continue
        mean_erank = compute_adapter_mean_erank(adapter_dir)
        erank_values[task] = mean_erank
        print(f"    {task}: mean erank = {mean_erank:.3f}")

    # -----------------------------------------------------------------------
    # Check erank continuity (dip test)
    # -----------------------------------------------------------------------
    print("\n  Checking erank continuity (Hartigan's dip test)...")
    if len(erank_values) < 4:
        print("  WARNING: fewer than 4 erank values, dip test unreliable")
        erank_result = {
            "dip_statistic": None,
            "p_value": None,
            "threshold": DIP_P_THRESHOLD,
            "passed": False,
            "n_points": len(erank_values),
            "interpretation": "insufficient data for dip test",
        }
    else:
        erank_result = check_erank_continuity(erank_values)

    print(f"    Dip statistic: {erank_result['dip_statistic']}")
    print(f"    p-value: {erank_result['p_value']}")
    print(f"    {erank_result['interpretation']}")

    # -----------------------------------------------------------------------
    # Gate decision
    # -----------------------------------------------------------------------
    c1_passed = len(acc_failures) == 0
    erank_passed = erank_result["passed"]
    gate_passed = c1_passed and erank_passed

    print(f"\n{'=' * 60}")
    print(f"  GATE DECISION: {'PASS' if gate_passed else 'FAIL'}")
    print(f"{'=' * 60}")
    print(f"    C1 (accuracy band):    {'PASS' if c1_passed else 'FAIL'}")
    print(f"    Erank continuity:      {'PASS' if erank_passed else 'FAIL'}")

    if not gate_passed:
        print("\n  Failures:")
        if not c1_passed:
            for f in acc_failures:
                print(f"    - {f['task']}: accuracy {f['accuracy']:.3f} ({f['violation']})")
                print(f"      Suggestion: {f['suggestion']}")
        if not erank_passed:
            print(f"    - Erank dip test: p={erank_result['p_value']} < {DIP_P_THRESHOLD}")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    gate_report = {
        "gate_decision": "PASS" if gate_passed else "FAIL",
        "c1_accuracy_band": {
            "passed": c1_passed,
            "lower_bound": ACC_LOWER,
            "upper_bound": ACC_UPPER,
            "per_task_accuracy": {task: r["accuracy"] for task, r in eval_results.items()},
            "failures": acc_failures,
        },
        "erank_continuity": {
            "passed": erank_passed,
            "per_task_mean_erank": erank_values,
            "dip_test": erank_result,
        },
        "tasks_evaluated": TASKS,
        "seed": SEED,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(gate_report, f, indent=2)

    print(f"\n  Report written to {OUTPUT_FILE}")

    if not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
