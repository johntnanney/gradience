"""N134 Phase 0b: Pilot gate validation (v2).

Checks §3.2 pilot-gate criteria on the 8 pilot adapters:

    C1 (accuracy band):  every adapter in [0.70, 0.90] on its validation set
    Erank continuity:    Hartigan's dip test p > 0.10 on 8-point distribution

On C1 failure, emits concrete retry commands per §3.2's retry ladder.
The retry ladder is tracked per-task in /workspace/n134/pilot/retry_ledger.json
so this script is idempotent across retry rounds.

Retry ladder (per task, max 2 attempts before replacement):
    acc too HIGH (>0.90, saturated): attempt_1 uses r=8, attempt_2 uses data_frac=0.5
    acc too LOW  (<0.70, underfit):  attempt_1 uses r=32, attempt_2 uses data_frac=2x steps

After 2 failed retries: task is flagged for replacement from the reserve
list (LogiQA, ANLI-R1, Cosmos QA; §3.2).

Exit codes:
    0 = gate passed (proceed to Phase 1)
    1 = gate failed, retries still available (emit retry commands)
    2 = gate failed, retries exhausted (emit replacement plan)
    3 = insufficient data (missing eval files)

Usage:
    python3 scripts/n134/01_pilot_gate.py
    python3 scripts/n134/01_pilot_gate.py --smoke
    python3 scripts/n134/01_pilot_gate.py --emit-commands  # print shell commands

Output:
    /workspace/n134/pilot/pilot_gate.json
    /workspace/n134/pilot/retry_ledger.json (state)
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
LEDGER_FILE = Path("/workspace/n134/pilot/retry_ledger.json")

TASKS = ["arc", "hellaswag", "winogrande", "openbookqa", "commonsenseqa", "piqa", "siqa", "boolq"]
RESERVE_TASKS = ["logiqa", "anli_r1", "cosmos_qa"]  # §3.2 reserve list
SEED = 42

# Gate thresholds (§3.2)
ACC_LOWER = 0.70
ACC_UPPER = 0.90
DIP_P_THRESHOLD = 0.10
MAX_RETRIES = 2  # after 2 failed retries, replace from reserve


# ---------------------------------------------------------------------------
# Spectral helpers (minimal; Phase 2 will recompute with full v2.1 schema)
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
    """QR-based fast SVD for low-rank delta_W = scaling * B @ A.

    For (d_out, r) @ (r, d_in) LoRA factors this is O(d*r^2) rather than
    O(d^2*r). Returns (U, S, Vt) with rank-r factors.
    """
    B_scaled = scaling * B
    Q_B, R_B = np.linalg.qr(B_scaled, mode="reduced")
    M = R_B @ A
    U_small, S, Vt = np.linalg.svd(M, full_matrices=False)
    U = Q_B @ U_small
    return U, S, Vt


def load_adapter_weights(adapter_dir: Path) -> dict[str, dict]:
    """Load LoRA A/B matrices from adapter safetensors."""
    weights: dict[str, dict] = {}
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
                layer_keys.setdefault(layer_name, {})["A"] = f.get_tensor(k).astype(np.float64)
            elif "lora_B" in k:
                layer_name = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if layer_name.startswith(prefix):
                        layer_name = layer_name[len(prefix):]
                layer_keys.setdefault(layer_name, {})["B"] = f.get_tensor(k).astype(np.float64)

        for layer_name, matrices in layer_keys.items():
            if "A" in matrices and "B" in matrices:
                weights[layer_name] = {"A": matrices["A"], "B": matrices["B"], "scaling": scaling}

    return weights


def compute_adapter_mean_erank(adapter_dir: Path) -> float:
    """Mean entropy effective rank across all LoRA layers in the adapter."""
    weights = load_adapter_weights(adapter_dir)
    eranks = []
    for layer_name in sorted(weights.keys()):
        lw = weights[layer_name]
        _U, S, _Vt = fast_svd_delta_W(lw["A"], lw["B"], lw["scaling"])
        eranks.append(entropy_effective_rank(S))
    return float(np.mean(eranks)) if eranks else 0.0


# ---------------------------------------------------------------------------
# Hartigan's dip test (real implementation via `diptest`, safe fallback)
# ---------------------------------------------------------------------------

def hartigans_dip_test(data: np.ndarray) -> tuple[float, float, str]:
    """Hartigan's dip test. Returns (dip_statistic, p_value, source).

    Uses the ``diptest`` package (wraps the original Hartigan & Hartigan 1985
    Fortran implementation). The N134 spec §3.2 commits to Hartigan's dip;
    substituting a different test (e.g. gap-based bootstrap, k-means
    silhouette) would be a silent deviation, so this function raises
    ImportError if ``diptest`` is not installed rather than falling back
    to a not-Hartigan test.

    Install with:  pip install diptest
    """
    # Handle degenerate edge cases first (same behavior regardless of library)
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n < 4:
        return 0.0, 1.0, "insufficient_n"
    if arr.max() - arr.min() < 1e-12:
        return 0.0, 1.0, "degenerate"

    try:
        import diptest
    except ImportError as exc:
        raise ImportError(
            "diptest is required for the N134 pilot gate — §3.2 commits to "
            "Hartigan's dip test and we will not substitute a different test. "
            "Install with:  pip install diptest"
        ) from exc

    dip, p_value = diptest.diptest(arr)
    return float(dip), float(p_value), "diptest"


# ---------------------------------------------------------------------------
# Accuracy-band check + retry ladder
# ---------------------------------------------------------------------------

def _next_retry_config(task: str, current_accuracy: float, ledger: dict) -> dict:
    """Decide the next retry configuration for a failing task per §3.2.

    Returns a dict with keys: action, config, command (or None if exhausted).
    action: "retry" | "replace"
    config: {"r": int, "data_frac": float, "max_steps": int} for retries
    command: shell command to run, or None
    """
    task_state = ledger.get("tasks", {}).get(task, {"attempts": []})
    n_attempts = len(task_state.get("attempts", []))

    # Original attempt is attempt 0 (the base r=16 / full-data run).
    # Retries are attempts 1 and 2. Replacement after attempt 2.
    #
    # Ladder depends on failure direction:
    #
    #   acc > UPPER (0.90, saturated):
    #     retry 1: r=8   (halve capacity)        -> attempt index 1
    #     retry 2: r=16 + data_frac=0.5          -> attempt index 2
    #
    #   acc < LOWER (0.70, underfit):
    #     retry 1: r=32  (double capacity)       -> attempt index 1
    #     retry 2: r=32 + max_steps=2000 (2x)    -> attempt index 2
    too_high = current_accuracy > ACC_UPPER
    too_low = current_accuracy < ACC_LOWER

    if n_attempts >= MAX_RETRIES:
        return {
            "action": "replace",
            "config": None,
            "command": (
                f"# {task}: retries exhausted (2 attempts), flag for reserve-list replacement\n"
                f"# Reserve list (§3.2): {', '.join(RESERVE_TASKS)}\n"
                f"# Manual: edit TASKS in scripts/n134/00_pilot_train.py "
                f"to replace '{task}' with a reserve task, then re-run."
            ),
        }

    if too_high:
        if n_attempts == 0:
            config = {"r": 8, "data_frac": 1.0, "max_steps": 1000}
            rationale = "acc saturated — halve LoRA capacity (r=16 -> r=8)"
        else:  # n_attempts == 1
            config = {"r": 16, "data_frac": 0.5, "max_steps": 1000}
            rationale = "acc still saturated — reduce training data to 50%"
    elif too_low:
        if n_attempts == 0:
            config = {"r": 32, "data_frac": 1.0, "max_steps": 1000}
            rationale = "acc underfit — double LoRA capacity (r=16 -> r=32)"
        else:  # n_attempts == 1
            config = {"r": 32, "data_frac": 1.0, "max_steps": 2000}
            rationale = "acc still underfit — r=32 + double training steps"
    else:
        # Shouldn't reach here if caller filtered to failures.
        return {"action": "none", "config": None, "command": None}

    cmd = (
        f"# {task}: retry attempt {n_attempts + 1} -- {rationale}\n"
        f"python3 scripts/n134/00_pilot_train.py --task {task} "
        f"--r {config['r']} --data-frac {config['data_frac']} "
        f"--max-steps {config['max_steps']} --retry-attempt {n_attempts + 1}"
    )
    return {"action": "retry", "config": config, "command": cmd}


def check_accuracy_band(eval_results: dict[str, dict], ledger: dict) -> tuple[list[dict], bool]:
    """Check C1: every adapter accuracy in [ACC_LOWER, ACC_UPPER].

    Returns (failures, any_retries_available).
    Each failure carries a concrete next-step command.
    """
    failures: list[dict] = []
    any_retries_available = False

    for task, result in eval_results.items():
        acc = float(result["accuracy"])
        if ACC_LOWER <= acc <= ACC_UPPER:
            continue

        violation = "above_upper_bound" if acc > ACC_UPPER else "below_lower_bound"
        bound = ACC_UPPER if acc > ACC_UPPER else ACC_LOWER
        plan = _next_retry_config(task, acc, ledger)
        if plan["action"] == "retry":
            any_retries_available = True

        failures.append({
            "task": task,
            "accuracy": acc,
            "violation": violation,
            "bound": bound,
            "retries_done": len(ledger.get("tasks", {}).get(task, {}).get("attempts", [])),
            "action": plan["action"],
            "retry_config": plan["config"],
            "command": plan["command"],
        })

    return failures, any_retries_available


def check_erank_continuity(erank_values: dict[str, float]) -> dict:
    """Check erank continuity via Hartigan's dip test (p > DIP_P_THRESHOLD)."""
    values = np.array(list(erank_values.values()))
    dip_stat, p_value, source = hartigans_dip_test(values)
    passed = p_value > DIP_P_THRESHOLD

    return {
        "dip_statistic": dip_stat,
        "p_value": p_value,
        "threshold": DIP_P_THRESHOLD,
        "passed": passed,
        "n_points": len(values),
        "test_source": source,
        "interpretation": (
            "erank distribution is consistent with unimodality (PASS)"
            if passed
            else "erank distribution shows significant multimodality (FAIL) — "
                 "adapters may have bimodal spectral profiles"
        ),
    }


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def load_ledger(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"tasks": {}, "reserve_drawn": []}


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gate(
    eval_root: Path = EVAL_ROOT,
    pilot_root: Path = PILOT_ROOT,
    ledger_path: Path = LEDGER_FILE,
    tasks: list[str] | None = None,
    seed: int = SEED,
) -> dict:
    """Core gate logic. Extracted for unit-testing on synthetic inputs."""
    if tasks is None:
        tasks = TASKS

    # 1) Load eval results
    eval_results: dict[str, dict] = {}
    missing_tasks = []
    for task in tasks:
        eval_file = eval_root / f"{task}_s{seed}_source_eval.json"
        if eval_file.exists():
            eval_results[task] = json.loads(eval_file.read_text())
        else:
            missing_tasks.append(task)

    if missing_tasks:
        return {
            "gate_decision": "MISSING_DATA",
            "missing_tasks": missing_tasks,
            "exit_code": 3,
        }

    # 2) Load retry ledger
    ledger = load_ledger(ledger_path)

    # 3) C1: accuracy band
    acc_failures, retries_available = check_accuracy_band(eval_results, ledger)
    c1_passed = len(acc_failures) == 0

    # 4) Erank continuity (always compute; gate doesn't depend on C1 first)
    erank_values: dict[str, float] = {}
    for task in tasks:
        adapter_dir = pilot_root / f"{task}_s{seed}"
        if (adapter_dir / "adapter_model.safetensors").exists():
            erank_values[task] = compute_adapter_mean_erank(adapter_dir)

    if len(erank_values) >= 4:
        erank_result = check_erank_continuity(erank_values)
    else:
        erank_result = {
            "dip_statistic": None,
            "p_value": None,
            "threshold": DIP_P_THRESHOLD,
            "passed": False,
            "n_points": len(erank_values),
            "test_source": "insufficient_data",
            "interpretation": "fewer than 4 erank values available",
        }
    erank_passed = erank_result["passed"]

    # 5) Gate decision
    gate_passed = c1_passed and erank_passed
    has_replacements = any(f["action"] == "replace" for f in acc_failures)

    if gate_passed:
        decision = "PASS"
        exit_code = 0
    elif has_replacements or not retries_available:
        decision = "FAIL_EXHAUSTED"
        exit_code = 2
    else:
        decision = "FAIL_RETRY_AVAILABLE"
        exit_code = 1

    report = {
        "gate_decision": decision,
        "exit_code": exit_code,
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
        "retry_ledger": ledger,
        "reserve_tasks": RESERVE_TASKS,
        "tasks_evaluated": tasks,
        "seed": seed,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="N134 Phase 0b: Pilot gate validation")
    parser.add_argument("--smoke", action="store_true", help="Accept smoke-mode eval files")
    parser.add_argument("--emit-commands", action="store_true",
                        help="Print retry commands to stdout (useful for piping to bash)")
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    print("=" * 60)
    print("  N134 Pilot Gate Validation (v2)")
    print("=" * 60)

    report = run_gate(
        eval_root=args.eval_root,
        pilot_root=args.pilot_root,
        ledger_path=args.ledger,
    )

    # Pretty-print
    if report["gate_decision"] == "MISSING_DATA":
        print(f"\n  ERROR: Missing eval results for {report['missing_tasks']}")
        print("  Run 00_pilot_train.py first.")
        sys.exit(3)

    acc_info = report["c1_accuracy_band"]
    print(f"\n  C1 (accuracy in [{ACC_LOWER}, {ACC_UPPER}]): "
          f"{'PASS' if acc_info['passed'] else 'FAIL'}")
    for task, acc in acc_info["per_task_accuracy"].items():
        marker = " " if ACC_LOWER <= acc <= ACC_UPPER else "!"
        print(f"    {marker} {task}: {acc:.3f}")

    if acc_info["failures"]:
        print("\n  Failures:")
        for f in acc_info["failures"]:
            print(f"    - {f['task']}: {f['accuracy']:.3f} ({f['violation']}, "
                  f"retries_done={f['retries_done']}, action={f['action']})")

    erank_info = report["erank_continuity"]
    print(f"\n  Erank continuity (dip p > {DIP_P_THRESHOLD}): "
          f"{'PASS' if erank_info['passed'] else 'FAIL'}")
    dt = erank_info["dip_test"]
    print(f"    source:    {dt['test_source']}")
    print(f"    dip stat:  {dt['dip_statistic']}")
    print(f"    p-value:   {dt['p_value']}")
    print(f"    n_points:  {dt['n_points']}")
    print(f"    interp:    {dt['interpretation']}")

    print(f"\n{'=' * 60}")
    print(f"  GATE DECISION: {report['gate_decision']}")
    print(f"{'=' * 60}")

    if args.emit_commands and acc_info["failures"]:
        print("\n# ===== Retry commands =====")
        for f in acc_info["failures"]:
            if f["command"]:
                print(f["command"])
                print()

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report written to {args.output}")

    sys.exit(report["exit_code"])


if __name__ == "__main__":
    main()
