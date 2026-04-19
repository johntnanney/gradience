"""Unit tests for 01_pilot_gate.py — run on synthetic JSONs.

Verifies:
  1. Accuracy-band check classifies over/under/in-band correctly
  2. Retry ladder emits correct r= / data_frac= / max_steps= per attempt
  3. After 2 retries, action flips to "replace"
  4. Dip test: unimodal synthetic data -> PASS; bimodal -> FAIL
  5. Exit codes: 0=PASS, 1=retry, 2=exhausted, 3=missing

Usage:
    python3 scripts/n134/test_pilot_gate.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Load the gate module by path (not as a package)
_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot_gate", _here / "01_pilot_gate.py")
gate = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(gate)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _write_evals(eval_root: Path, tasks_accuracies: dict[str, float], seed: int = 42) -> None:
    eval_root.mkdir(parents=True, exist_ok=True)
    for task, acc in tasks_accuracies.items():
        (eval_root / f"{task}_s{seed}_source_eval.json").write_text(
            json.dumps({
                "task": task,
                "seed": seed,
                "accuracy": acc,
                "correct": int(acc * 100),
                "total": 100,
            })
        )


def _ledger_with_attempts(task_attempts: dict[str, int]) -> dict:
    """Build a ledger with n attempts per task (contents arbitrary)."""
    tasks = {}
    for task, n in task_attempts.items():
        tasks[task] = {
            "attempts": [{"r": 16, "data_frac": 1.0, "accuracy": 0.50} for _ in range(n)]
        }
    return {"tasks": tasks, "reserve_drawn": []}


# ---------------------------------------------------------------------------
# Accuracy-band tests
# ---------------------------------------------------------------------------

def test_all_in_band_passes():
    """All 8 tasks in [0.70, 0.90] -> C1 passes, no failures."""
    eval_results = {f"t{i}": {"accuracy": 0.75 + 0.01 * i} for i in range(8)}
    ledger = {"tasks": {}, "reserve_drawn": []}
    failures, retries_avail = gate.check_accuracy_band(eval_results, ledger)
    assert failures == []
    assert retries_avail is False
    print("  PASS  test_all_in_band_passes")


def test_too_high_first_retry_is_r8():
    """acc > 0.90 with 0 prior attempts -> retry with r=8."""
    eval_results = {"arc": {"accuracy": 0.95}}
    ledger = {"tasks": {}, "reserve_drawn": []}
    failures, retries_avail = gate.check_accuracy_band(eval_results, ledger)
    assert len(failures) == 1
    assert retries_avail is True
    f = failures[0]
    assert f["violation"] == "above_upper_bound"
    assert f["action"] == "retry"
    assert f["retry_config"] == {"r": 8, "data_frac": 1.0, "max_steps": 1000}
    assert "--r 8" in f["command"]
    print("  PASS  test_too_high_first_retry_is_r8")


def test_too_high_second_retry_is_data_half():
    """acc > 0.90 with 1 prior attempt -> retry with data_frac=0.5."""
    eval_results = {"arc": {"accuracy": 0.92}}
    ledger = _ledger_with_attempts({"arc": 1})
    failures, retries_avail = gate.check_accuracy_band(eval_results, ledger)
    assert failures[0]["action"] == "retry"
    assert failures[0]["retry_config"]["data_frac"] == 0.5
    assert failures[0]["retry_config"]["r"] == 16
    print("  PASS  test_too_high_second_retry_is_data_half")


def test_too_low_first_retry_is_r32():
    """acc < 0.70 with 0 prior attempts -> retry with r=32."""
    eval_results = {"piqa": {"accuracy": 0.55}}
    ledger = {"tasks": {}, "reserve_drawn": []}
    failures, _ = gate.check_accuracy_band(eval_results, ledger)
    assert failures[0]["action"] == "retry"
    assert failures[0]["retry_config"] == {"r": 32, "data_frac": 1.0, "max_steps": 1000}
    assert "--r 32" in failures[0]["command"]
    print("  PASS  test_too_low_first_retry_is_r32")


def test_too_low_second_retry_doubles_steps():
    """acc < 0.70 with 1 prior attempt -> retry with max_steps=2000."""
    eval_results = {"piqa": {"accuracy": 0.60}}
    ledger = _ledger_with_attempts({"piqa": 1})
    failures, _ = gate.check_accuracy_band(eval_results, ledger)
    assert failures[0]["action"] == "retry"
    assert failures[0]["retry_config"] == {"r": 32, "data_frac": 1.0, "max_steps": 2000}
    print("  PASS  test_too_low_second_retry_doubles_steps")


def test_two_retries_exhausted_replaces():
    """After 2 failed retries -> action=replace, retries_avail=False."""
    eval_results = {"piqa": {"accuracy": 0.60}}
    ledger = _ledger_with_attempts({"piqa": 2})
    failures, retries_avail = gate.check_accuracy_band(eval_results, ledger)
    assert failures[0]["action"] == "replace"
    assert failures[0]["retry_config"] is None
    assert retries_avail is False
    assert "reserve" in failures[0]["command"].lower()
    print("  PASS  test_two_retries_exhausted_replaces")


def test_mixed_band_some_retry_some_replace():
    """One task with retries left + one exhausted -> both reported correctly."""
    eval_results = {
        "arc": {"accuracy": 0.95},       # too high, 0 attempts -> retry
        "piqa": {"accuracy": 0.55},      # too low, 2 attempts  -> replace
        "boolq": {"accuracy": 0.80},     # in-band                -> no failure
    }
    ledger = _ledger_with_attempts({"piqa": 2})
    failures, retries_avail = gate.check_accuracy_band(eval_results, ledger)
    by_task = {f["task"]: f for f in failures}
    assert set(by_task.keys()) == {"arc", "piqa"}
    assert by_task["arc"]["action"] == "retry"
    assert by_task["piqa"]["action"] == "replace"
    assert retries_avail is True  # arc can still retry
    print("  PASS  test_mixed_band_some_retry_some_replace")


# ---------------------------------------------------------------------------
# Dip-test tests
# ---------------------------------------------------------------------------

def test_dip_test_unimodal_passes():
    """Unimodal data (8 values from one cluster) -> p > 0.10 (PASS)."""
    # Simulate N133-like erank values clustered around 8.5
    rng = np.random.default_rng(seed=7)
    data = rng.normal(8.5, 0.5, size=8)
    dip, p, source = gate.hartigans_dip_test(data)
    assert p > 0.10, f"unimodal data should pass dip test, got p={p:.3f} ({source})"
    print(f"  PASS  test_dip_test_unimodal_passes  (p={p:.3f}, source={source})")


def test_dip_test_bimodal_rejects():
    """Clearly bimodal data (4 low + 4 high) -> small p (FAIL)."""
    # N133 failure mode: discriminative tasks erank ~5, generative ~13
    data = np.array([5.0, 5.2, 5.1, 5.3, 13.0, 13.2, 13.1, 12.9])
    dip, p, source = gate.hartigans_dip_test(data)
    # With the fallback gap test, p should be small for this obvious gap
    assert p < 0.20, f"bimodal data should have small p, got p={p:.3f} ({source})"
    print(f"  PASS  test_dip_test_bimodal_rejects  (p={p:.3f}, source={source})")


def test_dip_test_degenerate():
    """All values equal -> degenerate case, no crash."""
    data = np.array([8.0] * 8)
    dip, p, source = gate.hartigans_dip_test(data)
    assert dip == 0.0 or p >= 1.0 - 1e-9
    print(f"  PASS  test_dip_test_degenerate  (source={source})")


# ---------------------------------------------------------------------------
# End-to-end tests using run_gate()
# ---------------------------------------------------------------------------

def test_run_gate_all_pass_exit_0():
    """Happy path: 8 tasks in-band, unimodal erank -> exit 0."""
    with tempfile.TemporaryDirectory() as tmp:
        eval_root = Path(tmp) / "evals"
        pilot_root = Path(tmp) / "pilot"
        ledger = Path(tmp) / "ledger.json"

        tasks = [f"t{i}" for i in range(8)]
        _write_evals(eval_root, {t: 0.75 + 0.01 * i for i, t in enumerate(tasks)})
        # No adapter dirs -> erank values empty -> insufficient_data path
        # So we fake the erank by monkeypatching compute_adapter_mean_erank

        orig_fn = gate.compute_adapter_mean_erank
        rng = np.random.default_rng(seed=11)
        fake_eranks = rng.normal(8.5, 0.3, size=8)

        def fake_compute(adapter_dir):
            task = adapter_dir.name.split("_s")[0]
            return float(fake_eranks[tasks.index(task)])

        # Also monkeypatch the adapter_model.safetensors existence check
        for t in tasks:
            d = pilot_root / f"{t}_s42"
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_model.safetensors").write_bytes(b"")  # empty file, exists()=True

        gate.compute_adapter_mean_erank = fake_compute
        try:
            report = gate.run_gate(
                eval_root=eval_root,
                pilot_root=pilot_root,
                ledger_path=ledger,
                tasks=tasks,
            )
        finally:
            gate.compute_adapter_mean_erank = orig_fn

        assert report["gate_decision"] == "PASS", report
        assert report["exit_code"] == 0
    print("  PASS  test_run_gate_all_pass_exit_0")


def test_run_gate_missing_evals_exit_3():
    """Missing eval files -> exit 3 (MISSING_DATA)."""
    with tempfile.TemporaryDirectory() as tmp:
        eval_root = Path(tmp) / "evals"
        pilot_root = Path(tmp) / "pilot"
        ledger = Path(tmp) / "ledger.json"
        eval_root.mkdir()

        tasks = [f"t{i}" for i in range(8)]
        _write_evals(eval_root, {tasks[0]: 0.80})  # only 1 of 8

        report = gate.run_gate(
            eval_root=eval_root, pilot_root=pilot_root, ledger_path=ledger, tasks=tasks,
        )
        assert report["gate_decision"] == "MISSING_DATA"
        assert report["exit_code"] == 3
        assert len(report["missing_tasks"]) == 7
    print("  PASS  test_run_gate_missing_evals_exit_3")


def test_run_gate_c1_fail_exit_1():
    """One out-of-band task with retries available -> exit 1."""
    with tempfile.TemporaryDirectory() as tmp:
        eval_root = Path(tmp) / "evals"
        pilot_root = Path(tmp) / "pilot"
        ledger = Path(tmp) / "ledger.json"

        tasks = [f"t{i}" for i in range(8)]
        accs = {t: 0.80 for t in tasks}
        accs["t0"] = 0.95  # above upper bound
        _write_evals(eval_root, accs)
        for t in tasks:
            d = pilot_root / f"{t}_s42"
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_model.safetensors").write_bytes(b"")

        orig_fn = gate.compute_adapter_mean_erank
        rng = np.random.default_rng(seed=11)
        fake_eranks = rng.normal(8.5, 0.3, size=8)

        def fake_compute(adapter_dir):
            task = adapter_dir.name.split("_s")[0]
            return float(fake_eranks[tasks.index(task)])

        gate.compute_adapter_mean_erank = fake_compute
        try:
            report = gate.run_gate(
                eval_root=eval_root, pilot_root=pilot_root, ledger_path=ledger, tasks=tasks,
            )
        finally:
            gate.compute_adapter_mean_erank = orig_fn

        assert report["gate_decision"] == "FAIL_RETRY_AVAILABLE", report
        assert report["exit_code"] == 1
        assert len(report["c1_accuracy_band"]["failures"]) == 1
        assert report["c1_accuracy_band"]["failures"][0]["action"] == "retry"
    print("  PASS  test_run_gate_c1_fail_exit_1")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_all_in_band_passes,
        test_too_high_first_retry_is_r8,
        test_too_high_second_retry_is_data_half,
        test_too_low_first_retry_is_r32,
        test_too_low_second_retry_doubles_steps,
        test_two_retries_exhausted_replaces,
        test_mixed_band_some_retry_some_replace,
        test_dip_test_unimodal_passes,
        test_dip_test_bimodal_rejects,
        test_dip_test_degenerate,
        test_run_gate_all_pass_exit_0,
        test_run_gate_missing_evals_exit_3,
        test_run_gate_c1_fail_exit_1,
    ]
    print(f"Running {len(tests)} pilot-gate tests...")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
        sys.exit(0)
    else:
        print(f"{failed}/{len(tests)} tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
