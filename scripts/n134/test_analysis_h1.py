"""Dry-run verification of the H1 decision rule on fabricated data.

Validates that ``evaluate_h1_decision()`` in 06_analysis_h1.py correctly
applies the §4 H1 gate. Four scenarios are tested:

  1. NULL         -- S_H1 independent of max_degradation        -> NOT CONFIRMED
  2. CLEAR POS    -- max_deg = 0.5 * S_H1 + noise               -> CONFIRMED
  3. CONFOUND     -- max_deg = 0.8 * family_dummy + 0.1 * S_H1  -> NOT CONFIRMED
  4. WRONG SIGN   -- max_deg = -0.5 * S_H1 + noise              -> NOT CONFIRMED
                     (raw rho < 0 fails sign check)

If all four pass, the H1 decision rule is implemented correctly.
This is the committed reference test for the pre-registration.

Usage:
    python3 scripts/n134/test_analysis_h1.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load the analysis module by path (not a package)
_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis_h1", _here / "06_analysis_h1.py")
H1 = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(H1)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fabricated-pair generator
# ---------------------------------------------------------------------------

TASKS = ["arc_challenge", "hellaswag", "winogrande", "openbookqa",
         "commonsenseqa", "piqa", "siqa", "boolq"]
SEEDS = [42, 123, 456]
FAMILY_B = H1.FAMILY_B


def _family_pair_label(task_a: str, task_b: str) -> str:
    fa = FAMILY_B[task_a]
    fb = FAMILY_B[task_b]
    return "|".join(sorted([fa, fb]))


def fabricate_cross_task_pairs(n_pairs: int = 45, seed: int = 134) -> list[dict]:
    """Reproduce the deterministic sampling from 04_sample_pairs.py.

    Returns n_pairs cross-task pair dicts, each with keys:
        task_a, task_b, family_pair, pair_key
    """
    import random

    adapters = [(t, s) for t in TASKS for s in SEEDS]

    # Enumerate cross-task pairs, group by task-type cell
    cells: dict[tuple[str, str], list[dict]] = {}
    for i, (ta, sa) in enumerate(adapters):
        for tb, sb in adapters[i + 1:]:
            if ta == tb:
                continue
            cell = tuple(sorted([ta, tb]))
            pair = {
                "task_a": ta,
                "task_b": tb,
                "pair_key": f"{ta}_s{sa}__{tb}_s{sb}",
                "family_pair": _family_pair_label(ta, tb),
            }
            cells.setdefault(cell, []).append(pair)  # type: ignore[arg-type]

    rng = random.Random(seed)
    sampled: list[dict] = []
    # 1 per cell first (28 cells)
    sorted_cells = sorted(cells.keys())
    for cell in sorted_cells:
        candidates = cells[cell][:]
        rng.shuffle(candidates)
        sampled.append(candidates[0])

    # Fill remaining 17 by sampling cells without replacement, then pick a
    # new pair from each
    remaining = n_pairs - len(sampled)
    if remaining > 0:
        cell_pool = sorted_cells[:]
        rng.shuffle(cell_pool)
        extras = cell_pool[:remaining]
        for cell in extras:
            candidates = [p for p in cells[cell] if p not in sampled]
            if candidates:
                rng.shuffle(candidates)
                sampled.append(candidates[0])

    return sampled[:n_pairs]


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def scenario_null(pairs: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """S_H1 and max_deg independent (pure noise).

    Expected outcome: partial rho ~= 0, delta_R^2 ~= 0, H1 NOT CONFIRMED.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    scores = rng.uniform(0.02, 0.08, size=n)  # realistic S_H1 scale
    y = rng.uniform(0.0, 0.25, size=n)  # realistic max_degradation
    return scores, y


def scenario_clear_positive(pairs: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """max_deg = 0.5 * S_H1 + small noise.

    Standardize so the effect is large-signal. Expected: CONFIRMED.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    # Generate S_H1 with wide spread so ranking is informative
    scores = rng.uniform(0.02, 0.08, size=n)
    # Standardize S_H1, then max_deg = 0.5 * standardized + small noise
    # (scaled to realistic degradation units)
    s_std = (scores - scores.mean()) / (scores.std() + 1e-12)
    y = 0.10 + 0.05 * s_std + rng.normal(0, 0.015, size=n)
    y = np.clip(y, 0.0, 1.0)
    return scores, y


def scenario_confound(pairs: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """max_deg = 0.8 * family_dummy + 0.1 * S_H1 + noise.

    Family identity drives most variance; residual S_H1 signal is tiny.
    Expected: raw rho may look positive (because families covary), but
    partial rho << 0.50 and delta_R^2 < 0.10 -> NOT CONFIRMED.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    scores = rng.uniform(0.02, 0.08, size=n)
    s_std = (scores - scores.mean()) / (scores.std() + 1e-12)

    # Assign each family-pair cell a "severity" drawn from a wide distribution
    unique_families = sorted(set(p["family_pair"] for p in pairs))
    family_severity = {f: rng.normal(0, 0.08) for f in unique_families}

    # family component dominates (large variance contribution)
    fam_component = np.array([family_severity[p["family_pair"]] for p in pairs])
    # small S_H1 contribution (not enough to pass delta_R^2 >= 0.10)
    y = 0.10 + fam_component + 0.01 * s_std + rng.normal(0, 0.01, size=n)
    y = np.clip(y, 0.0, 1.0)
    return scores, y


def scenario_wrong_sign(pairs: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """max_deg = -0.5 * S_H1 + noise (negative correlation).

    Expected: |partial rho| may be large but sign check fails -> NOT CONFIRMED.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    scores = rng.uniform(0.02, 0.08, size=n)
    s_std = (scores - scores.mean()) / (scores.std() + 1e-12)
    y = 0.10 - 0.05 * s_std + rng.normal(0, 0.015, size=n)
    y = np.clip(y, 0.0, 1.0)
    return scores, y


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _format_decision(d: dict) -> str:
    return (
        f"rho_raw={d['rho_raw']:+.3f} (p={d['p_raw']:.3f})  "
        f"rho_partial={d['rho_partial']:+.3f} (p={d['p_partial']:.3f})  "
        f"dR2={d['delta_r2']:+.3f}  "
        f"crit1={d['crit1_pass']} crit2={d['crit2_pass']} sign={d['sign_correct']}  "
        f"-> {'CONFIRMED' if d['h1_confirmed'] else 'NOT CONFIRMED'}"
    )


def test_null(pairs, seed):
    """Null: scores and y independent. Expect NOT CONFIRMED."""
    scores, y = scenario_null(pairs, seed)
    cell_labels = [p["family_pair"] for p in pairs]
    d = H1.evaluate_h1_decision(scores, y, cell_labels, n_boot=500)
    print(f"  NULL           : {_format_decision(d)}")
    assert not d["h1_confirmed"], "null scenario wrongly CONFIRMED"
    # Rho should be within [-0.4, +0.4] — allow some noise
    assert abs(d["rho_partial"]) < 0.50, \
        f"null partial rho too large: {d['rho_partial']:+.3f}"
    assert d["delta_r2"] < 0.10, \
        f"null delta_R2 too large: {d['delta_r2']:+.3f}"
    return True


def test_clear_positive(pairs, seed):
    """Clear positive: strong S_H1 effect. Expect CONFIRMED."""
    scores, y = scenario_clear_positive(pairs, seed)
    cell_labels = [p["family_pair"] for p in pairs]
    d = H1.evaluate_h1_decision(scores, y, cell_labels, n_boot=500)
    print(f"  CLEAR POSITIVE : {_format_decision(d)}")
    assert d["h1_confirmed"], f"clear-positive wrongly NOT CONFIRMED: {_format_decision(d)}"
    assert d["sign_correct"], "sign should be correct for clear positive"
    assert d["rho_partial"] >= 0.50, \
        f"clear-positive partial rho below threshold: {d['rho_partial']:+.3f}"
    assert d["delta_r2"] >= 0.10, \
        f"clear-positive delta_R2 below threshold: {d['delta_r2']:+.3f}"
    return True


def test_confound(pairs, seed):
    """Confound: family dominates, small residual S_H1 effect. Expect NOT CONFIRMED."""
    scores, y = scenario_confound(pairs, seed)
    cell_labels = [p["family_pair"] for p in pairs]
    d = H1.evaluate_h1_decision(scores, y, cell_labels, n_boot=500)
    print(f"  CONFOUND       : {_format_decision(d)}")
    assert not d["h1_confirmed"], \
        f"confound wrongly CONFIRMED: {_format_decision(d)}"
    # Family-only R^2 should be substantial
    assert d["r2_base"] >= 0.50, \
        f"confound family-only R2 too small: {d['r2_base']:+.3f}"
    # Delta R^2 should NOT reach the threshold (S_H1 contributes little)
    assert d["delta_r2"] < 0.10, \
        f"confound delta_R2 reached threshold: {d['delta_r2']:+.3f}"
    return True


def test_wrong_sign(pairs, seed):
    """Wrong sign: negative S_H1 effect. Expect NOT CONFIRMED (sign check fails)."""
    scores, y = scenario_wrong_sign(pairs, seed)
    cell_labels = [p["family_pair"] for p in pairs]
    d = H1.evaluate_h1_decision(scores, y, cell_labels, n_boot=500)
    print(f"  WRONG SIGN     : {_format_decision(d)}")
    assert not d["h1_confirmed"], \
        f"wrong-sign wrongly CONFIRMED: {_format_decision(d)}"
    assert d["rho_raw"] < 0, "wrong-sign scenario should produce rho_raw < 0"
    assert not d["sign_correct"], "sign check should FAIL for wrong-sign scenario"
    # Magnitude should be large (it's a real effect, just wrong direction)
    assert abs(d["rho_raw"]) > 0.30, \
        f"wrong-sign effect too weak: |rho_raw|={abs(d['rho_raw']):.3f}"
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("  N134 H1 decision-rule dry-run on fabricated data")
    print("=" * 74)

    pairs = fabricate_cross_task_pairs(n_pairs=45, seed=134)
    print(f"\n  Fabricated {len(pairs)} cross-task pairs")
    print(f"  Family-pair cells: {len(set(p['family_pair'] for p in pairs))}")
    print()

    # Run all four scenarios with different seeds so results are comparable
    # across re-runs but not coupled within a single seed's sampling quirks.
    scenarios = [
        ("null",            test_null,           1001),
        ("clear_positive",  test_clear_positive, 1002),
        ("confound",        test_confound,       1003),
        ("wrong_sign",      test_wrong_sign,     1004),
    ]

    failed = 0
    for name, fn, seed in scenarios:
        try:
            fn(pairs, seed)
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("-" * 74)
    if failed == 0:
        print(f"  All {len(scenarios)} H1 dry-run scenarios passed.")
        print("  H1 decision rule is implemented correctly.")
        sys.exit(0)
    else:
        print(f"  {failed}/{len(scenarios)} H1 dry-run scenarios FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
