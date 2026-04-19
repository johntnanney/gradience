"""N133 family-residualized partial-correlation baseline for N134.

Pre-registered analysis infrastructure for N134 (see sidecar/notes/n134_spec.md).
Runs every candidate score against max_degradation on the 12 evaluated N133
cross-task pairs with and without task-family controls, and records baseline
Spearman/Pearson numbers that N134 must beat on the new task set.

This script is the reference implementation of the N134 H1/H2/H3 methodology:
family-residualized Spearman of an arbitrary per-pair score against
max_degradation, plus ΔR² over a task-family-only OLS baseline.

The N133 task set is known to be confounded (6 tasks form an effectively
binary partition by metric range × erank; see B-P5 in FINDINGS.md §28), so
the numbers reported here are NOT taken as evidence for or against any
metric — they are the locked baseline N134 must clear.

Inputs (local):
  sidecar/data/n133/pod_pull/audits/adapter_profiles.json
  sidecar/data/n133/pod_pull/audits/pair_alignment_full.json
  sidecar/data/n133/pod_pull/audits/pair_alignment_summary.json
  sidecar/data/n133/pod_pull/merges/merge_eval_summary.json

Output:
  sidecar/data/n133/family_residualized_baseline.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
POD = ROOT / "sidecar" / "data" / "n133" / "pod_pull"
OUT = ROOT / "sidecar" / "data" / "n133" / "family_residualized_baseline.json"

N_LAYERS = 32


# Two candidate family assignments for sensitivity reporting.
# FAMILY_A: the binary partition that drives the B-P5 confound —
#   "generation-ceiling" (code/summarization/squad) vs
#   "discriminative-headroom" (mnli/sst2/gsm8k).
# FAMILY_B: finer four-way split (sentiment, NLI, extractive-QA, generation).
FAMILY_A = {
    "code": "gen_ceiling",
    "summarization": "gen_ceiling",
    "squad": "gen_ceiling",
    "mnli": "disc_headroom",
    "sst2": "disc_headroom",
    "gsm8k": "disc_headroom",
}

FAMILY_B = {
    "sst2": "sentiment",
    "mnli": "nli",
    "squad": "extractive_qa",
    "gsm8k": "math_gen",
    "code": "code_gen",
    "summarization": "nl_gen",
}


# ---------------------------------------------------------------------------
# Data loading + feature extraction
# ---------------------------------------------------------------------------

def load():
    profiles = json.loads((POD / "audits" / "adapter_profiles.json").read_text())
    pair_sum = json.loads((POD / "audits" / "pair_alignment_summary.json").read_text())
    pair_full = json.loads((POD / "audits" / "pair_alignment_full.json").read_text())
    merges = json.loads((POD / "merges" / "merge_eval_summary.json").read_text())
    return profiles, pair_sum, pair_full, merges


def adapter_erank(profiles):
    return {n: float(np.mean([layer["erank"] for layer in p["per_layer"]]))
            for n, p in profiles.items()}


def depth_weighted_module(pair_full_entry, module, style="linear"):
    entries = sorted(
        (
            (layer["layer"], layer["alignment"])
            for layer in pair_full_entry["per_layer"]
            if layer["module"] == module
        ),
        key=lambda x: x[0],
    )
    d = np.array([e[0] for e in entries], dtype=float)
    a = np.array([e[1] for e in entries], dtype=float)
    if style == "linear":
        w = d / (N_LAYERS - 1)
    elif style == "quadratic":
        w = (d / (N_LAYERS - 1)) ** 2
    else:
        w = np.ones_like(d)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w * a))


def module_mean(pair_full_entry, module):
    a = np.array([layer["alignment"] for layer in pair_full_entry["per_layer"]
                  if layer["module"] == module])
    return float(a.mean())


def build_features(profiles, pair_sum, pair_full):
    adapter_e = adapter_erank(profiles)
    feats = {}
    for key, pair in pair_full.items():
        a, b = pair["adapter_a"], pair["adapter_b"]
        ta, tb = pair["task_a"], pair["task_b"]
        feats[key] = {
            "task_a": ta,
            "task_b": tb,
            "is_same_task": pair["is_same_task"],
            "mean_alignment": pair_sum[key]["mean_alignment"],
            "min_erank": min(adapter_e[a], adapter_e[b]),
            "max_erank": max(adapter_e[a], adapter_e[b]),
            "mean_erank": 0.5 * (adapter_e[a] + adapter_e[b]),
            "inv_min_erank": 1.0 / min(adapter_e[a], adapter_e[b]),
            "O_mean": module_mean(pair, "O"),
            "O_depth": depth_weighted_module(pair, "O", "linear"),
            "O_quad": depth_weighted_module(pair, "O", "quadratic"),
            "V_depth": depth_weighted_module(pair, "V", "linear"),
            "OVmix_depth": 0.7 * depth_weighted_module(pair, "O", "linear")
                          + 0.3 * depth_weighted_module(pair, "V", "linear"),
        }
    return feats


def max_degradation(r):
    ta, tb = r["task_a"], r["task_b"]
    return max(r.get(f"degradation_{ta}", 0.0), r.get(f"degradation_{tb}", 0.0))


# ---------------------------------------------------------------------------
# Family residualization
# ---------------------------------------------------------------------------

def family_pair_label(ta: str, tb: str, family_map: dict[str, str]) -> str:
    """Order-invariant family-pair label."""
    fa, fb = family_map[ta], family_map[tb]
    return "|".join(sorted([fa, fb]))


def ols_residuals(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Return residuals and R^2 from y = X beta + eps (X already includes
    intercept)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return resid, r2


def dummy_encode(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return design matrix with intercept and one-hot (drop-one) encoding."""
    uniq = sorted(set(labels))
    # drop first level as reference
    ref = uniq[0]
    cols = uniq[1:]
    n = len(labels)
    X = np.ones((n, 1 + len(cols)))
    for j, c in enumerate(cols):
        for i, lab in enumerate(labels):
            X[i, 1 + j] = 1.0 if lab == c else 0.0
    return X, [f"intercept", *[f"fampair_{c}" for c in cols]]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(features, eval_degs, family_map, family_name):
    pairs = sorted(eval_degs.keys())
    fam_labels = [
        family_pair_label(features[p]["task_a"], features[p]["task_b"], family_map)
        for p in pairs
    ]
    y = np.array([eval_degs[p] for p in pairs])

    X_fam, fam_cols = dummy_encode(fam_labels)
    resid_y, r2_family = ols_residuals(y, X_fam)

    # Per-pair block (additive, for downstream figure generation).
    # Residualize each variant feature against the family dummies so the
    # figure script does not need to re-implement the OLS procedure.
    per_pair_residuals: dict[str, list[float]] = {}
    variant_names_for_residuals = [
        "mean_alignment", "inv_min_erank",
        "O_mean", "O_depth", "O_quad",
        "V_depth", "OVmix_depth",
    ]
    for v in variant_names_for_residuals:
        x = np.array([features[p][v] for p in pairs])
        rx, _ = ols_residuals(x, X_fam)
        per_pair_residuals[v] = rx.tolist()
    per_pair_block = []
    for i, p in enumerate(pairs):
        entry = {
            "pair": p,
            "task_a": features[p]["task_a"],
            "task_b": features[p]["task_b"],
            "family_pair": fam_labels[i],
            "max_deg": float(y[i]),
            "max_deg_resid": float(resid_y[i]),
        }
        for v in variant_names_for_residuals:
            entry[v] = float(features[p][v])
            entry[f"{v}_resid"] = float(per_pair_residuals[v][i])
        per_pair_block.append(entry)

    print(f"\n--- Family scheme: {family_name} ---")
    print(f"  families: {sorted(set(family_map.values()))}")
    unique_fampairs = sorted(set(fam_labels))
    print(f"  unique family-pair labels in eval set: {len(unique_fampairs)}")
    for fp in unique_fampairs:
        n = sum(1 for lab in fam_labels if lab == fp)
        print(f"    {fp:<40s} n={n}")
    print(f"  task-family-only baseline R^2 (y = family-pair dummies): {r2_family:.4f}")

    variants = [
        "mean_alignment", "inv_min_erank",
        "O_mean", "O_depth", "O_quad",
        "V_depth", "OVmix_depth",
    ]
    rows = []
    for v in variants:
        x = np.array([features[p][v] for p in pairs])

        rho_raw, p_raw = stats.spearmanr(x, y)
        r_raw, pr_raw = stats.pearsonr(x, y)

        # Family-residualized: residualize x against family dummies, then
        # correlate with residualized y. This is the partial correlation of
        # x and y controlling for family-pair identity.
        resid_x, _ = ols_residuals(x, X_fam)
        if resid_x.std() < 1e-10 or resid_y.std() < 1e-10:
            rho_partial, p_partial = float("nan"), float("nan")
            r_partial, pr_partial = float("nan"), float("nan")
        else:
            rho_partial, p_partial = stats.spearmanr(resid_x, resid_y)
            r_partial, pr_partial = stats.pearsonr(resid_x, resid_y)

        # Incremental R^2 (ΔR^2) from adding x to family-only baseline
        X_full = np.hstack([X_fam, x.reshape(-1, 1)])
        _, r2_full = ols_residuals(y, X_full)
        delta_r2 = r2_full - r2_family

        rows.append({
            "variant": v,
            "spearman_raw": float(rho_raw),
            "spearman_raw_p": float(p_raw),
            "pearson_raw": float(r_raw),
            "pearson_raw_p": float(pr_raw),
            "spearman_partial": float(rho_partial),
            "spearman_partial_p": float(p_partial),
            "pearson_partial": float(r_partial),
            "pearson_partial_p": float(pr_partial),
            "r2_family_only": float(r2_family),
            "r2_family_plus_x": float(r2_full),
            "delta_r2": float(delta_r2),
        })

    print(f"\n  {'variant':20s} {'ρ_raw':>8s} {'ρ_partial':>10s} "
          f"{'ΔR²':>8s} {'notes':>8s}")
    for r in rows:
        note = ""
        if np.isnan(r["spearman_partial"]):
            note = "collin"
        elif r["delta_r2"] < 0.01:
            note = "null"
        elif r["delta_r2"] >= 0.10:
            note = "PASSES"
        print(f"  {r['variant']:20s} {r['spearman_raw']:+8.3f} "
              f"{r['spearman_partial']:+10.3f} {r['delta_r2']:+8.3f} "
              f"{note:>8s}")
    return {
        "family_scheme": family_name,
        "family_map": family_map,
        "fampair_labels": fam_labels,
        "r2_family_only": r2_family,
        "variants": rows,
        "per_pair": per_pair_block,
    }


def main():
    print("=" * 72)
    print("  N133 family-residualized partial-correlation baseline (for N134)")
    print("=" * 72)

    profiles, pair_sum, pair_full, merges = load()
    features = build_features(profiles, pair_sum, pair_full)
    eval_degs = {r["pair"]: max_degradation(r)
                 for r in merges["results"] if r["category"] != "same_task"}
    print(f"\n  {len(eval_degs)} evaluated cross-task pairs")
    print(f"  6 tasks → C(6,2)=15 unordered task pairs; "
          f"actual evaluated = {len(eval_degs)}")

    family_A_result = analyze(features, eval_degs, FAMILY_A, "FAMILY_A (binary)")
    family_B_result = analyze(features, eval_degs, FAMILY_B, "FAMILY_B (four-way)")

    # The N134 primary decision rule (from n134_spec.md H1):
    #   - Primary score: O_depth
    #   - Requirement: Spearman ρ ≥ 0.50 with max_degradation AND
    #                  ΔR² ≥ 0.10 over a task-family baseline
    # Record the N133 value as pre-N134 baseline / negative control.
    print("\n" + "=" * 72)
    print("  Pre-registered N134 decision rule applied to N133 (negative control)")
    print("=" * 72)
    print("  H1 primary: O_depth must satisfy ρ_partial ≥ 0.50 AND ΔR² ≥ 0.10")
    for result in [family_A_result, family_B_result]:
        row = next(r for r in result["variants"] if r["variant"] == "O_depth")
        passes = (row["spearman_partial"] >= 0.50 and row["delta_r2"] >= 0.10)
        print(f"\n    {result['family_scheme']}:")
        print(f"      O_depth ρ_partial = {row['spearman_partial']:+.3f}   "
              f"ΔR² = {row['delta_r2']:+.3f}   →  {'PASSES' if passes else 'FAILS (expected)'}")

    # Save
    report = {
        "experiment": "N133 family-residualized partial-correlation baseline",
        "purpose": ("Pre-N134 baseline: the N133 sample is confounded, "
                    "these numbers are the level N134 must beat."),
        "n_evaluated_cross_task_pairs": len(eval_degs),
        "family_A_binary": family_A_result,
        "family_B_four_way": family_B_result,
        "n134_h1_decision_rule": {
            "primary_score": "O_depth",
            "thresholds": {"spearman_partial_min": 0.50, "delta_r2_min": 0.10},
            "n133_family_A": {
                "O_depth_spearman_partial": next(
                    r for r in family_A_result["variants"] if r["variant"] == "O_depth"
                )["spearman_partial"],
                "O_depth_delta_r2": next(
                    r for r in family_A_result["variants"] if r["variant"] == "O_depth"
                )["delta_r2"],
            },
            "n133_family_B": {
                "O_depth_spearman_partial": next(
                    r for r in family_B_result["variants"] if r["variant"] == "O_depth"
                )["spearman_partial"],
                "O_depth_delta_r2": next(
                    r for r in family_B_result["variants"] if r["variant"] == "O_depth"
                )["delta_r2"],
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved {OUT}")


if __name__ == "__main__":
    main()
