"""N133 KnOTS / TSV proxy analysis (no new compute, option 3).

A faithful KnOTS and TSV head-to-head against mean_alignment and
inv_min_erank cannot be done from the local N133 artifacts alone,
because:

  - adapter_profiles.json stores per-layer singular VALUES only
    (16 σ's per layer × 128 layers × 12 adapters).
  - pair_alignment_full.json stores the per-layer SCALAR alignment
    number, which is Gradience's own SV-weighted subspace-overlap
    measurement — already computed from U/V, then discarded.
  - adapter .safetensors files (U, V) are on the Pod, not locally.

KnOTS (Stoica et al. 2024) and TSV (Gargiulo et al. 2024) both score
adapter pairs using joint-SVD / shared-subspace quantities that
*require* U/V matrices. Neither is computable from singular values
plus a scalar alignment.

What we CAN do, cheaply and honestly, is test whether alternative
aggregations of the per-layer alignment number — the quantities that
a KnOTS-style or TSV-style aggregation would be expressing over the
same per-layer subspace-overlap values — move the B-P5.b conclusion
at all. If max-pool, top-k-layers, O-module-only, etc. all fail the
family-residualized test, then the "try KnOTS/TSV" escape route is
known to be closed at least for the mean-alignment family of signals,
and a faithful test can be scheduled with narrower expectations.

This script therefore:

  1. Builds 10+ per-pair scores that differ only in how they
     aggregate the SAME 128 per-layer alignment values per pair.
  2. Runs each against max_degradation on the 12 evaluated
     cross-task pairs both raw and after residualizing against
     FAMILY_B (the six-family scheme from n133_family_residualized_baseline.py).
  3. Records whether any aggregation produces a partial Spearman
     ρ ≥ 0.50 and ΔR² ≥ 0.10 over the family baseline.

Inputs (all local):
  sidecar/data/n133/pod_pull/audits/pair_alignment_full.json
  sidecar/data/n133/pod_pull/audits/pair_alignment_summary.json
  sidecar/data/n133/pod_pull/audits/adapter_profiles.json
  sidecar/data/n133/pod_pull/merges/merge_eval_summary.json

Output:
  sidecar/data/n133/knots_tsv_proxy.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
POD = ROOT / "sidecar" / "data" / "n133" / "pod_pull"
OUT = ROOT / "sidecar" / "data" / "n133" / "knots_tsv_proxy.json"

N_LAYERS = 32

FAMILY_B = {
    "sst2": "sentiment",
    "mnli": "nli",
    "squad": "extractive_qa",
    "gsm8k": "math_gen",
    "code": "code_gen",
    "summarization": "nl_gen",
}


def load():
    profiles = json.loads((POD / "audits" / "adapter_profiles.json").read_text())
    pair_sum = json.loads((POD / "audits" / "pair_alignment_summary.json").read_text())
    pair_full = json.loads((POD / "audits" / "pair_alignment_full.json").read_text())
    merges = json.loads((POD / "merges" / "merge_eval_summary.json").read_text())
    return profiles, pair_sum, pair_full, merges


def max_degradation(r):
    ta, tb = r["task_a"], r["task_b"]
    return max(r.get(f"degradation_{ta}", 0.0), r.get(f"degradation_{tb}", 0.0))


# ---------------------------------------------------------------------------
# Per-pair alignment aggregations
# ---------------------------------------------------------------------------

def per_layer_table(pair):
    """Return a structured per-layer table: (layer_idx, module, alignment)."""
    return [(layer["layer"], layer["module"], layer["alignment"])
            for layer in pair["per_layer"]]


def agg_mean(entries):
    return float(np.mean([a for _, _, a in entries]))


def agg_max(entries):
    return float(np.max([a for _, _, a in entries]))


def agg_topk_mean(entries, k):
    vals = sorted([a for _, _, a in entries], reverse=True)
    return float(np.mean(vals[:k]))


def agg_module(entries, module):
    vals = [a for _, m, a in entries if m == module]
    return float(np.mean(vals)) if vals else float("nan")


def agg_module_max(entries, module):
    vals = [a for _, m, a in entries if m == module]
    return float(np.max(vals)) if vals else float("nan")


def agg_deep_layers(entries, depth_frac=0.5):
    """Mean over layers at or below the top `depth_frac` of the stack."""
    threshold = int((1 - depth_frac) * N_LAYERS)
    vals = [a for layer, _, a in entries if layer >= threshold]
    return float(np.mean(vals)) if vals else float("nan")


def agg_l2_norm(entries):
    """L2 norm of the per-layer alignment vector — the nearest proxy to a
    'subspace-energy' style aggregation from per-layer numbers alone."""
    vals = np.array([a for _, _, a in entries])
    return float(np.linalg.norm(vals) / np.sqrt(len(vals)))


def agg_l_inf(entries):
    return float(np.max(np.abs([a for _, _, a in entries])))


def agg_percentile(entries, q):
    return float(np.percentile([a for _, _, a in entries], q))


def agg_qk_mean(entries):
    vals = [a for _, m, a in entries if m in ("Q", "K")]
    return float(np.mean(vals))


def agg_vo_mean(entries):
    vals = [a for _, m, a in entries if m in ("V", "O")]
    return float(np.mean(vals))


def build_scores(pair_full):
    scores: dict[str, dict[str, float]] = {}
    for key, pair in pair_full.items():
        entries = per_layer_table(pair)
        scores[key] = {
            # Baseline: current Gradience metric
            "mean_alignment": agg_mean(entries),
            # Reduced aggregations on the SAME per-layer values
            "max_alignment": agg_max(entries),
            "p90_alignment": agg_percentile(entries, 90),
            "topk16_mean": agg_topk_mean(entries, 16),
            "topk32_mean": agg_topk_mean(entries, 32),
            "deep_half_mean": agg_deep_layers(entries, 0.5),
            "deep_quarter_mean": agg_deep_layers(entries, 0.25),
            "l2_alignment": agg_l2_norm(entries),
            "l_inf_alignment": agg_l_inf(entries),
            # Module-specific
            "O_mean": agg_module(entries, "O"),
            "O_max": agg_module_max(entries, "O"),
            "V_mean": agg_module(entries, "V"),
            "Q_mean": agg_module(entries, "Q"),
            "K_mean": agg_module(entries, "K"),
            "QK_mean": agg_qk_mean(entries),
            "VO_mean": agg_vo_mean(entries),
        }
    return scores


# ---------------------------------------------------------------------------
# Family residualization (shared with n133_family_residualized_baseline)
# ---------------------------------------------------------------------------

def family_pair_label(ta, tb, fam):
    return "|".join(sorted([fam[ta], fam[tb]]))


def dummy_encode(labels):
    uniq = sorted(set(labels))
    cols = uniq[1:]
    n = len(labels)
    X = np.ones((n, 1 + len(cols)))
    for j, c in enumerate(cols):
        for i, lab in enumerate(labels):
            X[i, 1 + j] = 1.0 if lab == c else 0.0
    return X


def ols_residuals(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return resid, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    profiles, pair_sum, pair_full, merges = load()
    scores_by_pair = build_scores(pair_full)

    eval_degs = {r["pair"]: max_degradation(r)
                 for r in merges["results"] if r["category"] != "same_task"}
    eval_pairs = sorted(eval_degs.keys())
    y = np.array([eval_degs[p] for p in eval_pairs])

    # Task-family-pair identity
    labels = [family_pair_label(pair_full[p]["task_a"], pair_full[p]["task_b"],
                                FAMILY_B) for p in eval_pairs]
    X_fam = dummy_encode(labels)
    resid_y, r2_family = ols_residuals(y, X_fam)

    print("=" * 78)
    print("  N133 KnOTS/TSV proxy — per-layer-alignment aggregation sweep")
    print("=" * 78)
    print()
    print("Context: a faithful KnOTS / TSV comparison needs U/V matrices, which")
    print("this local dataset does not contain. This script instead sweeps every")
    print("reasonable aggregation of the per-layer alignment values that were")
    print("saved, and asks whether any of them beats the family-identity baseline.")
    print()
    print(f"  n evaluated cross-task pairs: {len(eval_pairs)}")
    print(f"  FAMILY_B baseline (family-pair dummies only) R² = {r2_family:.4f}")
    print()

    variants = list(next(iter(scores_by_pair.values())).keys())
    rows = []
    for v in variants:
        x = np.array([scores_by_pair[p][v] for p in eval_pairs])
        if not np.all(np.isfinite(x)):
            rows.append({"variant": v, "skip": True})
            continue

        rho_raw, p_raw = stats.spearmanr(x, y)
        r_raw, pr_raw = stats.pearsonr(x, y)

        resid_x, _ = ols_residuals(x, X_fam)
        if resid_x.std() < 1e-10:
            rho_partial, p_partial = float("nan"), float("nan")
        else:
            rho_partial, p_partial = stats.spearmanr(resid_x, resid_y)

        X_full = np.hstack([X_fam, x.reshape(-1, 1)])
        _, r2_full = ols_residuals(y, X_full)
        delta_r2 = r2_full - r2_family

        passes = (not np.isnan(rho_partial)
                  and rho_partial >= 0.50
                  and delta_r2 >= 0.10)
        rows.append({
            "variant": v,
            "spearman_raw": float(rho_raw),
            "spearman_raw_p": float(p_raw),
            "spearman_partial": float(rho_partial),
            "spearman_partial_p": float(p_partial),
            "delta_r2": float(delta_r2),
            "passes_H1": bool(passes),
        })

    print(f"  {'variant':20s} {'ρ_raw':>8s} {'ρ_partial':>10s} "
          f"{'ΔR²':>8s}   {'H1':>6s}")
    for r in rows:
        if r.get("skip"):
            print(f"  {r['variant']:20s}  (skip: non-finite)")
            continue
        tag = "PASS" if r["passes_H1"] else "fail"
        print(f"  {r['variant']:20s} {r['spearman_raw']:+8.3f} "
              f"{r['spearman_partial']:+10.3f} {r['delta_r2']:+8.3f}   "
              f"{tag:>6s}")

    any_pass = any(r.get("passes_H1") for r in rows if not r.get("skip"))
    print()
    print(f"Any aggregation passes pre-registered N134 H1 rule "
          f"(ρ_partial ≥ 0.50 AND ΔR² ≥ 0.10)? {'YES' if any_pass else 'NO'}")
    if not any_pass:
        print()
        print("Interpretation: no aggregation of per-layer mean_alignment — "
              "over any")
        print("module subset, depth band, top-k selection, or L_p reduction — "
              "beats")
        print("the family-pair baseline on the 12 evaluated N133 pairs. Since "
              "KnOTS")
        print("and TSV are themselves (different) aggregations over per-layer "
              "/ per-")
        print("adapter subspace-overlap quantities, this is negative evidence "
              "that a")
        print("faithful KnOTS or TSV comparison on this specific N133 sample "
              "would")
        print("improve over mean_alignment under the same confound. It is "
              "NOT evidence")
        print("about KnOTS/TSV performance in general, and does not "
              "substitute for a")
        print("faithful comparison on an N134-style unconfounded task set.")

    # Save
    report = {
        "experiment": "N133 KnOTS/TSV proxy",
        "caveat": (
            "A faithful KnOTS/TSV comparison requires U/V matrices from the "
            "adapter .safetensors files, which are not in the local N133 "
            "artifact bundle. This script instead tests whether any alternative "
            "aggregation of the per-layer alignment scalars beats the "
            "family-pair baseline. It is negative evidence only; a faithful "
            "head-to-head is scheduled alongside N134."
        ),
        "family_scheme": "FAMILY_B",
        "family_R2": r2_family,
        "n_evaluated_cross_task_pairs": len(eval_pairs),
        "n_variants": len(rows),
        "variants": rows,
        "any_passes_H1": any_pass,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved {OUT}")


if __name__ == "__main__":
    main()
