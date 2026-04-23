"""Primary H1 analysis + confirmatory replications (decoder-scale controlled triage).
# ANON: original title "N134 primary H1 analysis + confirmatory replications";
# ANON: project-number identifier stripped for double-blind review.

H1 metric: O-module depth-weighted alignment.

    S_H1(pair) = sum(w_l * alpha_O_l) / sum(w_l)  for l=1..32
    where alpha_O_l is SV-weighted alignment on O-projection of layer l,
    w_l = l/L (linear depth weight).

Decision rule (both must hold):
    1. Spearman partial rho(S_H1, max_degradation), residualized on
       FAMILY_B task-family-pair dummies, >= 0.50 with p < 0.05
    2. OLS delta-R2 of S_H1 over FAMILY_B dummy-only baseline >= 0.10
    3. Sign must be correct: higher alignment -> higher degradation

Confirmatory replications (non-gating):
    B-P1: 0/24 same-task pairs below midpoint, 0/252 cross-task above
    B-P2: Same/cross mean alignment ratio >= 2.0, Welch t p < 0.001
    B-P4: ANOVA of per-adapter mean erank across tasks, p < 1e-6

Bootstrap: 5000 resamples, block-bootstrap by family-pair cell.

Inputs:
    /workspace/n134/audit/pair_alignment_full.json
    /workspace/n134/audit/pair_alignment_summary.json
    /workspace/n134/audit/adapter_profiles.json
    /workspace/n134/pair_sample.json
    /workspace/n134/merges/merge_eval_summary.json

Outputs:
    /workspace/n134/analysis_h1.json
    /workspace/n134/figures/h1_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# WORKSPACE defaults to the pod-absolute path that produced the committed
# ANON: comment block rewritten to drop sidecar/ internal path example.
# bundle artifacts. Replicators running outside the original environment can
# override via the N134_WORKSPACE environment variable — pointing it at,
# e.g., the bundle's raw_analytical_artifacts/ directory or an isolated /tmp/ workspace — without
# editing the script or breaking its default pod-reproduction behavior.
# The default is preserved to keep the pod-era provenance accurate.
import os  # noqa: E402

WORKSPACE = Path(os.environ.get("N134_WORKSPACE", "/workspace/n134"))
AUDIT_DIR = WORKSPACE / "audit"
MERGE_DIR = WORKSPACE / "merges"
FIG_DIR = WORKSPACE / "figures"

N_LAYERS = 32
N_BOOTSTRAP = 5000
RNG_SEED = 20260418

# ---------------------------------------------------------------------------
# FAMILY_B partition
# ---------------------------------------------------------------------------

FAMILY_B: dict[str, str] = {
    "arc_challenge": "science_qa",
    "hellaswag": "commonsense_completion",
    "winogrande": "coreference_commonsense",
    "openbookqa": "knowledge_qa",
    "commonsenseqa": "commonsense_qa",
    "piqa": "physical_commonsense",
    "siqa": "social_commonsense",
    "boolq": "reading_comprehension",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def lookup_pair(pair_key: str, pair_full: dict) -> dict:
    """Return the pair_full entry for this key, tolerating reversed order.

    merge_eval stores pair keys in committed (pair_sample.json) order;
    pair_alignment_full keys are alphabetical ({min}_vs_{max}). This
    helper tries the requested key then the reversed form so downstream
    code can use merge-eval-style keys against the audit dictionary.
    """
    if pair_key in pair_full:
        return pair_full[pair_key]
    parts = pair_key.split("_vs_")
    if len(parts) == 2:
        reversed_key = f"{parts[1]}_vs_{parts[0]}"
        if reversed_key in pair_full:
            return pair_full[reversed_key]
    raise KeyError(f"Pair key not found in either order: {pair_key}")


def load_inputs() -> tuple[dict, dict, dict, dict, dict]:
    pair_full = json.loads((AUDIT_DIR / "pair_alignment_full.json").read_text())
    pair_summary = json.loads((AUDIT_DIR / "pair_alignment_summary.json").read_text())
    profiles = json.loads((AUDIT_DIR / "adapter_profiles.json").read_text())
    pair_sample = json.loads((WORKSPACE / "pair_sample.json").read_text())
    merges = json.loads((MERGE_DIR / "merge_eval_summary.json").read_text())
    return pair_full, pair_summary, profiles, pair_sample, merges


# ---------------------------------------------------------------------------
# H1 score computation
# ---------------------------------------------------------------------------

def compute_s_h1(pair: dict) -> float:
    """O-module depth-weighted alignment.

    S_H1 = sum(w_l * alpha_O_l) / sum(w_l)  for l in 0..31
    w_l = (l + 1) / N_LAYERS   (so layer 0 gets w=1/32, layer 31 gets w=1)
    """
    # pair_alignment_full.json stores per-layer entries with:
    #   - layer_idx (not "layer")
    #   - module as "q_proj" / "k_proj" / "v_proj" / "o_proj" (not "O")
    # Accept either naming convention for robustness.
    def _is_o(mod: str) -> bool:
        return mod == "O" or mod == "o_proj"

    def _layer_idx(layer: dict) -> int:
        if "layer_idx" in layer:
            return int(layer["layer_idx"])
        if "layer" in layer:
            return int(layer["layer"])
        raise KeyError(f"Layer entry missing layer_idx / layer: {layer}")

    o_entries = [
        (_layer_idx(layer), layer["alignment"])
        for layer in pair["per_layer"]
        if _is_o(layer["module"])
    ]
    o_entries.sort(key=lambda x: x[0])
    if not o_entries:
        return float("nan")

    depths = np.array([e[0] for e in o_entries], dtype=float)
    alphas = np.array([e[1] for e in o_entries], dtype=float)
    weights = (depths + 1.0) / N_LAYERS
    return float(np.sum(weights * alphas) / np.sum(weights))


# ---------------------------------------------------------------------------
# Merge degradation
# ---------------------------------------------------------------------------

def max_degradation(merge_entry: dict) -> float:
    ta, tb = merge_entry["task_a"], merge_entry["task_b"]
    degs = []
    for t in [ta, tb]:
        k = f"degradation_{t}"
        if k in merge_entry:
            degs.append(merge_entry[k])
    return max(degs) if degs else float("nan")


def build_eval_pairs(merges: dict) -> dict[str, float]:
    """Return {pair_key: max_degradation} for all evaluated cross-task pairs."""
    out = {}
    for r in merges["results"]:
        if r.get("category") == "same_task":
            continue
        deg = max_degradation(r)
        if np.isfinite(deg):
            out[r["pair"]] = deg
    return out


# ---------------------------------------------------------------------------
# Family-pair dummies and OLS helpers
# ---------------------------------------------------------------------------

def family_pair_label(task_a: str, task_b: str) -> str:
    fa = FAMILY_B.get(task_a, task_a)
    fb = FAMILY_B.get(task_b, task_b)
    return "|".join(sorted([fa, fb]))


def dummy_encode(labels: list[str]) -> np.ndarray:
    """Intercept + K-1 dummy columns for K unique labels."""
    uniq = sorted(set(labels))
    cols = uniq[1:]  # drop first for identification
    n = len(labels)
    X = np.ones((n, 1 + len(cols)))
    for j, c in enumerate(cols):
        for i, lab in enumerate(labels):
            X[i, 1 + j] = 1.0 if lab == c else 0.0
    return X


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


# ---------------------------------------------------------------------------
# Partial Spearman correlation
# ---------------------------------------------------------------------------

def partial_spearman(x: np.ndarray, y: np.ndarray, X_cov: np.ndarray) -> tuple[float, float]:
    """Partial Spearman rho: residualize x and y on X_cov, then Spearman."""
    resid_x = ols_residuals(x, X_cov)
    resid_y = ols_residuals(y, X_cov)
    if resid_x.std() < 1e-10 or resid_y.std() < 1e-10:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(resid_x, resid_y)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Tied-pair detection
# ---------------------------------------------------------------------------

def detect_tied_pairs(scores: np.ndarray, tol: float = 1e-6) -> dict:
    """Detect tied-score clusters."""
    n = len(scores)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and abs(scores[i] - scores[j]) < tol:
                cluster.append(j)
                visited[j] = True
        if len(cluster) > 1:
            clusters.append(cluster)
    return {
        "n_tied_clusters": len(clusters),
        "tied_cluster_sizes": [len(c) for c in clusters],
        "n_tied_pairs": sum(len(c) for c in clusters),
    }


# ---------------------------------------------------------------------------
# Block bootstrap by family-pair cell
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    scores: np.ndarray,
    y: np.ndarray,
    cell_labels: list[str],
    X_fam: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """Block-bootstrap CIs for partial Spearman rho and delta-R2."""
    rng = np.random.default_rng(seed)
    unique_cells = sorted(set(cell_labels))
    cell_indices = {c: [i for i, lab in enumerate(cell_labels) if lab == c]
                    for c in unique_cells}

    rho_boots = []
    dr2_boots = []

    for _ in range(n_boot):
        # Resample cells with replacement
        sampled_cells = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        idx = []
        for c in sampled_cells:
            idx.extend(cell_indices[c])
        idx = np.array(idx)

        if len(idx) < 4:
            continue

        x_b = scores[idx]
        y_b = y[idx]
        X_b = X_fam[idx]

        # Partial Spearman
        rho_b, _ = partial_spearman(x_b, y_b, X_b)

        # Delta R2
        r2_base = ols_r2(y_b, X_b)
        X_full = np.hstack([X_b, x_b.reshape(-1, 1)])
        r2_full = ols_r2(y_b, X_full)
        dr2_b = r2_full - r2_base

        if np.isfinite(rho_b) and np.isfinite(dr2_b):
            rho_boots.append(rho_b)
            dr2_boots.append(dr2_b)

    rho_boots = np.array(rho_boots)
    dr2_boots = np.array(dr2_boots)

    return {
        "n_valid_boots": len(rho_boots),
        "rho_mean": float(rho_boots.mean()) if len(rho_boots) else float("nan"),
        "rho_ci_025": float(np.percentile(rho_boots, 2.5)) if len(rho_boots) else float("nan"),
        "rho_ci_975": float(np.percentile(rho_boots, 97.5)) if len(rho_boots) else float("nan"),
        "dr2_mean": float(dr2_boots.mean()) if len(dr2_boots) else float("nan"),
        "dr2_ci_025": float(np.percentile(dr2_boots, 2.5)) if len(dr2_boots) else float("nan"),
        "dr2_ci_975": float(np.percentile(dr2_boots, 97.5)) if len(dr2_boots) else float("nan"),
    }


# ---------------------------------------------------------------------------
# H1 decision rule (extracted for testing)
# ---------------------------------------------------------------------------


def evaluate_h1_decision(
    scores: np.ndarray,
    y: np.ndarray,
    cell_labels: list[str],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """Apply §4 H1 decision rule and return a structured decision dict.

    This is the single source of truth for the H1 gate. Both the live
    pipeline and the dry-run test harness call this function.

    Parameters
    ----------
    scores : (n,) array of S_H1 values, one per evaluated cross-task pair
    y      : (n,) array of max_degradation values
    cell_labels : list of family-pair labels, length n
    n_boot : number of block-bootstrap resamples
    seed   : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        rho_raw, p_raw, rho_partial, p_partial, r2_base, r2_full, delta_r2,
        crit1_pass, crit2_pass, sign_correct, h1_confirmed, bootstrap, X_fam_shape
    """
    X_fam = dummy_encode(cell_labels)

    # Raw Spearman
    rho_raw, p_raw = stats.spearmanr(scores, y)
    rho_raw = float(rho_raw)
    p_raw = float(p_raw)

    # Criterion 1: partial Spearman ρ on family-residuals >= 0.50, p < 0.05
    rho_partial, p_partial = partial_spearman(scores, y, X_fam)
    crit1_pass = bool(
        np.isfinite(rho_partial) and rho_partial >= 0.50 and p_partial < 0.05
    )

    # Criterion 2: Δ R² >= 0.10
    r2_base = ols_r2(y, X_fam)
    X_full = np.hstack([X_fam, scores.reshape(-1, 1)])
    r2_full = ols_r2(y, X_full)
    delta_r2 = r2_full - r2_base
    crit2_pass = bool(delta_r2 >= 0.10)

    # Criterion 3: sign check (higher alignment -> higher degradation)
    sign_correct = bool(np.isfinite(rho_raw) and rho_raw > 0)

    h1_confirmed = crit1_pass and crit2_pass and sign_correct

    boot = block_bootstrap_ci(scores, y, cell_labels, X_fam, n_boot=n_boot, seed=seed)

    return {
        "rho_raw": rho_raw,
        "p_raw": p_raw,
        "rho_partial": rho_partial,
        "p_partial": p_partial,
        "r2_base": r2_base,
        "r2_full": r2_full,
        "delta_r2": delta_r2,
        "crit1_pass": crit1_pass,
        "crit2_pass": crit2_pass,
        "sign_correct": sign_correct,
        "h1_confirmed": h1_confirmed,
        "bootstrap": boot,
        "n_eval": int(len(scores)),
        "n_family_pair_cells": int(len(set(cell_labels))),
    }


# ---------------------------------------------------------------------------
# Confirmatory replications (non-gating)
# ---------------------------------------------------------------------------

def replication_bp1(pair_full: dict, pair_sample: dict) -> dict:
    """B-P1: 0/24 same-task pairs below midpoint, 0/252 cross-task above.

    Midpoint = midpoint between the minimum same-task and maximum cross-task
    mean_alignment values.
    """
    same_aligns = []
    cross_aligns = []
    for _key, pair in pair_full.items():
        is_same = pair.get("is_same_task", False)
        mean_align = pair.get("mean_alignment", float("nan"))
        if not np.isfinite(mean_align):
            continue
        if is_same:
            same_aligns.append(mean_align)
        else:
            cross_aligns.append(mean_align)

    if not same_aligns or not cross_aligns:
        return {"status": "insufficient_data", "n_same": len(same_aligns), "n_cross": len(cross_aligns)}

    same_arr = np.array(same_aligns)
    cross_arr = np.array(cross_aligns)
    midpoint = 0.5 * (same_arr.min() + cross_arr.max())

    same_below = int((same_arr < midpoint).sum())
    cross_above = int((cross_arr > midpoint).sum())

    confirmed = same_below == 0 and cross_above == 0

    return {
        "n_same": len(same_aligns),
        "n_cross": len(cross_aligns),
        "midpoint": float(midpoint),
        "same_below_midpoint": same_below,
        "cross_above_midpoint": cross_above,
        "same_mean": float(same_arr.mean()),
        "cross_mean": float(cross_arr.mean()),
        "confirmed": confirmed,
    }


def replication_bp2(pair_full: dict) -> dict:
    """B-P2: same/cross mean alignment ratio >= 2.0, Welch t p < 0.001."""
    same_aligns = []
    cross_aligns = []
    for pair in pair_full.values():
        is_same = pair.get("is_same_task", False)
        mean_align = pair.get("mean_alignment", float("nan"))
        if not np.isfinite(mean_align):
            continue
        if is_same:
            same_aligns.append(mean_align)
        else:
            cross_aligns.append(mean_align)

    if len(same_aligns) < 2 or len(cross_aligns) < 2:
        return {"status": "insufficient_data"}

    same_mean = float(np.mean(same_aligns))
    cross_mean = float(np.mean(cross_aligns))
    ratio = same_mean / (cross_mean + 1e-12)
    t_stat, p_val = stats.ttest_ind(same_aligns, cross_aligns, equal_var=False)

    confirmed = ratio >= 2.0 and p_val < 0.001

    return {
        "same_mean": same_mean,
        "cross_mean": cross_mean,
        "ratio": float(ratio),
        "welch_t": float(t_stat),
        "welch_p": float(p_val),
        "n_same": len(same_aligns),
        "n_cross": len(cross_aligns),
        "confirmed": confirmed,
    }


def replication_bp4(profiles: dict) -> dict:
    """B-P4: ANOVA of per-adapter mean erank across tasks, p < 1e-6."""
    # Group adapters by task
    task_eranks: dict[str, list[float]] = {}
    for name, profile in profiles.items():
        task = profile.get("task", name.split("_seed")[0])
        mean_erank = float(np.mean([layer["erank"] for layer in profile["per_layer"]]))
        task_eranks.setdefault(task, []).append(mean_erank)

    if len(task_eranks) < 2:
        return {"status": "insufficient_data"}

    groups = [np.array(v) for v in task_eranks.values()]
    f_stat, p_val = stats.f_oneway(*groups)

    confirmed = p_val < 1e-6

    return {
        "n_tasks": len(task_eranks),
        "n_adapters_per_task": {k: len(v) for k, v in task_eranks.items()},
        "task_means": {k: float(np.mean(v)) for k, v in task_eranks.items()},
        "anova_F": float(f_stat),
        "anova_p": float(p_val),
        "confirmed": confirmed,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_h1_scatter(
    scores: np.ndarray,
    y: np.ndarray,
    pairs: list[str],
    pair_full: dict,
    rho_partial: float,
    delta_r2: float,
    out_path: Path,
) -> None:
    """S_H1 vs max_degradation scatter, colored by family pair."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Color by family pair (use symmetric lookup to tolerate reversed keys)
    labels = [family_pair_label(lookup_pair(p, pair_full)["task_a"], lookup_pair(p, pair_full)["task_b"]) for p in pairs]
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    color_map = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    colors = [color_map[lab] for lab in labels]

    ax.scatter(scores, y * 100, c=colors, s=80, edgecolors="white", linewidth=0.7, zorder=3)
    ax.axhline(10, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="10% threshold")

    ax.set_xlabel("S_H1 (O-module depth-weighted alignment)", fontsize=11)
    ax.set_ylabel("max degradation (%)", fontsize=11)
    ax.set_title(
        f"H1: S_H1 vs merge degradation\n"  # ANON: project-number prefix stripped from figure title.
        f"partial rho = {rho_partial:+.3f}, delta-R2 = {delta_r2:+.3f}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    # Legend for family pairs (if manageable)
    if len(unique_labels) <= 15:
        for lab in unique_labels:
            ax.scatter([], [], c=[color_map[lab]], label=lab, s=60)
        ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bootstrap_distribution(boot_results: dict, out_path: Path) -> None:
    """Bootstrap distributions for partial rho and delta-R2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Partial Spearman rho
    ax = axes[0]
    rho_mean = boot_results["rho_mean"]
    rho_lo = boot_results["rho_ci_025"]
    rho_hi = boot_results["rho_ci_975"]
    ax.axvline(rho_mean, color="blue", linewidth=2, label=f"mean = {rho_mean:.3f}")
    ax.axvline(rho_lo, color="blue", linestyle="--", linewidth=1, label=f"2.5% = {rho_lo:.3f}")
    ax.axvline(rho_hi, color="blue", linestyle="--", linewidth=1, label=f"97.5% = {rho_hi:.3f}")
    ax.axvline(0.50, color="red", linestyle=":", linewidth=2, label="H1 threshold (0.50)")
    ax.set_xlabel("Partial Spearman rho", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Bootstrap: partial Spearman rho", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Delta R2
    ax = axes[1]
    dr2_mean = boot_results["dr2_mean"]
    dr2_lo = boot_results["dr2_ci_025"]
    dr2_hi = boot_results["dr2_ci_975"]
    ax.axvline(dr2_mean, color="green", linewidth=2, label=f"mean = {dr2_mean:.3f}")
    ax.axvline(dr2_lo, color="green", linestyle="--", linewidth=1, label=f"2.5% = {dr2_lo:.3f}")
    ax.axvline(dr2_hi, color="green", linestyle="--", linewidth=1, label=f"97.5% = {dr2_hi:.3f}")
    ax.axvline(0.10, color="red", linestyle=":", linewidth=2, label="H1 threshold (0.10)")
    ax.set_xlabel("Delta R2 over family baseline", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Bootstrap: delta-R2", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"H1 block-bootstrap CIs ({boot_results['n_valid_boots']} resamples)", fontsize=13)  # ANON: project-number prefix stripped.
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_replications(bp1: dict, bp2: dict, bp4: dict, out_path: Path) -> None:
    """Summary figure for confirmatory replications."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # B-P1: same vs cross alignment separation
    ax = axes[0]
    if bp1.get("status") != "insufficient_data":
        ax.bar(["Same-task", "Cross-task"], [bp1["same_mean"], bp1["cross_mean"]],
               color=["#27ae60", "#c0392b"], edgecolor="white")
        ax.axhline(bp1["midpoint"], color="gray", linestyle="--", alpha=0.7,
                   label=f"midpoint = {bp1['midpoint']:.4f}")
        ax.set_ylabel("Mean alignment")
        status = "CONFIRMED" if bp1["confirmed"] else "FAILED"
        ax.set_title(f"B-P1: task-boundary detection\n{status}", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B-P1: task-boundary detection", fontsize=10)

    # B-P2: spectral separation ratio
    ax = axes[1]
    if bp2.get("status") != "insufficient_data":
        ax.bar(["Same-task", "Cross-task"], [bp2["same_mean"], bp2["cross_mean"]],
               color=["#27ae60", "#c0392b"], edgecolor="white")
        ax.set_ylabel("Mean alignment")
        status = "CONFIRMED" if bp2["confirmed"] else "FAILED"
        ax.set_title(f"B-P2: spectral separation\nratio={bp2['ratio']:.2f}, "
                     f"p={bp2['welch_p']:.2e}\n{status}", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B-P2: spectral separation", fontsize=10)

    # B-P4: erank ANOVA
    ax = axes[2]
    if bp4.get("status") != "insufficient_data":
        task_means = bp4["task_means"]
        tasks = sorted(task_means.keys())
        vals = [task_means[t] for t in tasks]
        ax.barh(tasks, vals, color="#3498db", edgecolor="white")
        ax.set_xlabel("Mean erank")
        status = "CONFIRMED" if bp4["confirmed"] else "FAILED"
        ax.set_title(f"B-P4: erank ANOVA\nF={bp4['anova_F']:.1f}, "
                     f"p={bp4['anova_p']:.2e}\n{status}", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B-P4: erank ANOVA", fontsize=10)

    fig.suptitle("Confirmatory replications (non-gating)", fontsize=13)  # ANON: project-number prefix stripped.
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 74)
    print("  Primary H1 analysis: O-module depth-weighted alignment")  # ANON: stripped.
    print("=" * 74)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    pair_full, _pair_summary, profiles, pair_sample, merges = load_inputs()
    print(f"\n  Loaded {len(profiles)} adapter profiles, "
          f"{len(pair_full)} pairs, {len(merges['results'])} merge evals")

    # ---- Build eval set ----
    eval_degs = build_eval_pairs(merges)
    eval_pairs = sorted(eval_degs.keys())
    n_eval = len(eval_pairs)
    print(f"  Cross-task pairs with evaluation: {n_eval}")

    # ---- Compute S_H1 for all pairs (with symmetric key lookup) ----
    # merge_eval writes pair keys in the order committed by pair_sample.json
    # (whichever adapter was "a"); pair_alignment_full.json (Phase 2) uses
    # alphabetical "{min}_vs_{max}" ordering. Use the module-level symmetric
    # lookup so both orderings resolve to the same pair entry.
    s_h1_raw = {key: compute_s_h1(pair) for key, pair in pair_full.items()}

    def _lookup_s_h1(pair_key: str) -> float:
        entry = lookup_pair(pair_key, pair_full)
        # map pair_full entry back to its key to fetch s_h1
        for k, v in pair_full.items():
            if v is entry:
                return s_h1_raw[k]
        raise KeyError(pair_key)

    scores = np.array([_lookup_s_h1(p) for p in eval_pairs])
    y = np.array([eval_degs[p] for p in eval_pairs])

    print(f"\n  S_H1 range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  max_degradation range: [{y.min():.4f}, {y.max():.4f}]")

    # ---- Tied pairs ----
    tied_info = detect_tied_pairs(scores)
    print(f"\n  Tied-pair clusters (|S_A - S_B| < 1e-6): {tied_info['n_tied_clusters']}")
    if tied_info["n_tied_clusters"] > 0:
        print(f"  Tied cluster sizes: {tied_info['tied_cluster_sizes']}")

    # ---- Family-pair dummies ----
    cell_labels = [family_pair_label(lookup_pair(p, pair_full)["task_a"],
                                     lookup_pair(p, pair_full)["task_b"])
                   for p in eval_pairs]
    n_cells = len(set(cell_labels))
    print(f"  Family-pair cells: {n_cells}")

    # ---- H1 decision rule ----
    print("\n" + "-" * 74)
    print("  H1 DECISION RULE")
    print("-" * 74)

    decision = evaluate_h1_decision(scores, y, cell_labels)

    rho_raw = decision["rho_raw"]
    p_raw = decision["p_raw"]
    rho_partial = decision["rho_partial"]
    p_partial = decision["p_partial"]
    r2_base = decision["r2_base"]
    r2_full = decision["r2_full"]
    delta_r2 = decision["delta_r2"]
    crit1_pass = decision["crit1_pass"]
    crit2_pass = decision["crit2_pass"]
    sign_correct = decision["sign_correct"]
    h1_confirmed = decision["h1_confirmed"]
    boot = decision["bootstrap"]

    print(f"\n  Raw Spearman rho(S_H1, max_deg): {rho_raw:+.4f} (p = {p_raw:.4e})")
    print(f"  Partial Spearman rho (residualized on FAMILY_B): {rho_partial:+.4f} "
          f"(p = {p_partial:.4e})")
    print(f"    Criterion 1 (rho >= 0.50, p < 0.05): {'PASS' if crit1_pass else 'FAIL'}")
    print(f"\n  R2 (family-only baseline): {r2_base:.4f}")
    print(f"  R2 (family + S_H1):        {r2_full:.4f}")
    print(f"  Delta R2:                  {delta_r2:+.4f}")
    print(f"    Criterion 2 (delta-R2 >= 0.10): {'PASS' if crit2_pass else 'FAIL'}")
    print(f"\n  Sign check (rho_raw > 0 means higher alignment -> higher degradation): "
          f"{'CORRECT' if sign_correct else 'WRONG'}")
    print(f"\n  >>> H1 OVERALL: {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'} <<<")

    # ---- Bootstrap CIs ----
    print("\n" + "-" * 74)
    print(f"  BLOCK-BOOTSTRAP CIs ({N_BOOTSTRAP} resamples, family-pair blocked)")
    print("-" * 74)

    print(f"\n  Partial rho: {boot['rho_mean']:.3f} "
          f"[{boot['rho_ci_025']:.3f}, {boot['rho_ci_975']:.3f}]")
    print(f"  Delta R2:    {boot['dr2_mean']:.3f} "
          f"[{boot['dr2_ci_025']:.3f}, {boot['dr2_ci_975']:.3f}]")
    print(f"  Valid bootstrap samples: {boot['n_valid_boots']}")

    # ---- Confirmatory replications ----
    print("\n" + "-" * 74)
    print("  CONFIRMATORY REPLICATIONS (non-gating)")
    print("-" * 74)

    bp1 = replication_bp1(pair_full, pair_sample)
    print("\n  B-P1 (task-boundary detection):")
    if bp1.get("status") != "insufficient_data":
        print(f"    Same-task below midpoint: {bp1['same_below_midpoint']}/{bp1['n_same']} "
              f"(expected 0)")
        print(f"    Cross-task above midpoint: {bp1['cross_above_midpoint']}/{bp1['n_cross']} "
              f"(expected 0)")
        print(f"    {'CONFIRMED' if bp1['confirmed'] else 'FAILED'}")
    else:
        print("    Insufficient data")

    bp2 = replication_bp2(pair_full)
    print("\n  B-P2 (spectral separation):")
    if bp2.get("status") != "insufficient_data":
        print(f"    Same mean: {bp2['same_mean']:.4f}, Cross mean: {bp2['cross_mean']:.4f}")
        print(f"    Ratio: {bp2['ratio']:.2f} (threshold: 2.0)")
        print(f"    Welch t: {bp2['welch_t']:.2f}, p = {bp2['welch_p']:.2e} (threshold: 0.001)")
        print(f"    {'CONFIRMED' if bp2['confirmed'] else 'FAILED'}")
    else:
        print("    Insufficient data")

    bp4 = replication_bp4(profiles)
    print("\n  B-P4 (erank ANOVA across tasks):")
    if bp4.get("status") != "insufficient_data":
        print(f"    F = {bp4['anova_F']:.2f}, p = {bp4['anova_p']:.2e} "
              f"(threshold: 1e-6)")
        print(f"    {'CONFIRMED' if bp4['confirmed'] else 'FAILED'}")
    else:
        print("    Insufficient data")

    # ---- Figures ----
    print("\n" + "-" * 74)
    print("  FIGURES")
    print("-" * 74)

    scatter_path = FIG_DIR / "h1_scatter.png"
    plot_h1_scatter(scores, y, eval_pairs, pair_full, rho_partial, delta_r2, scatter_path)
    print(f"\n  Saved {scatter_path}")

    boot_path = FIG_DIR / "h1_bootstrap.png"
    plot_bootstrap_distribution(boot, boot_path)
    print(f"  Saved {boot_path}")

    repl_path = FIG_DIR / "h1_replications.png"
    plot_replications(bp1, bp2, bp4, repl_path)
    print(f"  Saved {repl_path}")

    # ---- Save JSON ----
    result = {
        "experiment": "H1: O-module depth-weighted alignment (decoder-scale controlled triage)",  # ANON: stripped.
        "n_eval_cross_task_pairs": n_eval,
        "n_family_pair_cells": n_cells,
        "h1_score": {
            "description": "S_H1 = sum(w_l * alpha_O_l) / sum(w_l), w_l = (l+1)/L",
            "range": [float(scores.min()), float(scores.max())],
        },
        "h1_decision_rule": {
            "criterion_1_partial_spearman": {
                "rho_partial": rho_partial,
                "p_partial": p_partial,
                "threshold_rho": 0.50,
                "threshold_p": 0.05,
                "pass": crit1_pass,
            },
            "criterion_2_delta_r2": {
                "r2_base": r2_base,
                "r2_full": r2_full,
                "delta_r2": delta_r2,
                "threshold": 0.10,
                "pass": crit2_pass,
            },
            "criterion_3_sign": {
                "rho_raw": float(rho_raw),
                "sign_correct": sign_correct,
            },
            "h1_confirmed": h1_confirmed,
        },
        "raw_spearman": {"rho": float(rho_raw), "p": float(p_raw)},
        "bootstrap": boot,
        "tied_pairs": tied_info,
        "replications": {
            "bp1_task_boundary": bp1,
            "bp2_spectral_separation": bp2,
            "bp4_erank_anova": bp4,
        },
        "per_pair": [
            {
                "pair": p,
                "task_a": lookup_pair(p, pair_full)["task_a"],
                "task_b": lookup_pair(p, pair_full)["task_b"],
                "family_pair": family_pair_label(
                    lookup_pair(p, pair_full)["task_a"],
                    lookup_pair(p, pair_full)["task_b"],
                ),
                "s_h1": float(_lookup_s_h1(p)),
                "max_degradation": float(eval_degs[p]),
            }
            for p in eval_pairs
        ],
    }

    out_path = WORKSPACE / "analysis_h1.json"

    def _json_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"\n  Saved {out_path}")

    print("\n" + "=" * 74)
    print(f"  H1 result: {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'}")  # ANON: stripped.
    print("=" * 74)


if __name__ == "__main__":
    main()
