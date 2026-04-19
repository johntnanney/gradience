"""N132: DeBERTa Effective Rank Replication.

Replicates N130 effective-rank finding on DeBERTa-v3-base using existing N07
adapter data (8 adapters: 4 GLUE tasks × 2 seeds, rank 16). Tests:
  A-P1: Task dimensionality varies across GLUE tasks (ANOVA)
  A-P2: C_k of DeBERTa W₀ predicts alignment for high-erank tasks
  A-P3: Cross-architecture erank comparison (DistilBERT vs DeBERTa)

CPU-only. Uses N07 experiment_a_results (singular values, pairwise overlap)
and DeBERTa-v3-base pretrained weights from HuggingFace.

Output: sidecar/data/n132/
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - fallback path for lightweight envs
    sm = None

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N07_DIR = PROJECT_ROOT / "scripts" / "n07_deberta"
PROFILES_FILE = N07_DIR / "experiment_a_results" / "per_module_profiles.json"
PAIRWISE_FILE = N07_DIR / "experiment_a_results" / "per_module_pairwise.json"
N130_PHASE5 = PROJECT_ROOT / "sidecar" / "data" / "n130" / "phase5_prediction_3.json"
OUTPUT_DIR = PROJECT_ROOT / "sidecar" / "data" / "n132"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEBERTA_MODEL = "microsoft/deberta-v3-base"

# Module name mapping: N07 uses short names, W0 uses full param names
MODULE_MAP = {
    "query_proj": "attention.self.query_proj",
    "key_proj": "attention.self.key_proj",
    "value_proj": "attention.self.value_proj",
    "attention.output.dense": "attention.output.dense",
}

# Tasks in the N07 study
TASKS = ["mrpc", "qnli", "rte", "sst2"]
SEEDS = [42, 7]


# ---------------------------------------------------------------------------
# Core computations
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


def mp_threshold(S: np.ndarray, shape: tuple[int, int]) -> tuple[float, int]:
    """Gavish-Donoho optimal hard threshold."""
    d_out, d_in = shape
    beta = min(d_out, d_in) / max(d_out, d_in)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median_sv = float(np.median(S))
    if median_sv < 1e-12:
        return 0.0, len(S)
    tau = omega * median_sv
    k = int(np.sum(tau < S))
    return tau, max(k, 1)


def energy_concentration(S: np.ndarray, k: int) -> float:
    """Energy concentration C_k = sum(S[:k]^2) / sum(S^2)."""
    total = np.sum(S**2)
    if total < 1e-20:
        return 0.0
    return float(np.sum(S[:k]**2) / total)


def fit_ols(y: pd.Series | np.ndarray, X: pd.DataFrame | np.ndarray) -> dict:
    """Fit OLS with either statsmodels or a lightweight NumPy fallback."""
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(X, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    design = np.column_stack([np.ones(len(y_arr)), x_arr])

    if sm is not None:
        model = sm.OLS(y_arr, design).fit()
        params = np.asarray(model.params, dtype=np.float64)
        pvalues = np.asarray(model.pvalues, dtype=np.float64)
        return {
            "rsquared": float(model.rsquared),
            "params": params,
            "pvalues": pvalues,
        }

    beta, residuals, rank, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ beta
    resid = y_arr - fitted
    n = len(y_arr)
    p = design.shape[1]
    y_mean = float(np.mean(y_arr))
    ss_tot = float(np.sum((y_arr - y_mean) ** 2))
    ss_res = float(np.sum(resid**2))
    rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if n <= p:
        pvalues = np.full(p, np.nan, dtype=np.float64)
    else:
        dof = n - p
        sigma2 = ss_res / dof
        xtx_inv = np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        pvalues = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)

    return {
        "rsquared": float(rsquared),
        "params": np.asarray(beta, dtype=np.float64),
        "pvalues": np.asarray(pvalues, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Phase 1: DeBERTa W₀ spectral properties
# ---------------------------------------------------------------------------

def phase1_w0_properties() -> pd.DataFrame:
    """Compute C_k for each LoRA-targeted layer of DeBERTa-v3-base."""
    print("\n=== Phase 1: DeBERTa W₀ Spectral Properties ===\n")

    from transformers import AutoModel
    model = AutoModel.from_pretrained(DEBERTA_MODEL, local_files_only=True)

    rows = []
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        # Match LoRA-targeted layers
        matched_module = None
        for short_name, full_name in MODULE_MAP.items():
            if full_name in name:
                matched_module = short_name
                break
        if matched_module is None:
            continue

        # Extract layer index
        parts = name.split(".")
        if "layer" not in parts:
            continue
        layer_idx = int(parts[parts.index("layer") + 1])

        W = param.detach().cpu().numpy().astype(np.float64)
        S = svd(W, compute_uv=False)
        tau, k = mp_threshold(S, W.shape)
        C_k = energy_concentration(S, k)

        # Module short name for matching
        module_short = {"query_proj": "Q", "key_proj": "K",
                        "value_proj": "V", "attention.output.dense": "O"}[matched_module]

        rows.append({
            "layer": layer_idx,
            "module": matched_module,
            "module_short": module_short,
            "shape_0": W.shape[0],
            "shape_1": W.shape[1],
            "k": k,
            "tau": tau,
            "C_k": C_k,
            "sigma_1": float(S[0]),
        })

    del model
    df = pd.DataFrame(rows).sort_values(["layer", "module_short"]).reset_index(drop=True)
    print(f"  Computed C_k for {len(df)} layers")
    print(f"  C_k range: [{df['C_k'].min():.3f}, {df['C_k'].max():.3f}]")
    print(f"  k range: [{df['k'].min()}, {df['k'].max()}]")
    return df


# ---------------------------------------------------------------------------
# Phase 2: Compute adapter effective rank from N07 singular values
# ---------------------------------------------------------------------------

def phase2_adapter_erank() -> pd.DataFrame:
    """Compute entropy effective rank from stored singular values."""
    print("\n=== Phase 2: Adapter Effective Rank ===\n")

    with open(PROFILES_FILE) as f:
        profiles = json.load(f)

    rows = []
    for adapter_name, layer_data in profiles.items():
        # Parse task and seed
        parts = adapter_name.split("_")
        task = parts[1]  # deberta_{task}_s{seed}
        seed = int(parts[2][1:])

        for entry in layer_data:
            svs = np.array(entry["singular_values"], dtype=np.float64)
            erank = entropy_effective_rank(svs)

            rows.append({
                "adapter": adapter_name,
                "task": task,
                "seed": seed,
                "layer": entry["layer"],
                "module": entry["module"],
                "module_short": entry["module_short"],
                "erank": erank,
                "stable_rank": entry["stable_rank"],
                "frobenius_norm": entry["frobenius_norm"],
            })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} entries ({len(profiles)} adapters × {len(layer_data)} layers)")

    # Per-task summary
    for task in TASKS:
        sub = df[df.task == task]
        print(f"  {task.upper():5s}: mean erank = {sub['erank'].mean():.2f}, "
              f"sd = {sub['erank'].std():.2f}")

    return df


# ---------------------------------------------------------------------------
# Phase 3: Per-layer alignment from N07 pairwise overlap
# ---------------------------------------------------------------------------

def phase3_pairwise_alignment() -> pd.DataFrame:
    """Load per-layer overlap from N07 pairwise data."""
    print("\n=== Phase 3: Pairwise Alignment ===\n")

    with open(PAIRWISE_FILE) as f:
        pairwise = json.load(f)

    rows = []
    for pair_name, layer_data in pairwise.items():
        # Parse tasks from pair name
        parts = pair_name.split("_vs_")
        adapter_a, adapter_b = parts[0], parts[1]
        task_a = adapter_a.split("_")[1]
        task_b = adapter_b.split("_")[1]
        is_same_task = task_a == task_b

        for entry in layer_data:
            rows.append({
                "pair": pair_name,
                "task_a": task_a,
                "task_b": task_b,
                "is_same_task": is_same_task,
                "layer": entry["layer"],
                "module": entry["module"],
                "module_short": entry["module_short"],
                "overlap": entry["overlap"],
                "dimensionality_ratio": entry["dimensionality_ratio"],
                "agreement": entry["agreement"],
            })

    df = pd.DataFrame(rows)
    same = df[df.is_same_task]
    cross = df[~df.is_same_task]
    print(f"  {len(df)} entries ({len(pairwise)} pairs × {len(layer_data)} layers)")
    print(f"  Same-task pairs: {same['pair'].nunique()} ({len(same)} entries)")
    print(f"  Cross-task pairs: {cross['pair'].nunique()} ({len(cross)} entries)")
    print(f"  Same-task mean overlap: {same['overlap'].mean():.4f}")
    print(f"  Cross-task mean overlap: {cross['overlap'].mean():.4f}")

    return df


# ---------------------------------------------------------------------------
# Phase 4: Prediction A-P1 — Task dimensionality
# ---------------------------------------------------------------------------

def phase4_ap1(erank_df: pd.DataFrame) -> dict:
    """Test: erank varies significantly across GLUE tasks on DeBERTa."""
    print("\n=== Phase 4: A-P1 — Task Dimensionality ===\n")

    # Mean erank per adapter (average across layers)
    adapter_means = erank_df.groupby(["adapter", "task"])["erank"].mean().reset_index()

    # One-way ANOVA across tasks
    groups = [adapter_means[adapter_means.task == t]["erank"].values for t in TASKS]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"  One-way ANOVA: F = {f_stat:.3f}, p = {p_val:.6f}")

    # Per-task means
    task_means = {}
    for task in TASKS:
        vals = adapter_means[adapter_means.task == task]["erank"].values
        task_means[task] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                           "values": vals.tolist()}
        print(f"  {task.upper():5s}: mean erank = {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Pairwise comparisons (Tukey-like, using t-tests with Bonferroni)
    pairwise_results = []
    max_d = 0
    for t1, t2 in combinations(TASKS, 2):
        v1 = adapter_means[adapter_means.task == t1]["erank"].values
        v2 = adapter_means[adapter_means.task == t2]["erank"].values
        t_stat, p = stats.ttest_ind(v1, v2)
        pooled_sd = np.sqrt((np.var(v1) + np.var(v2)) / 2)
        d = abs(np.mean(v1) - np.mean(v2)) / pooled_sd if pooled_sd > 0 else 0
        max_d = max(max_d, d)
        pairwise_results.append({
            "pair": f"{t1} vs {t2}",
            "t": float(t_stat), "p": float(p),
            "cohens_d": float(d),
        })
        if d >= 0.5:
            print(f"    {t1.upper()} vs {t2.upper()}: d = {d:.2f}, p = {p:.4f}")

    # Also do per-layer ANOVA (more statistical power: n=48 per task)
    layer_groups = [erank_df[erank_df.task == t]["erank"].values for t in TASKS]
    f_layer, p_layer = stats.f_oneway(*layer_groups)
    print(f"\n  Per-layer ANOVA (n=96 per task): F = {f_layer:.3f}, p = {p_layer:.2e}")

    # Decision
    if p_layer < 0.01 and max_d >= 1.0:
        decision = "CONFIRMED"
    elif p_layer < 0.01:
        decision = "PARTIAL"
    else:
        decision = "DISCONFIRMED"
    print(f"  Decision: {decision}")

    return {
        "anova_f": float(f_stat), "anova_p": float(p_val),
        "anova_layer_f": float(f_layer), "anova_layer_p": float(p_layer),
        "task_means": task_means,
        "pairwise": pairwise_results,
        "max_cohens_d": float(max_d),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Phase 5: Prediction A-P2 — C_k predicts alignment
# ---------------------------------------------------------------------------

def phase5_ap2(w0_df: pd.DataFrame, erank_df: pd.DataFrame,
               align_df: pd.DataFrame) -> dict:
    """Test: C_k predicts alignment for high-erank tasks, not low-erank."""
    print("\n=== Phase 5: A-P2 — C_k vs Alignment ===\n")

    # Build per-(layer, module) C_k lookup
    ck_lookup = {}
    for _, row in w0_df.iterrows():
        ck_lookup[(row["layer"], row["module_short"])] = row["C_k"]

    # For each same-task pair, compute mean overlap per (layer, module)
    same_task = align_df[align_df.is_same_task].copy()

    # Mean erank per task (to rank tasks)
    task_erank = erank_df.groupby("task")["erank"].mean()
    sorted_tasks = task_erank.sort_values()
    print(f"  Task erank ranking: {dict(sorted_tasks.round(2))}")
    lowest_task = sorted_tasks.index[0]
    highest_task = sorted_tasks.index[-1]
    print(f"  Lowest erank: {lowest_task.upper()} ({sorted_tasks.iloc[0]:.2f})")
    print(f"  Highest erank: {highest_task.upper()} ({sorted_tasks.iloc[-1]:.2f})")

    # Per-task C_k-alignment correlation
    results = {}
    for task in TASKS:
        # Get same-task pair for this task
        task_pairs = same_task[
            (same_task.task_a == task) & (same_task.task_b == task)
        ]
        if len(task_pairs) == 0:
            print(f"  {task.upper()}: no same-task pairs found")
            results[task] = {"rho": float("nan"), "p": float("nan"), "n": 0}
            continue

        # Mean overlap per (layer, module)
        mean_overlap = task_pairs.groupby(["layer", "module_short"])["overlap"].mean()

        ck_vals = []
        align_vals = []
        for (layer, module), overlap in mean_overlap.items():
            if (layer, module) in ck_lookup:
                ck_vals.append(ck_lookup[(layer, module)])
                align_vals.append(overlap)

        ck_arr = np.array(ck_vals)
        align_arr = np.array(align_vals)

        rho, p = stats.spearmanr(ck_arr, align_arr)
        print(f"  {task.upper():5s}: ρ(C_k, overlap) = {rho:+.4f}, p = {p:.4f}, n = {len(ck_arr)}")
        results[task] = {"rho": float(rho), "p": float(p), "n": len(ck_arr)}

    # Decision: highest erank task ρ ≥ 0.30, lowest erank task ρ < 0.15
    rho_high = results[highest_task]["rho"]
    rho_low = results[lowest_task]["rho"]
    if abs(rho_high) >= 0.30 and abs(rho_low) < 0.15:
        decision = "CONFIRMED"
    elif abs(rho_high) >= 0.30:
        decision = "PARTIAL"
    else:
        decision = "DISCONFIRMED"

    print(f"\n  Highest-erank task ({highest_task.upper()}): ρ = {rho_high:+.4f}")
    print(f"  Lowest-erank task ({lowest_task.upper()}): ρ = {rho_low:+.4f}")
    print(f"  Decision: {decision}")

    # Also: pooled hierarchical regression (N131 replication)
    print("\n  --- N131 Replication: Hierarchical Regression (pooled) ---")

    # Build pooled dataset: one row per (layer, module, task)
    pooled_rows = []
    for task in TASKS:
        task_pairs = same_task[(same_task.task_a == task) & (same_task.task_b == task)]
        if len(task_pairs) == 0:
            continue
        mean_overlap = task_pairs.groupby(["layer", "module_short"])["overlap"].mean()
        task_erank_vals = erank_df[erank_df.task == task].groupby(
            ["layer", "module_short"]
        )["erank"].mean()

        for (layer, module), overlap in mean_overlap.items():
            if (layer, module) not in ck_lookup:
                continue
            er = task_erank_vals.get((layer, module), np.nan)
            if np.isnan(er):
                continue
            pooled_rows.append({
                "layer": layer, "module": module, "task": task,
                "C_k": ck_lookup[(layer, module)],
                "mean_erank": er,
                "mean_alignment": overlap,
            })

    pooled_df = pd.DataFrame(pooled_rows)
    if len(pooled_df) >= 10:
        # Pooled Spearman + within-module residualized correlation
        # (Simpson's paradox guard). See docs/FINDINGS.md §28 and
        # sidecar/notes/n133b_distilbert_per_module_reanalysis.md
        rho_pool, p_pool = stats.spearmanr(pooled_df["C_k"], pooled_df["mean_alignment"])
        C_res = pooled_df["C_k"].astype(float).to_numpy().copy()
        A_res = pooled_df["mean_alignment"].astype(float).to_numpy().copy()
        mods = pooled_df["module"].to_numpy()
        for m in np.unique(mods):
            mask = mods == m
            if mask.sum() == 0:
                continue
            C_res[mask] -= C_res[mask].mean()
            A_res[mask] -= A_res[mask].mean()
        rho_wm, p_wm = stats.spearmanr(C_res, A_res)
        r_wm, pr_wm = stats.pearsonr(C_res, A_res)
        print(f"  Pooled ρ(C_k, align):        {rho_pool:+.4f} (p = {p_pool:.4f})")
        print(f"  Within-module ρ(C_k, align): {rho_wm:+.4f} (p = {p_wm:.4f})")
        print(f"  Within-module Pearson r:     {r_wm:+.4f} (p = {pr_wm:.4f})")
        if abs(rho_pool) >= 0.25 and abs(rho_wm) < 0.15:
            print(f"  ⚠ Simpson warning: pooled ρ={rho_pool:+.3f} but "
                  f"within-module ρ={rho_wm:+.3f} — pooled may be a "
                  f"between-module artifact.")
        if np.sign(rho_pool) != np.sign(rho_wm) and abs(rho_pool) >= 0.15:
            print(f"  ⚠ Simpson warning: pooled and within-module C_k "
                  f"correlations have OPPOSITE SIGN.")
        results["within_module"] = {
            "rho_pooled": float(rho_pool), "p_pooled": float(p_pool),
            "rho_within": float(rho_wm), "p_within": float(p_wm),
            "pearson_r_within": float(r_wm), "pearson_p_within": float(pr_wm),
            "n": int(len(pooled_df)),
        }

        pooled_df["C_k_z"] = (pooled_df["C_k"] - pooled_df["C_k"].mean()) / pooled_df["C_k"].std()
        pooled_df["erank_z"] = (pooled_df["mean_erank"] - pooled_df["mean_erank"].mean()) / pooled_df["mean_erank"].std()
        pooled_df["interaction"] = pooled_df["C_k_z"] * pooled_df["erank_z"]

        y = pooled_df["mean_alignment"]
        m1 = fit_ols(y, pooled_df[["C_k_z"]])
        m2 = fit_ols(y, pooled_df[["C_k_z", "erank_z"]])
        m3 = fit_ols(y, pooled_df[["C_k_z", "erank_z", "interaction"]])

        interaction_p = m3["pvalues"][3] if len(m3["pvalues"]) > 3 else float("nan")
        print(f"  M1 (C_k):              R² = {m1['rsquared']:.4f}")
        print(f"  M2 (C_k + erank):      R² = {m2['rsquared']:.4f}")
        print(f"  M3 (C_k + erank + ix): R² = {m3['rsquared']:.4f}")
        print(f"  Interaction p = {interaction_p:.4f}")

        results["hierarchical"] = {
            "m1_r2": float(m1["rsquared"]),
            "m2_r2": float(m2["rsquared"]),
            "m3_r2": float(m3["rsquared"]),
            "interaction_p": float(interaction_p),
            "n": len(pooled_df),
        }

    results["highest_erank_task"] = highest_task
    results["lowest_erank_task"] = lowest_task
    results["decision"] = decision

    return results


# ---------------------------------------------------------------------------
# Phase 6: Prediction A-P3 — Cross-architecture comparison
# ---------------------------------------------------------------------------

def phase6_ap3(erank_df: pd.DataFrame) -> dict:
    """Test: DeBERTa erank ordering matches DistilBERT for shared tasks."""
    print("\n=== Phase 6: A-P3 — Cross-Architecture Comparison ===\n")

    # DistilBERT eranks from N130
    distilbert = {"sst2": 5.473, "qnli": 13.316}

    # DeBERTa mean eranks
    deberta = {}
    for task in TASKS:
        deberta[task] = float(erank_df[erank_df.task == task]["erank"].mean())

    # Shared tasks
    shared = ["sst2", "qnli"]
    print("  Per-task mean erank:")
    print(f"  {'Task':8s} {'DistilBERT':>12s} {'DeBERTa':>12s}")
    for task in shared:
        db_val = distilbert.get(task, "—")
        de_val = deberta.get(task, "—")
        print(f"  {task.upper():8s} {db_val:>12.3f} {de_val:>12.3f}")
    for task in TASKS:
        if task not in shared:
            print(f"  {task.upper():8s} {'—':>12s} {deberta[task]:>12.3f}")

    # Rank ordering preserved?
    db_order = sorted(shared, key=lambda t: distilbert[t])
    de_order = sorted(shared, key=lambda t: deberta[t])
    ordering_preserved = db_order == de_order

    print(f"\n  DistilBERT ordering: {' < '.join(t.upper() for t in db_order)}")
    print(f"  DeBERTa ordering:   {' < '.join(t.upper() for t in de_order)}")
    print(f"  Ordering preserved: {ordering_preserved}")

    # Ratio comparison
    db_ratio = distilbert["qnli"] / distilbert["sst2"]
    de_ratio = deberta["qnli"] / deberta["sst2"]
    print(f"\n  DistilBERT QNLI/SST-2 ratio: {db_ratio:.2f}")
    print(f"  DeBERTa QNLI/SST-2 ratio:   {de_ratio:.2f}")

    # Per-layer comparison for shared tasks (Spearman across layers)
    # DeBERTa has 12 layers × 4 modules = 48 entries per task,
    # DistilBERT has 6 layers × 4 modules = 24, so can't do direct correlation.
    # Instead, compare task-level mean erank (n=2 is too small for Spearman).
    # Use the ratio as the comparison metric.

    if ordering_preserved:
        decision = "CONFIRMED"
    else:
        decision = "DISCONFIRMED"

    print(f"  Decision: {decision}")

    # Full DeBERTa task ordering
    full_order = sorted(TASKS, key=lambda t: deberta[t])
    print(f"\n  Full DeBERTa ordering: {' < '.join(t.upper() for t in full_order)}")

    return {
        "distilbert_eranks": distilbert,
        "deberta_eranks": deberta,
        "ordering_preserved": ordering_preserved,
        "db_ratio": db_ratio,
        "de_ratio": de_ratio,
        "full_ordering": full_order,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Phase 7: Diagnostic plots
# ---------------------------------------------------------------------------

def phase7_plots(w0_df: pd.DataFrame, erank_df: pd.DataFrame,
                 align_df: pd.DataFrame) -> None:
    """Generate diagnostic figures."""
    print("\n=== Phase 7: Diagnostic Plots ===\n")

    colors = {"mrpc": "#e74c3c", "qnli": "#2980b9", "rte": "#27ae60", "sst2": "#f39c12"}

    # --- Fig 1: Erank by task (box plot) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    task_data = [erank_df[erank_df.task == t]["erank"].values for t in TASKS]
    bp = ax.boxplot(task_data, tick_labels=[t.upper() for t in TASKS], patch_artist=True)
    for patch, task in zip(bp["boxes"], TASKS):
        patch.set_facecolor(colors[task])
        patch.set_alpha(0.6)
    ax.set_ylabel("Entropy Effective Rank", fontsize=12)
    ax.set_title("DeBERTa-v3-base: Adapter Effective Rank by Task\n"
                 "(N07, rank 16, 2 seeds per task)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Add DistilBERT reference points
    distilbert = {"sst2": 5.473, "qnli": 13.316}
    labeled = False
    for i, task in enumerate(TASKS):
        if task in distilbert:
            label = "DistilBERT" if not labeled else None
            ax.scatter(i + 1, distilbert[task], marker="*", s=200, c="black",
                       zorder=5, label=label)
            labeled = True
    if labeled:
        ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n132_fig1_erank_by_task.png", dpi=150)
    plt.close(fig)
    print("  Saved n132_fig1_erank_by_task.png")

    # --- Fig 2: C_k vs alignment per task ---
    ck_lookup = {(r["layer"], r["module_short"]): r["C_k"] for _, r in w0_df.iterrows()}
    same_task = align_df[align_df.is_same_task]

    fig, ax = plt.subplots(figsize=(8, 6))
    for task in TASKS:
        task_pairs = same_task[(same_task.task_a == task) & (same_task.task_b == task)]
        if len(task_pairs) == 0:
            continue
        mean_overlap = task_pairs.groupby(["layer", "module_short"])["overlap"].mean()
        ck_vals, align_vals = [], []
        for (layer, module), overlap in mean_overlap.items():
            if (layer, module) in ck_lookup:
                ck_vals.append(ck_lookup[(layer, module)])
                align_vals.append(overlap)

        ax.scatter(ck_vals, align_vals, c=colors[task], label=task.upper(),
                   alpha=0.5, s=40, edgecolors="white", linewidth=0.5)
        # Regression line
        if len(ck_vals) >= 3:
            z = np.polyfit(ck_vals, align_vals, 1)
            x_range = np.linspace(min(ck_vals), max(ck_vals), 50)
            ax.plot(x_range, np.polyval(z, x_range), c=colors[task],
                    linestyle="--", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("W₀ Energy Concentration (C_k)", fontsize=12)
    ax.set_ylabel("Per-Layer Overlap (same-task pairs)", fontsize=12)
    ax.set_title("DeBERTa W₀ Concentration vs Adapter Alignment by Task", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n132_fig2_Ck_vs_alignment.png", dpi=150)
    plt.close(fig)
    print("  Saved n132_fig2_Ck_vs_alignment.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(p1: dict, p2: dict, p3: dict, elapsed: float) -> None:
    """Print final results summary."""
    print(f"\n{'=' * 70}")
    print(f"  N132 Results Summary ({elapsed:.0f}s)")
    print(f"{'=' * 70}")

    print("\n  A-P1: Task Dimensionality")
    print(f"    ANOVA F = {p1['anova_layer_f']:.2f}, p = {p1['anova_layer_p']:.2e}")
    print(f"    Max pairwise d = {p1['max_cohens_d']:.2f}")
    for task in TASKS:
        print(f"    {task.upper():5s}: erank = {p1['task_means'][task]['mean']:.2f}")
    print(f"    Decision: {p1['decision']}")

    print("\n  A-P2: C_k Predicts Alignment")
    for task in TASKS:
        r = p2.get(task, {})
        if "rho" in r:
            print(f"    {task.upper():5s}: ρ = {r['rho']:+.4f}, p = {r['p']:.4f}")
    print(f"    Decision: {p2['decision']}")

    if "hierarchical" in p2:
        h = p2["hierarchical"]
        print(f"    Hierarchical: M1 R²={h['m1_r2']:.4f}, M2 R²={h['m2_r2']:.4f}, "
              f"M3 R²={h['m3_r2']:.4f}, ix p={h['interaction_p']:.4f}")

    print("\n  A-P3: Cross-Architecture Comparison")
    print(f"    Ordering preserved: {p3['ordering_preserved']}")
    print(f"    DistilBERT QNLI/SST-2 = {p3['db_ratio']:.2f}")
    print(f"    DeBERTa QNLI/SST-2 = {p3['de_ratio']:.2f}")
    print(f"    Full ordering: {' < '.join(t.upper() for t in p3['full_ordering'])}")
    print(f"    Decision: {p3['decision']}")

    print(f"\n  Output: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  N132: DeBERTa Effective Rank Replication")
    print("=" * 70)

    t0 = time.time()

    # Phase 1: W₀ properties
    w0_df = phase1_w0_properties()
    w0_df.to_csv(OUTPUT_DIR / "n132_deberta_Ck.csv", index=False)

    # Phase 2: Adapter effective rank
    erank_df = phase2_adapter_erank()
    erank_df.to_csv(OUTPUT_DIR / "n132_deberta_erank.csv", index=False)

    # Phase 3: Pairwise alignment
    align_df = phase3_pairwise_alignment()
    align_df.to_csv(OUTPUT_DIR / "n132_deberta_alignment.csv", index=False)

    # Phase 4: A-P1
    p1_results = phase4_ap1(erank_df)

    # Phase 5: A-P2
    p2_results = phase5_ap2(w0_df, erank_df, align_df)

    # Phase 6: A-P3
    p3_results = phase6_ap3(erank_df)

    # Phase 7: Plots
    phase7_plots(w0_df, erank_df, align_df)

    elapsed = time.time() - t0

    # Summary
    print_summary(p1_results, p2_results, p3_results, elapsed)

    # Save
    summary = {
        "experiment": "N132: DeBERTa Effective Rank Replication",
        "elapsed_s": elapsed,
        "prediction_ap1": p1_results,
        "prediction_ap2": {k: v for k, v in p2_results.items()},
        "prediction_ap3": p3_results,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Saved summary.json")


if __name__ == "__main__":
    main()
