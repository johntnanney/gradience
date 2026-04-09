"""N133 Phase 4: Analysis — test B-P1 through B-P6.

Reads spectral audit and merge evaluation results, tests all 6 predictions,
generates figures.

Usage:
    python3 n133_analysis.py

Output: /workspace/n133/analysis/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AUDIT_DIR = Path("/workspace/n133/audits")
MERGE_DIR = Path("/workspace/n133/merges")
EVAL_DIR = Path("/workspace/n133/evals")
OUTPUT_DIR = Path("/workspace/n133/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["sst2", "mnli", "squad", "gsm8k", "code", "summarization"]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load all audit and merge data."""
    with open(AUDIT_DIR / "adapter_profiles.json") as f:
        profiles = json.load(f)
    with open(AUDIT_DIR / "w0_properties.json") as f:
        w0 = json.load(f)
    with open(AUDIT_DIR / "pair_alignment_summary.json") as f:
        pair_summary = json.load(f)

    # Load full pair data for per-layer analysis
    pair_full_path = AUDIT_DIR / "pair_alignment_full.json"
    pair_full = {}
    if pair_full_path.exists():
        with open(pair_full_path) as f:
            pair_full = json.load(f)

    # Load merge results
    merge_summary_path = MERGE_DIR / "merge_eval_summary.json"
    merge_results = {}
    if merge_summary_path.exists():
        with open(merge_summary_path) as f:
            merge_results = json.load(f)

    return profiles, w0, pair_summary, pair_full, merge_results


# ---------------------------------------------------------------------------
# B-P1: Task-boundary detection
# ---------------------------------------------------------------------------

def test_bp1(pair_summary: dict, merge_results: dict) -> dict:
    """B-P1: Zero false positives on task-boundary detection."""
    print("\n=== B-P1: Task-Boundary Detection ===\n")

    same_task = [(k, v) for k, v in pair_summary.items() if v["is_same_task"]]
    cross_task = [(k, v) for k, v in pair_summary.items() if not v["is_same_task"]]

    # Use alignment threshold to classify: pairs with alignment > threshold = compatible
    same_aligns = [v["mean_alignment"] for _, v in same_task]
    cross_aligns = [v["mean_alignment"] for _, v in cross_task]

    same_mean = np.mean(same_aligns) if same_aligns else 0
    cross_mean = np.mean(cross_aligns) if cross_aligns else 0

    # Automatic threshold: midpoint
    threshold = (same_mean + cross_mean) / 2

    # Check for false positives: same-task classified as cross-task
    false_positives = sum(1 for a in same_aligns if a < threshold)
    # Check for false negatives: cross-task classified as same-task
    false_negatives = sum(1 for a in cross_aligns if a > threshold)

    print(f"  Same-task pairs: {len(same_task)}")
    print(f"  Cross-task pairs: {len(cross_task)}")
    print(f"  Same-task mean alignment: {same_mean:.4f}")
    print(f"  Cross-task mean alignment: {cross_mean:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  False positives (same classified as cross): {false_positives}")
    print(f"  False negatives (cross classified as same): {false_negatives}")

    decision = "CONFIRMED" if false_positives == 0 else "FALSE_POSITIVE"
    print(f"  Decision: {decision}")

    return {
        "n_same_task": len(same_task),
        "n_cross_task": len(cross_task),
        "same_mean_alignment": float(same_mean),
        "cross_mean_alignment": float(cross_mean),
        "threshold": float(threshold),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# B-P2: Same/cross spectral separation
# ---------------------------------------------------------------------------

def test_bp2(pair_summary: dict) -> dict:
    """B-P2: Same-task vs cross-task spectral separation."""
    print("\n=== B-P2: Spectral Separation ===\n")

    same_aligns = [v["mean_alignment"] for v in pair_summary.values() if v["is_same_task"]]
    cross_aligns = [v["mean_alignment"] for v in pair_summary.values() if not v["is_same_task"]]

    t_stat, p_val = stats.ttest_ind(same_aligns, cross_aligns)
    ratio = np.mean(same_aligns) / np.mean(cross_aligns) if np.mean(cross_aligns) > 0 else float("inf")

    print(f"  Same-task mean: {np.mean(same_aligns):.4f} (n={len(same_aligns)})")
    print(f"  Cross-task mean: {np.mean(cross_aligns):.4f} (n={len(cross_aligns)})")
    print(f"  Ratio: {ratio:.2f}×")
    print(f"  t = {t_stat:.3f}, p = {p_val:.6f}")

    decision = "CONFIRMED" if p_val < 0.001 and ratio >= 2.0 else "DISCONFIRMED"
    print(f"  Decision: {decision}")

    return {
        "same_mean": float(np.mean(same_aligns)),
        "cross_mean": float(np.mean(cross_aligns)),
        "ratio": float(ratio),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# B-P3: C_k predicts alignment
# ---------------------------------------------------------------------------

def test_bp3(w0: dict, pair_full: dict) -> dict:
    """B-P3: C_k of Mistral W₀ predicts per-layer alignment."""
    print("\n=== B-P3: C_k Predicts Alignment ===\n")

    # Build C_k lookup
    ck_lookup = {}
    for layer_name, props in w0.items():
        ck_lookup[(props["layer"], props["module"])] = props["C_k"]

    # Compute per-layer alignment for same-task pairs
    same_task_pairs = {k: v for k, v in pair_full.items() if v["is_same_task"]}

    # Group by task
    task_correlations = {}
    all_ck = []
    all_align = []

    for task in TASKS:
        task_pairs = {k: v for k, v in same_task_pairs.items()
                      if v["task_a"] == task}
        if not task_pairs:
            continue

        ck_vals = []
        align_vals = []
        for pair_key, pair_data in task_pairs.items():
            for entry in pair_data["per_layer"]:
                key = (entry["layer"], entry["module"])
                if key in ck_lookup:
                    ck_vals.append(ck_lookup[key])
                    align_vals.append(entry["alignment"])

        if len(ck_vals) >= 5:
            rho, p = stats.spearmanr(ck_vals, align_vals)
            print(f"  {task.upper():15s}: ρ(C_k, alignment) = {rho:+.4f}, p = {p:.4f}, n = {len(ck_vals)}")
            task_correlations[task] = {"rho": float(rho), "p": float(p), "n": len(ck_vals)}
            all_ck.extend(ck_vals)
            all_align.extend(align_vals)

    # Pooled correlation
    if all_ck:
        rho_pooled, p_pooled = stats.spearmanr(all_ck, all_align)
        print(f"\n  Pooled (same-task): ρ = {rho_pooled:+.4f}, p = {p_pooled:.4f}, n = {len(all_ck)}")
    else:
        rho_pooled, p_pooled = 0, 1

    # Decision
    max_rho = max((abs(v["rho"]) for v in task_correlations.values()), default=0)
    if abs(rho_pooled) >= 0.30 or max_rho >= 0.40:
        decision = "CONFIRMED"
    else:
        decision = "DISCONFIRMED"
    print(f"  Decision: {decision}")

    return {
        "per_task": task_correlations,
        "pooled_rho": float(rho_pooled),
        "pooled_p": float(p_pooled),
        "max_task_rho": float(max_rho),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# B-P4: Erank varies by task
# ---------------------------------------------------------------------------

def test_bp4(profiles: dict, w0: dict, pair_full: dict) -> dict:
    """B-P4: Effective rank varies by task and moderates C_k."""
    print("\n=== B-P4: Erank Varies by Task ===\n")

    # Collect per-adapter mean erank
    task_eranks = {t: [] for t in TASKS}
    for adapter_name, profile in profiles.items():
        task = profile["task"]
        task_eranks[task].append(profile["mean_erank"])

    # Print per-task
    for task in TASKS:
        vals = task_eranks[task]
        if vals:
            print(f"  {task.upper():15s}: mean erank = {np.mean(vals):.2f} ± {np.std(vals):.2f} (n={len(vals)})")

    # One-way ANOVA
    groups = [task_eranks[t] for t in TASKS if task_eranks[t]]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n  ANOVA: F = {f_stat:.3f}, p = {p_val:.6f}")
    else:
        f_stat, p_val = 0, 1

    # Per-layer ANOVA (more power)
    layer_eranks = {t: [] for t in TASKS}
    for adapter_name, profile in profiles.items():
        task = profile["task"]
        for layer in profile["per_layer"]:
            layer_eranks[task].append(layer["erank"])

    layer_groups = [layer_eranks[t] for t in TASKS if layer_eranks[t]]
    if len(layer_groups) >= 2:
        f_layer, p_layer = stats.f_oneway(*layer_groups)
        print(f"  Per-layer ANOVA: F = {f_layer:.3f}, p = {p_layer:.2e}")
    else:
        f_layer, p_layer = 0, 1

    # Interaction test (N131 replication)
    print("\n  --- Hierarchical regression (N131 replication) ---")
    # This requires per-layer alignment + C_k + erank... build if data available
    ck_lookup = {(v["layer"], v["module"]): v["C_k"] for v in w0.values()}

    same_task_pairs = {k: v for k, v in pair_full.items() if v["is_same_task"]}
    rows = []
    for task in TASKS:
        task_pairs = {k: v for k, v in same_task_pairs.items() if v["task_a"] == task}
        if not task_pairs:
            continue
        # Mean alignment per layer
        layer_aligns = {}
        for pair_data in task_pairs.values():
            for entry in pair_data["per_layer"]:
                key = (entry["layer"], entry["module"])
                if key not in layer_aligns:
                    layer_aligns[key] = []
                layer_aligns[key].append(entry["alignment"])

        # Mean erank per layer for this task
        task_adapters = [p for p in profiles.values() if p["task"] == task]
        layer_erank_map = {}
        for adapter in task_adapters:
            for layer in adapter["per_layer"]:
                key = (layer["layer"], layer["module"])
                if key not in layer_erank_map:
                    layer_erank_map[key] = []
                layer_erank_map[key].append(layer["erank"])

        for key in layer_aligns:
            if key in ck_lookup and key in layer_erank_map:
                rows.append({
                    "layer": key[0], "module": key[1], "task": task,
                    "C_k": ck_lookup[key],
                    "mean_erank": float(np.mean(layer_erank_map[key])),
                    "mean_alignment": float(np.mean(layer_aligns[key])),
                })

    interaction_p = 1.0
    if len(rows) >= 20:
        import statsmodels.api as sm
        df = pd.DataFrame(rows)
        df["C_k_z"] = (df["C_k"] - df["C_k"].mean()) / df["C_k"].std()
        df["erank_z"] = (df["mean_erank"] - df["mean_erank"].mean()) / df["mean_erank"].std()
        df["interaction"] = df["C_k_z"] * df["erank_z"]

        y = df["mean_alignment"]
        m1 = sm.OLS(y, sm.add_constant(df[["C_k_z"]])).fit()
        m3 = sm.OLS(y, sm.add_constant(df[["C_k_z", "erank_z", "interaction"]])).fit()
        interaction_p = float(m3.pvalues.get("interaction", 1.0))
        print(f"  M1 R² = {m1.rsquared:.4f}")
        print(f"  M3 R² = {m3.rsquared:.4f}, interaction p = {interaction_p:.4f}")

    # Decision
    if p_layer < 0.001 and interaction_p < 0.05:
        decision = "CONFIRMED"
    elif p_layer < 0.001:
        decision = "PARTIAL"
    else:
        decision = "DISCONFIRMED"
    print(f"  Decision: {decision}")

    return {
        "task_eranks": {t: {"mean": float(np.mean(v)), "values": v}
                        for t, v in task_eranks.items() if v},
        "anova_f": float(f_stat),
        "anova_p": float(p_val),
        "anova_layer_f": float(f_layer),
        "anova_layer_p": float(p_layer),
        "interaction_p": interaction_p,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# B-P5: Triage eliminates >= 70% of pairs
# ---------------------------------------------------------------------------

def test_bp5(pair_summary: dict, merge_results: dict) -> dict:
    """B-P5: Triage eliminates >= 70% while retaining best pairs."""
    print("\n=== B-P5: Triage Effectiveness ===\n")

    if not merge_results or "results" not in merge_results:
        print("  No merge results available — SKIPPED")
        return {"decision": "SKIPPED"}

    # Rank pairs by spectral alignment
    all_pairs = sorted(pair_summary.items(), key=lambda x: x[1]["mean_alignment"], reverse=True)
    cross_pairs = [(k, v) for k, v in all_pairs if not v["is_same_task"]]
    n_cross = len(cross_pairs)

    # Find evaluated pairs and their degradation
    eval_results = {r["pair"]: r for r in merge_results["results"]}

    # Identify pairs where merge is beneficial (low degradation)
    good_merges = []
    bad_merges = []
    for pair_key, pair_data in eval_results.items():
        if pair_data.get("category") == "same_task":
            continue
        max_deg = max(pair_data.get(f"degradation_{t}", 0)
                      for t in [pair_data["task_a"], pair_data["task_b"]]
                      if f"degradation_{t}" in pair_data)
        if max_deg < 0.10:  # less than 10% degradation = acceptable
            good_merges.append(pair_key)
        else:
            bad_merges.append(pair_key)

    # How many cross-task pairs would triage eliminate?
    # Threshold: bottom 70% by alignment
    cutoff_idx = int(n_cross * 0.30)  # keep top 30%
    retained = set(k for k, _ in cross_pairs[:cutoff_idx])
    eliminated = set(k for k, _ in cross_pairs[cutoff_idx:])

    # Check if any good merges were eliminated
    missed = [p for p in good_merges if p in eliminated]

    elimination_rate = len(eliminated) / n_cross if n_cross > 0 else 0
    print(f"  Cross-task pairs: {n_cross}")
    print(f"  Retained (top 30%): {len(retained)}")
    print(f"  Eliminated (bottom 70%): {len(eliminated)}")
    print(f"  Elimination rate: {elimination_rate:.1%}")
    print(f"  Good merges found: {len(good_merges)}")
    print(f"  Good merges missed by triage: {len(missed)}")

    if elimination_rate >= 0.70 and len(missed) == 0:
        decision = "CONFIRMED"
    elif elimination_rate >= 0.50:
        decision = "PARTIAL"
    else:
        decision = "NOT_SUPPORTED"
    print(f"  Decision: {decision}")

    return {
        "n_cross": n_cross,
        "n_retained": len(retained),
        "n_eliminated": len(eliminated),
        "elimination_rate": float(elimination_rate),
        "n_good_merges": len(good_merges),
        "n_missed": len(missed),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# B-P6: Module-type asymmetry
# ---------------------------------------------------------------------------

def test_bp6(profiles: dict) -> dict:
    """B-P6: No systematic attn vs MLP module-type asymmetry."""
    print("\n=== B-P6: Module-Type Asymmetry ===\n")

    # Mistral LoRA targets q/k/v/o_proj only (all attention, no MLP)
    # So we test Q/K vs V/O or similar
    # Actually, B-P6 is about attention vs MLP — but our LoRA only targets attention.
    # Report: all modules are attention, so no attn/MLP comparison possible.
    # Instead, test Q/K vs V/O asymmetry (the encoder finding was V-module specific)

    qk_eranks = []
    vo_eranks = []

    for adapter_name, profile in profiles.items():
        for layer in profile["per_layer"]:
            if layer["module"] in ("Q", "K"):
                qk_eranks.append(layer["erank"])
            elif layer["module"] in ("V", "O"):
                vo_eranks.append(layer["erank"])

    if qk_eranks and vo_eranks:
        t_stat, p_val = stats.ttest_ind(qk_eranks, vo_eranks)
        print(f"  Q/K mean erank: {np.mean(qk_eranks):.3f} (n={len(qk_eranks)})")
        print(f"  V/O mean erank: {np.mean(vo_eranks):.3f} (n={len(vo_eranks)})")
        print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        t_stat, p_val = 0, 1

    # Note: LoRA only targets attention modules, so no attn/MLP comparison
    print(f"\n  Note: LoRA targets q/k/v/o_proj only — no MLP modules to compare.")
    print(f"  Reporting Q+K vs V+O asymmetry instead.")

    decision = "CONFIRMED" if p_val > 0.10 else "DISCONFIRMED"
    print(f"  Decision: {decision} (no module-type asymmetry)")

    return {
        "qk_mean": float(np.mean(qk_eranks)) if qk_eranks else None,
        "vo_mean": float(np.mean(vo_eranks)) if vo_eranks else None,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "note": "LoRA targets attention only — Q/K vs V/O comparison, not attn vs MLP",
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(profiles, w0, pair_summary, pair_full):
    """Generate diagnostic figures."""
    print("\n=== Generating Figures ===\n")

    # Fig 1: Erank by task
    fig, ax = plt.subplots(figsize=(10, 6))
    task_eranks = {t: [] for t in TASKS}
    for profile in profiles.values():
        task_eranks[profile["task"]].append(profile["mean_erank"])

    active_tasks = [t for t in TASKS if task_eranks[t]]
    data = [task_eranks[t] for t in active_tasks]
    positions = range(1, len(active_tasks) + 1)
    bp = ax.boxplot(data, positions=positions, tick_labels=[t.upper() for t in active_tasks],
                    patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(active_tasks)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean Effective Rank", fontsize=12)
    ax.set_title("Mistral-7B Adapter Effective Rank by Task (N133)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n133_fig1_erank_by_task.png", dpi=150)
    plt.close(fig)
    print("  Saved n133_fig1_erank_by_task.png")

    # Fig 2: Same vs cross alignment distribution
    same_aligns = [v["mean_alignment"] for v in pair_summary.values() if v["is_same_task"]]
    cross_aligns = [v["mean_alignment"] for v in pair_summary.values() if not v["is_same_task"]]

    if same_aligns and cross_aligns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(cross_aligns, bins=20, alpha=0.6, label="Cross-task", color="#e74c3c")
        ax.hist(same_aligns, bins=10, alpha=0.6, label="Same-task", color="#2980b9")
        ax.set_xlabel("Mean SV-Weighted Alignment", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Same-Task vs Cross-Task Spectral Alignment (N133)", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "n133_fig2_alignment_dist.png", dpi=150)
        plt.close(fig)
        print("  Saved n133_fig2_alignment_dist.png")

    # Fig 3: C_k vs alignment scatter
    if pair_full and w0:
        ck_lookup = {(v["layer"], v["module"]): v["C_k"] for v in w0.values()}
        same_task_pairs = {k: v for k, v in pair_full.items() if v["is_same_task"]}

        fig, ax = plt.subplots(figsize=(8, 6))
        colors_map = dict(zip(TASKS, plt.cm.tab10(np.linspace(0, 1, len(TASKS)))))

        for task in TASKS:
            task_pairs = {k: v for k, v in same_task_pairs.items() if v["task_a"] == task}
            if not task_pairs:
                continue
            ck_vals, align_vals = [], []
            for pair_data in task_pairs.values():
                for entry in pair_data["per_layer"]:
                    key = (entry["layer"], entry["module"])
                    if key in ck_lookup:
                        ck_vals.append(ck_lookup[key])
                        align_vals.append(entry["alignment"])
            if ck_vals:
                ax.scatter(ck_vals, align_vals, alpha=0.3, s=20,
                           color=colors_map.get(task, "gray"), label=task.upper())

        ax.set_xlabel("W₀ Energy Concentration (C_k)", fontsize=12)
        ax.set_ylabel("Per-Layer SV-Weighted Alignment", fontsize=12)
        ax.set_title("Mistral W₀ Concentration vs Alignment by Task", fontsize=13)
        ax.legend(fontsize=10, markerscale=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "n133_fig3_Ck_vs_alignment.png", dpi=150)
        plt.close(fig)
        print("  Saved n133_fig3_Ck_vs_alignment.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  N133 Phase 4: Analysis")
    print("=" * 60)

    t0 = time.time()

    profiles, w0, pair_summary, pair_full, merge_results = load_data()

    bp1 = test_bp1(pair_summary, merge_results)
    bp2 = test_bp2(pair_summary)
    bp3 = test_bp3(w0, pair_full)
    bp4 = test_bp4(profiles, w0, pair_full)
    bp5 = test_bp5(pair_summary, merge_results)
    bp6 = test_bp6(profiles)

    generate_figures(profiles, w0, pair_summary, pair_full)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"  N133 Results Summary ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"\n  | Prediction | Description                    | Decision        |")
    print(f"  |------------|--------------------------------|-----------------|")
    print(f"  | B-P1       | Task-boundary zero FP          | {bp1['decision']:15s} |")
    print(f"  | B-P2       | Same/cross separation ≥ 2.0×   | {bp2['decision']:15s} |")
    print(f"  | B-P3       | C_k predicts alignment         | {bp3['decision']:15s} |")
    print(f"  | B-P4       | Erank varies + moderates       | {bp4['decision']:15s} |")
    print(f"  | B-P5       | Triage eliminates ≥ 70%        | {bp5['decision']:15s} |")
    print(f"  | B-P6       | No module-type asymmetry       | {bp6['decision']:15s} |")

    # Save
    summary = {
        "experiment": "N133: Decoder-Scale Controlled Merge Triage",
        "elapsed_s": elapsed,
        "BP1": bp1,
        "BP2": bp2,
        "BP3": bp3,
        "BP4": bp4,
        "BP5": bp5,
        "BP6": bp6,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
