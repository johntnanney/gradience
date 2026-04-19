"""N131: Composite Predictor Validation — C_k × f(erank).

Tests whether a composite predictor combining W₀ energy concentration (C_k)
and adapter effective rank (erank) outperforms C_k alone in predicting
per-layer SV-weighted inter-adapter alignment.

CPU-only. Uses N130 and N127 outputs. No model loading required.

Output: sidecar/data/n131/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N130_DIR = PROJECT_ROOT / "sidecar" / "data" / "n130"
N127_EXT = PROJECT_ROOT / "sidecar" / "results" / "mp_partition_test" / "extension_results.json"
OUTPUT_DIR = PROJECT_ROOT / "sidecar" / "data" / "n131"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1: Assemble the dataset
# ---------------------------------------------------------------------------

def phase1_assemble_dataset() -> pd.DataFrame:
    """Build the pooled (layer, task) dataset from N130 and N127 outputs."""
    print("\n=== Phase 1: Assemble Dataset ===\n")

    # Load C_k per layer from N130 phase1
    with open(N130_DIR / "phase1_w0_properties.json") as f:
        w0_data = json.load(f)
    ck_by_layer = {r["layer"]: r["C_k"] for r in w0_data["per_layer"]}
    module_by_layer = {r["layer"]: r["module"] for r in w0_data["per_layer"]}

    # Load per-layer erank from N130 phase5
    with open(N130_DIR / "phase5_prediction_3.json") as f:
        p5_data = json.load(f)
    erank_by_layer = {}
    for entry in p5_data["per_layer"]:
        erank_by_layer[entry["layer"]] = {
            "sst2": entry["sst2_erank"],
            "qnli": entry["qnli_erank"],
        }

    # Load per-layer alignment from N127 extension results
    with open(N127_EXT) as f:
        ext_data = json.load(f)
    ext1 = ext_data["extension_1_spectral_gap"]

    sst2_align_by_layer = {}
    for entry in ext1["per_layer"]:
        sst2_align_by_layer[entry["layer"]] = entry["mean_high_align"]

    qnli_align_by_layer = {}
    for entry in ext1["qnli"]["per_layer"]:
        qnli_align_by_layer[entry["layer"]] = entry["mean_high_align"]

    # Build rows
    rows = []
    for layer_name in sorted(ck_by_layer.keys()):
        if layer_name not in erank_by_layer:
            continue
        if layer_name not in sst2_align_by_layer:
            continue
        if layer_name not in qnli_align_by_layer:
            continue

        # Extract transformer layer index for clustering
        parts = layer_name.split(".")
        tf_layer = int(parts[parts.index("layer") + 1])
        module = module_by_layer[layer_name]
        c_k = ck_by_layer[layer_name]

        for task, align_dict, erank_val in [
            ("SST2", sst2_align_by_layer, erank_by_layer[layer_name]["sst2"]),
            ("QNLI", qnli_align_by_layer, erank_by_layer[layer_name]["qnli"]),
        ]:
            rows.append({
                "layer": layer_name,
                "layer_id": f"L{tf_layer}_{module.upper()}",
                "tf_layer": tf_layer,
                "module_type": module.upper(),
                "task": task,
                "C_k": c_k,
                "mean_erank": erank_val,
                "mean_alignment": align_dict[layer_name],
            })

    df = pd.DataFrame(rows)
    print(f"  Dataset: {len(df)} rows ({len(df)//2} layers × 2 tasks)")
    print(f"  C_k range: [{df['C_k'].min():.3f}, {df['C_k'].max():.3f}]")
    print(f"  erank range: [{df['mean_erank'].min():.2f}, {df['mean_erank'].max():.2f}]")
    print(f"  alignment range: [{df['mean_alignment'].min():.3f}, {df['mean_alignment'].max():.3f}]")

    # Summary by task
    for task in ["SST2", "QNLI"]:
        sub = df[df.task == task]
        print(f"  {task}: mean_erank={sub['mean_erank'].mean():.2f}, "
              f"mean_align={sub['mean_alignment'].mean():.3f}")

    return df


# ---------------------------------------------------------------------------
# Phase 2: Prediction 1 — Hierarchical regression
# ---------------------------------------------------------------------------

def phase2_prediction_1(df: pd.DataFrame) -> dict:
    """Test: C_k × erank outperforms C_k alone via interaction effect."""
    print("\n=== Phase 2: Prediction 1 — Hierarchical Regression ===\n")

    # Standardize predictors
    df = df.copy()
    df["C_k_z"] = (df["C_k"] - df["C_k"].mean()) / df["C_k"].std()
    df["erank_z"] = (df["mean_erank"] - df["mean_erank"].mean()) / df["mean_erank"].std()
    df["interaction"] = df["C_k_z"] * df["erank_z"]

    y = df["mean_alignment"]

    # Model 1: C_k only
    X1 = sm.add_constant(df[["C_k_z"]])
    m1 = sm.OLS(y, X1).fit()

    # Model 2: C_k + erank (additive)
    X2 = sm.add_constant(df[["C_k_z", "erank_z"]])
    m2 = sm.OLS(y, X2).fit()

    # Model 3: C_k + erank + interaction
    X3 = sm.add_constant(df[["C_k_z", "erank_z", "interaction"]])
    m3 = sm.OLS(y, X3).fit()

    print(f"  Model 1 (C_k):              R² = {m1.rsquared:.4f}, adj R² = {m1.rsquared_adj:.4f}")
    print(f"  Model 2 (C_k + erank):      R² = {m2.rsquared:.4f}, adj R² = {m2.rsquared_adj:.4f}")
    print(f"  Model 3 (C_k + erank + ix): R² = {m3.rsquared:.4f}, adj R² = {m3.rsquared_adj:.4f}")
    print()

    # Interaction term significance
    ix_p = m3.pvalues["interaction"]
    ix_coef = m3.params["interaction"]
    print(f"  Interaction coefficient: {ix_coef:.4f} (p = {ix_p:.4f})")

    # F-test: Model 3 vs Model 2
    from statsmodels.stats.anova import anova_lm
    f_test = anova_lm(m2, m3)
    f_stat = f_test["F"].iloc[1]
    f_pval = f_test["Pr(>F)"].iloc[1]
    print(f"  F-test (M3 vs M2): F = {f_stat:.3f}, p = {f_pval:.4f}")

    # F-test: Model 2 vs Model 1
    f_test_21 = anova_lm(m1, m2)
    f21_stat = f_test_21["F"].iloc[1]
    f21_pval = f_test_21["Pr(>F)"].iloc[1]
    print(f"  F-test (M2 vs M1): F = {f21_stat:.3f}, p = {f21_pval:.4f}")

    # Spearman for composite C_k × erank
    rho_composite, p_composite = stats.spearmanr(
        df["C_k"] * df["mean_erank"], df["mean_alignment"]
    )
    rho_ck_alone, p_ck_alone = stats.spearmanr(df["C_k"], df["mean_alignment"])
    print(f"\n  Spearman ρ(C_k, align) pooled:        {rho_ck_alone:+.4f} (p = {p_ck_alone:.4f})")
    print(f"  Spearman ρ(C_k × erank, align) pooled: {rho_composite:+.4f} (p = {p_composite:.4f})")

    # Within-module residualized correlation (Simpson's paradox guard).
    # Subtract per-module mean C_k and per-module mean alignment, then
    # correlate residuals. See docs/FINDINGS.md §28 and
    # sidecar/notes/n133b_distilbert_per_module_reanalysis.md
    C_res = df["C_k"].astype(float).to_numpy().copy()
    A_res = df["mean_alignment"].astype(float).to_numpy().copy()
    Comp_res = (df["C_k"] * df["mean_erank"]).astype(float).to_numpy().copy()
    mods = df["module_type"].to_numpy()
    for m in np.unique(mods):
        mask = mods == m
        if mask.sum() == 0:
            continue
        C_res[mask] -= C_res[mask].mean()
        A_res[mask] -= A_res[mask].mean()
        Comp_res[mask] -= Comp_res[mask].mean()
    rho_ck_wm, p_ck_wm = stats.spearmanr(C_res, A_res)
    r_ck_wm, pr_ck_wm = stats.pearsonr(C_res, A_res)
    rho_comp_wm, p_comp_wm = stats.spearmanr(Comp_res, A_res)
    print(f"  Within-module ρ(C_k, align):          {rho_ck_wm:+.4f} (p = {p_ck_wm:.4f})")
    print(f"  Within-module Pearson r(C_k, align):  {r_ck_wm:+.4f} (p = {pr_ck_wm:.4f})")
    print(f"  Within-module ρ(C_k×erank, align):    {rho_comp_wm:+.4f} (p = {p_comp_wm:.4f})")
    if abs(rho_ck_alone) >= 0.25 and abs(rho_ck_wm) < 0.15:
        print(f"  ⚠ Simpson warning: pooled ρ(C_k)={rho_ck_alone:+.3f} but "
              f"within-module ρ={rho_ck_wm:+.3f} — pooled may be a "
              f"between-module artifact.")
    if np.sign(rho_ck_alone) != np.sign(rho_ck_wm) and abs(rho_ck_alone) >= 0.15:
        print(f"  ⚠ Simpson warning: pooled and within-module C_k correlations "
              f"have OPPOSITE SIGN.")

    # Decision
    delta_r2 = m3.rsquared - m1.rsquared
    if ix_p < 0.05 and delta_r2 >= 0.10:
        decision = "CONFIRMED"
    elif f21_pval < 0.05:
        decision = "ADDITIVE"
    else:
        decision = "NULL"

    print(f"\n  ΔR² (M3 vs M1) = {delta_r2:.4f}")
    print(f"  Decision: {decision}")

    # Clustered standard errors (conservative: cluster by transformer layer)
    print("\n  --- Clustered SE check (6 transformer-layer clusters) ---")
    m3_clustered = sm.OLS(y, X3).fit(
        cov_type="cluster", cov_kwds={"groups": df["tf_layer"]}
    )
    ix_p_clustered = m3_clustered.pvalues["interaction"]
    print(f"  Interaction p (clustered): {ix_p_clustered:.4f}")
    if ix_p < 0.05 and ix_p_clustered >= 0.05:
        print("  ⚠ Interaction loses significance with clustered SEs")

    results = {
        "model_1": {"R2": m1.rsquared, "adj_R2": m1.rsquared_adj, "AIC": m1.aic, "BIC": m1.bic},
        "model_2": {"R2": m2.rsquared, "adj_R2": m2.rsquared_adj, "AIC": m2.aic, "BIC": m2.bic},
        "model_3": {
            "R2": m3.rsquared, "adj_R2": m3.rsquared_adj, "AIC": m3.aic, "BIC": m3.bic,
            "interaction_coef": float(ix_coef), "interaction_p": float(ix_p),
            "interaction_p_clustered": float(ix_p_clustered),
        },
        "f_test_m3_vs_m2": {"F": float(f_stat), "p": float(f_pval)},
        "f_test_m2_vs_m1": {"F": float(f21_stat), "p": float(f21_pval)},
        "delta_R2_m3_vs_m1": float(delta_r2),
        "spearman_ck_pooled": {"rho": float(rho_ck_alone), "p": float(p_ck_alone)},
        "spearman_composite_pooled": {"rho": float(rho_composite), "p": float(p_composite)},
        "within_module": {
            "rho_ck": float(rho_ck_wm), "p_ck": float(p_ck_wm),
            "pearson_r_ck": float(r_ck_wm), "pearson_p_ck": float(pr_ck_wm),
            "rho_composite": float(rho_comp_wm), "p_composite": float(p_comp_wm),
        },
        "decision": decision,
    }

    # Store models for later use
    results["_models"] = (m1, m2, m3)
    results["_df"] = df

    return results


# ---------------------------------------------------------------------------
# Phase 3: Prediction 2 — Within-task replication
# ---------------------------------------------------------------------------

def phase3_prediction_2(df: pd.DataFrame) -> dict:
    """Replicate: C_k predicts alignment for QNLI but not SST-2."""
    print("\n=== Phase 3: Prediction 2 — Within-Task Replication ===\n")

    results = {}
    for task in ["SST2", "QNLI"]:
        sub = df[df.task == task]
        rho, p = stats.spearmanr(sub["C_k"], sub["mean_alignment"])
        print(f"  {task}: ρ(C_k, alignment) = {rho:+.4f}, p = {p:.4f}, n = {len(sub)}")
        results[task.lower()] = {"rho": float(rho), "p": float(p), "n": len(sub)}

    # Decision
    rho_qnli = results["qnli"]["rho"]
    rho_sst2 = results["sst2"]["rho"]
    if abs(rho_qnli) > 0.40 and abs(rho_sst2) < 0.20:
        decision = "CONFIRMED"
    else:
        decision = "DISCONFIRMED"

    print(f"\n  Decision: {decision}")
    print("  (threshold: |ρ_QNLI| > 0.40 and |ρ_SST2| < 0.20)")
    results["decision"] = decision

    return results


# ---------------------------------------------------------------------------
# Phase 4: Prediction 3 — Functional form comparison
# ---------------------------------------------------------------------------

def phase4_prediction_3(df: pd.DataFrame) -> dict:
    """Compare functional forms: linear, log, threshold for f(erank)."""
    print("\n=== Phase 4: Prediction 3 — Functional Form ===\n")

    df = df.copy()

    # Composite predictors
    df["composite_linear"] = df["C_k"] * df["mean_erank"]
    df["composite_log"] = df["C_k"] * np.log(df["mean_erank"])
    tau = (5.5 + 13.3) / 2  # midpoint of task means
    df["composite_thresh"] = df["C_k"] * (df["mean_erank"] > tau).astype(float)

    print(f"  Threshold τ = {tau:.1f}")
    print()

    # Spearman correlations
    print("  Spearman correlations:")
    spearman_results = {}
    for name, col in [
        ("C_k alone", "C_k"),
        ("C_k × erank", "composite_linear"),
        ("C_k × log(erank)", "composite_log"),
        ("C_k × I(erank>τ)", "composite_thresh"),
    ]:
        rho, p = stats.spearmanr(df[col], df["mean_alignment"])
        print(f"    {name:25s}: ρ = {rho:+.4f}, p = {p:.4f}")
        spearman_results[name] = {"rho": float(rho), "p": float(p)}

    # OLS models for AIC/BIC
    print("\n  OLS model comparison:")
    y = df["mean_alignment"]
    ols_results = {}
    for name, col in [
        ("C_k alone", "C_k"),
        ("linear", "composite_linear"),
        ("log", "composite_log"),
        ("threshold", "composite_thresh"),
    ]:
        X = sm.add_constant(df[col])
        model = sm.OLS(y, X).fit()
        print(f"    {name:12s}: R² = {model.rsquared:.4f}, "
              f"AIC = {model.aic:.1f}, BIC = {model.bic:.1f}")
        ols_results[name] = {
            "R2": float(model.rsquared),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
        }

    # Determine best form
    composite_forms = {k: v for k, v in ols_results.items() if k != "C_k alone"}
    best_form = min(composite_forms, key=lambda k: composite_forms[k]["AIC"])
    best_aic = composite_forms[best_form]["AIC"]
    ck_aic = ols_results["C_k alone"]["AIC"]

    print(f"\n  Best composite form: {best_form} (AIC = {best_aic:.1f})")
    print(f"  C_k alone AIC: {ck_aic:.1f}")
    print(f"  ΔAIC (best composite vs C_k alone): {best_aic - ck_aic:.1f}")

    # Check if forms are discriminable (ΔAIC > 2 between best and worst composite)
    worst_form = max(composite_forms, key=lambda k: composite_forms[k]["AIC"])
    delta_aic_range = composite_forms[worst_form]["AIC"] - composite_forms[best_form]["AIC"]
    if delta_aic_range < 2.0:
        form_decision = "UNDERDETERMINED"
    else:
        form_decision = best_form.upper()
    print(f"  ΔAIC range across composites: {delta_aic_range:.1f}")
    print(f"  Form decision: {form_decision}")

    return {
        "spearman": spearman_results,
        "ols": ols_results,
        "best_form": best_form,
        "form_decision": form_decision,
        "threshold_tau": tau,
        "delta_aic_range": delta_aic_range,
    }


# ---------------------------------------------------------------------------
# Phase 5: Diagnostic plots
# ---------------------------------------------------------------------------

def phase5_diagnostic_plots(df: pd.DataFrame, p1_results: dict) -> None:
    """Generate four diagnostic figures."""
    print("\n=== Phase 5: Diagnostic Plots ===\n")

    df = df.copy()

    colors = {"SST2": "#e74c3c", "QNLI": "#2980b9"}
    markers = {"SST2": "o", "QNLI": "s"}

    # --- Fig 1: Scatter C_k vs alignment, colored by task ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for task in ["SST2", "QNLI"]:
        sub = df[df.task == task]
        ax.scatter(sub["C_k"], sub["mean_alignment"],
                   c=colors[task], marker=markers[task], label=task,
                   alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
        # Regression line
        z = np.polyfit(sub["C_k"], sub["mean_alignment"], 1)
        x_range = np.linspace(sub["C_k"].min(), sub["C_k"].max(), 50)
        ax.plot(x_range, np.polyval(z, x_range), c=colors[task],
                linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xlabel("W₀ Energy Concentration (C_k)", fontsize=12)
    ax.set_ylabel("Mean Same-Task Alignment (high-SV band)", fontsize=12)
    ax.set_title("C_k vs Alignment by Task\n(slope difference = moderation by erank)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n131_fig1_scatter_by_task.png", dpi=150)
    plt.close(fig)
    print("  Saved n131_fig1_scatter_by_task.png")

    # --- Fig 2: Composite C_k × erank vs alignment ---
    fig, ax = plt.subplots(figsize=(8, 6))
    composite = df["C_k"] * df["mean_erank"]
    for task in ["SST2", "QNLI"]:
        sub = df[df.task == task]
        comp_sub = composite[df.task == task]
        ax.scatter(comp_sub, sub["mean_alignment"],
                   c=colors[task], marker=markers[task], label=task,
                   alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
    # Overall regression line
    z = np.polyfit(composite, df["mean_alignment"], 1)
    x_range = np.linspace(composite.min(), composite.max(), 50)
    ax.plot(x_range, np.polyval(z, x_range), c="black",
            linestyle="-", alpha=0.5, linewidth=2)
    rho, p = stats.spearmanr(composite, df["mean_alignment"])
    ax.set_xlabel("C_k × erank (composite predictor)", fontsize=12)
    ax.set_ylabel("Mean Same-Task Alignment (high-SV band)", fontsize=12)
    ax.set_title(f"Composite Predictor vs Alignment\nSpearman ρ = {rho:.3f}, p = {p:.4f}", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n131_fig2_composite.png", dpi=150)
    plt.close(fig)
    print("  Saved n131_fig2_composite.png")

    # --- Fig 3: Interaction plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for task in ["SST2", "QNLI"]:
        sub = df[df.task == task]
        z = np.polyfit(sub["C_k"], sub["mean_alignment"], 1)
        x_range = np.linspace(df["C_k"].min(), df["C_k"].max(), 50)
        ax.plot(x_range, np.polyval(z, x_range), c=colors[task],
                linewidth=2.5, label=f"{task} (slope={z[0]:.3f})")
        ax.scatter(sub["C_k"], sub["mean_alignment"],
                   c=colors[task], marker=markers[task],
                   alpha=0.4, s=40, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("W₀ Energy Concentration (C_k)", fontsize=12)
    ax.set_ylabel("Mean Same-Task Alignment (high-SV band)", fontsize=12)
    ax.set_title("Interaction: C_k × Task (proxy for erank level)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n131_fig3_interaction.png", dpi=150)
    plt.close(fig)
    print("  Saved n131_fig3_interaction.png")

    # --- Fig 4: Residual plot for Model 3 ---
    m3 = p1_results["_models"][2]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m3.fittedvalues, m3.resid, alpha=0.6, s=50,
               edgecolors="white", linewidth=0.5)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title("Model 3 Residuals vs Fitted Values", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n131_fig4_residuals.png", dpi=150)
    plt.close(fig)
    print("  Saved n131_fig4_residuals.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(p1: dict, p2: dict, p3: dict, elapsed: float) -> None:
    """Print the final summary table."""
    print(f"\n{'=' * 70}")
    print(f"  N131 Results Summary ({elapsed:.0f}s)")
    print(f"{'=' * 70}")

    # Table 1: Hierarchical regression
    print("\n  Hierarchical Regression:")
    print("  | Model                      | R²     | ΔR² vs M1 | Key p-value      | Decision     |")
    print("  |----------------------------|--------|-----------|------------------|--------------|")
    r2_m1 = p1["model_1"]["R2"]
    r2_m2 = p1["model_2"]["R2"]
    r2_m3 = p1["model_3"]["R2"]
    print(f"  | M1: C_k                   | {r2_m1:.4f} |     —     |                  |              |")
    print(f"  | M2: C_k + erank           | {r2_m2:.4f} | {r2_m2-r2_m1:+.4f}   | F p={p1['f_test_m2_vs_m1']['p']:.4f}       |              |")
    print(f"  | M3: C_k + erank + C_k×er  | {r2_m3:.4f} | {r2_m3-r2_m1:+.4f}   | ix p={p1['model_3']['interaction_p']:.4f}      | {p1['decision']:12s} |")

    # Table 2: Composite predictor comparison
    print("\n  Composite Predictor Comparison:")
    print("  | Predictor            | Spearman ρ | p      | OLS R² | AIC    | BIC    |")
    print("  |----------------------|-----------|--------|--------|--------|--------|")
    for name in ["C_k alone", "C_k × erank", "C_k × log(erank)", "C_k × I(erank>τ)"]:
        sp = p3["spearman"][name]
        # Map spearman names to ols names
        ols_name_map = {
            "C_k alone": "C_k alone",
            "C_k × erank": "linear",
            "C_k × log(erank)": "log",
            "C_k × I(erank>τ)": "threshold",
        }
        ols = p3["ols"][ols_name_map[name]]
        print(f"  | {name:20s} | {sp['rho']:+.4f}    | {sp['p']:.4f} | {ols['R2']:.4f} | {ols['AIC']:6.1f} | {ols['BIC']:6.1f} |")

    # Table 3: Within-task replication
    print("\n  Within-Task Replication:")
    print("  | Task  | ρ(C_k, alignment) | p      | Consistent? |")
    print("  |-------|-------------------|--------|-------------|")
    for task in ["sst2", "qnli"]:
        r = p2[task]
        consistent = "Yes" if (task == "sst2" and abs(r["rho"]) < 0.20) or \
                              (task == "qnli" and abs(r["rho"]) > 0.40) else "No"
        print(f"  | {task.upper():5s} | {r['rho']:+.4f}            | {r['p']:.4f} | {consistent:11s} |")
    print(f"  | Decision: {p2['decision']}")

    print(f"\n  Best composite form: {p3['best_form']} ({p3['form_decision']})")
    print(f"  Output: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  N131: Composite Predictor Validation — C_k × f(erank)")
    print("=" * 70)

    t0 = time.time()

    # Phase 1: Assemble data
    df = phase1_assemble_dataset()
    df.to_csv(OUTPUT_DIR / "n131_composite_predictor.csv", index=False)

    # Phase 2: Prediction 1 — Hierarchical regression
    p1_results = phase2_prediction_1(df)

    # Phase 3: Prediction 2 — Within-task replication
    p2_results = phase3_prediction_2(df)

    # Phase 4: Prediction 3 — Functional form
    p3_results = phase4_prediction_3(df)

    # Phase 5: Diagnostic plots
    phase5_diagnostic_plots(df, p1_results)

    elapsed = time.time() - t0

    # Print summary
    print_summary(p1_results, p2_results, p3_results, elapsed)

    # Save results (exclude non-serializable objects)
    summary = {
        "experiment": "N131: Composite Predictor Validation",
        "elapsed_s": elapsed,
        "n_rows": len(df),
        "prediction_1": {k: v for k, v in p1_results.items() if not k.startswith("_")},
        "prediction_2": p2_results,
        "prediction_3": p3_results,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Saved summary.json")


if __name__ == "__main__":
    main()
