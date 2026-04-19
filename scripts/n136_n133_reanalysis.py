"""N136: Bounded reanalysis of N133 decoder-scale merge data.

Tests novel spectral-SV features against 12 evaluated cross-task pairs
to determine whether the task-family confound runs uniformly through all
computable features, or whether any feature shows within-cluster resolution.

Informed by N135 (C_k regime transition at C* ~ 0.30-0.40).

CPU-only.  Operates entirely on stored N133 JSON data — no model loading.

Output: sidecar/data/n136/
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N133_DIR = PROJECT_ROOT / "sidecar" / "data" / "n133"
AUDITS_DIR = N133_DIR / "pod_pull" / "audits"
MERGES_DIR = N133_DIR / "pod_pull" / "merges"
OUTPUT_DIR = PROJECT_ROOT / "sidecar" / "data" / "n136"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# N135 threshold
CK_THRESHOLD = 0.35  # midpoint of inferred [0.30, 0.40] regime transition


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adapter_profiles() -> dict:
    with open(AUDITS_DIR / "adapter_profiles.json") as f:
        return json.load(f)


def load_pair_alignments() -> dict:
    with open(AUDITS_DIR / "pair_alignment_full.json") as f:
        return json.load(f)


def load_w0_properties() -> dict:
    with open(AUDITS_DIR / "w0_properties.json") as f:
        return json.load(f)


def load_merge_outcomes() -> list[dict]:
    with open(MERGES_DIR / "merge_eval_summary.json") as f:
        data = json.load(f)
    return data["results"]


def load_bp5_reference() -> dict:
    with open(N133_DIR / "bp5_composite_risk.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build W_0 C_k lookup: (layer_idx, module) -> C_k
# ---------------------------------------------------------------------------

def build_ck_lookup(w0: dict) -> dict[tuple[int, str], float]:
    """Map (layer, module_short) -> C_k from w0_properties."""
    lookup = {}
    for key, props in w0.items():
        layer = props["layer"]
        module = props["module"]  # Q, K, V, O
        lookup[(layer, module)] = props["C_k"]
    return lookup


# ---------------------------------------------------------------------------
# Feature extraction: novel metrics per pair
# ---------------------------------------------------------------------------

def extract_pair_features(
    pair_key: str,
    pair_data: dict,
    profiles: dict,
    ck_lookup: dict[tuple[int, str], float],
) -> dict:
    """Compute novel features for one adapter pair."""
    adapter_a = pair_data["adapter_a"]
    adapter_b = pair_data["adapter_b"]
    per_layer = pair_data["per_layer"]

    # --- Per-layer alignment arrays by module ---
    layers_by_module: dict[str, list[dict]] = {"Q": [], "K": [], "V": [], "O": []}
    for entry in per_layer:
        mod = entry["module"]
        if mod in layers_by_module:
            layers_by_module[mod].append(entry)

    n_layers_total = 32  # Mistral-7B

    # --- 1. Existing baseline: mean_alignment ---
    mean_alignment = pair_data["mean_alignment"]

    # --- 2. C_k-gated alignment (N135-informed) ---
    # Only include layers where W_0 C_k > threshold
    gated_alignments = []
    gated_alignments_O = []
    for entry in per_layer:
        layer = entry["layer"]
        mod = entry["module"]
        ck = ck_lookup.get((layer, mod), 0.0)
        if ck > CK_THRESHOLD:
            gated_alignments.append(entry["alignment"])
            if mod == "O":
                gated_alignments_O.append(entry["alignment"])

    ck_gated_mean = float(np.mean(gated_alignments)) if gated_alignments else 0.0
    ck_gated_O_mean = float(np.mean(gated_alignments_O)) if gated_alignments_O else 0.0
    ck_gated_fraction = len(gated_alignments) / len(per_layer) if per_layer else 0.0

    # --- 3. O-module deep layers only (layers 16-31) ---
    O_deep = [e["alignment"] for e in layers_by_module["O"] if e["layer"] >= 16]
    O_deep_mean = float(np.mean(O_deep)) if O_deep else 0.0

    # --- 4. V-module alignment ---
    V_all = [e["alignment"] for e in layers_by_module["V"]]
    V_mean = float(np.mean(V_all)) if V_all else 0.0

    # --- 5. Q/K-excluded alignment (O + V only) ---
    OV_all = [e["alignment"] for e in per_layer if e["module"] in ("O", "V")]
    OV_mean = float(np.mean(OV_all)) if OV_all else 0.0

    # --- 6. High-percentile alignment (P90, P95) ---
    all_align = [e["alignment"] for e in per_layer]
    p90_align = float(np.percentile(all_align, 90)) if all_align else 0.0
    p95_align = float(np.percentile(all_align, 95)) if all_align else 0.0

    # --- 7. Alignment variance (cross-layer dispersion) ---
    align_std = float(np.std(all_align)) if all_align else 0.0
    align_cv = align_std / mean_alignment if mean_alignment > 1e-10 else 0.0

    # --- 8. Spectral shape divergence ---
    # KL divergence between adapter spectral energy profiles
    prof_a = profiles.get(adapter_a, {}).get("per_layer", [])
    prof_b = profiles.get(adapter_b, {}).get("per_layer", [])

    kl_divs = []
    slope_ratios = []
    erank_layer_diffs = []

    # Build lookup: (layer, module) -> singular_values for each adapter
    svs_a = {(e["layer"], e["module"]): e["singular_values"] for e in prof_a}
    svs_b = {(e["layer"], e["module"]): e["singular_values"] for e in prof_b}
    erank_a_lookup = {(e["layer"], e["module"]): e["erank"] for e in prof_a}
    erank_b_lookup = {(e["layer"], e["module"]): e["erank"] for e in prof_b}

    for key in svs_a:
        if key not in svs_b:
            continue
        sa = np.array(svs_a[key], dtype=np.float64)
        sb = np.array(svs_b[key], dtype=np.float64)

        # Normalize to energy distributions
        pa = sa**2 / (np.sum(sa**2) + 1e-20)
        pb = sb**2 / (np.sum(sb**2) + 1e-20)

        # Symmetric KL (Jensen-Shannon style)
        m = 0.5 * (pa + pb)
        kl_ab = np.sum(pa * np.log((pa + 1e-20) / (m + 1e-20)))
        kl_ba = np.sum(pb * np.log((pb + 1e-20) / (m + 1e-20)))
        jsd = 0.5 * (kl_ab + kl_ba)
        kl_divs.append(float(jsd))

        # Spectral slope ratio
        if sa[0] > 1e-10 and sb[0] > 1e-10 and len(sa) > 1 and len(sb) > 1:
            slope_a = np.log(sa[0] / (sa[-1] + 1e-10))
            slope_b = np.log(sb[0] / (sb[-1] + 1e-10))
            if slope_b > 1e-10:
                slope_ratios.append(max(slope_a / slope_b, slope_b / slope_a))

        # Per-layer erank difference
        ea = erank_a_lookup.get(key, 0)
        eb = erank_b_lookup.get(key, 0)
        if ea > 0 and eb > 0:
            erank_layer_diffs.append(abs(ea - eb))

    mean_jsd = float(np.mean(kl_divs)) if kl_divs else 0.0
    mean_slope_ratio = float(np.mean(slope_ratios)) if slope_ratios else 0.0
    mean_erank_layer_diff = float(np.mean(erank_layer_diffs)) if erank_layer_diffs else 0.0

    # --- 9. Adapter-level erank features ---
    erank_a = profiles.get(adapter_a, {}).get("mean_erank", 0)
    erank_b = profiles.get(adapter_b, {}).get("mean_erank", 0)
    min_erank = min(erank_a, erank_b)
    max_erank = max(erank_a, erank_b)
    erank_ratio = max_erank / min_erank if min_erank > 0.1 else 0.0
    erank_diff = abs(erank_a - erank_b)

    # --- 10. Frobenius norm ratio ---
    frob_a_layers = [e["frobenius_norm"] for e in prof_a]
    frob_b_layers = [e["frobenius_norm"] for e in prof_b]
    total_frob_a = float(np.sqrt(sum(f**2 for f in frob_a_layers))) if frob_a_layers else 0.0
    total_frob_b = float(np.sqrt(sum(f**2 for f in frob_b_layers))) if frob_b_layers else 0.0
    frob_ratio = (max(total_frob_a, total_frob_b) /
                  (min(total_frob_a, total_frob_b) + 1e-10))

    # --- 11. Composites ---
    inv_min_erank = 1.0 / min_erank if min_erank > 0.1 else 0.0
    ck_gated_O_x_inv_erank = ck_gated_O_mean * inv_min_erank
    O_deep_x_inv_erank = O_deep_mean * inv_min_erank
    jsd_x_erank_ratio = mean_jsd * erank_ratio

    return {
        # Baselines (already tested)
        "mean_alignment": mean_alignment,
        "inv_min_erank": inv_min_erank,
        # N135-informed
        "ck_gated_mean": ck_gated_mean,
        "ck_gated_O_mean": ck_gated_O_mean,
        "ck_gated_fraction": ck_gated_fraction,
        # Module-stratified
        "O_deep_mean": O_deep_mean,
        "V_mean": V_mean,
        "OV_mean": OV_mean,
        # Distributional
        "p90_align": p90_align,
        "p95_align": p95_align,
        "align_std": align_std,
        "align_cv": align_cv,
        # Spectral shape
        "mean_jsd": mean_jsd,
        "mean_slope_ratio": mean_slope_ratio,
        "mean_erank_layer_diff": mean_erank_layer_diff,
        # Adapter-level
        "erank_ratio": erank_ratio,
        "erank_diff": erank_diff,
        "frob_ratio": frob_ratio,
        # Composites
        "ck_gated_O_x_inv_erank": ck_gated_O_x_inv_erank,
        "O_deep_x_inv_erank": O_deep_x_inv_erank,
        "jsd_x_erank_ratio": jsd_x_erank_ratio,
    }


# ---------------------------------------------------------------------------
# Task-family classification
# ---------------------------------------------------------------------------

TASK_FAMILIES = {
    "sst2": "classification",
    "mnli": "classification",
    "squad": "extractive",
    "gsm8k": "generation",
    "code": "generation",
    "summarization": "generation",
}


def family_pair_label(task_a: str, task_b: str) -> str:
    """Label for the task-family pair."""
    fa = TASK_FAMILIES.get(task_a, "unknown")
    fb = TASK_FAMILIES.get(task_b, "unknown")
    return f"{min(fa, fb)}_x_{max(fa, fb)}"


# ---------------------------------------------------------------------------
# Analysis phases
# ---------------------------------------------------------------------------

def phase1_feature_extraction(
    pairs: dict, profiles: dict, ck_lookup: dict, outcomes: list[dict],
) -> pd.DataFrame:
    """Extract all features for evaluated cross-task pairs."""
    print("\n=== Phase 1: Feature Extraction ===")

    # Build outcome lookup
    outcome_map = {}
    for result in outcomes:
        if result["category"] != "cross_top":
            continue
        pair_key = result["pair"]
        # Compute max degradation across both tasks
        degs = [v for k, v in result.items() if k.startswith("degradation_") and isinstance(v, (int, float))]
        max_deg = max(degs) if degs else 0.0
        min_deg = min(degs) if degs else 0.0
        outcome_map[pair_key] = {
            "max_degradation": max_deg,
            "min_degradation": min_deg,
            "mean_degradation": float(np.mean(degs)) if degs else 0.0,
            "asymmetry": max_deg - min_deg,
            "task_a": result["task_a"],
            "task_b": result["task_b"],
        }

    rows = []
    for pair_key, pair_data in pairs.items():
        if pair_data["is_same_task"]:
            continue
        if pair_key not in outcome_map:
            continue

        features = extract_pair_features(pair_key, pair_data, profiles, ck_lookup)
        outcome = outcome_map[pair_key]
        family_label = family_pair_label(outcome["task_a"], outcome["task_b"])

        row = {
            "pair": pair_key,
            "task_a": outcome["task_a"],
            "task_b": outcome["task_b"],
            "family_pair": family_label,
            **features,
            **outcome,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Extracted {len(df)} evaluated cross-task pairs with {len(features)} features each")
    return df


def phase2_global_correlations(df: pd.DataFrame) -> dict:
    """Spearman correlations of all features vs max_degradation."""
    print("\n=== Phase 2: Global Correlations (all 12 pairs) ===")

    feature_cols = [c for c in df.columns if c not in
                    ("pair", "task_a", "task_b", "family_pair",
                     "max_degradation", "min_degradation", "mean_degradation", "asymmetry")]

    results = {}
    print(f"\n  {'Feature':<30} {'ρ':>7} {'p':>8} {'direction':>10}")
    print("  " + "-" * 58)

    for col in sorted(feature_cols):
        vals = df[col].values
        if np.std(vals) < 1e-12:
            continue
        rho, p = stats.spearmanr(vals, df["max_degradation"].values)
        direction = "+" if rho > 0 else "-"
        sig = "*" if p < 0.05 else " "
        print(f"  {col:<30} {rho:+7.3f} {p:8.4f} {direction:>10}{sig}")
        results[col] = {"rho": float(rho), "p": float(p)}

    return results


def phase3_family_residualization(df: pd.DataFrame) -> dict:
    """Test whether features predict degradation beyond task-family membership."""
    print("\n=== Phase 3: Family Residualization ===")

    feature_cols = [c for c in df.columns if c not in
                    ("pair", "task_a", "task_b", "family_pair",
                     "max_degradation", "min_degradation", "mean_degradation", "asymmetry")]

    # Family-only R²
    family_dummies = pd.get_dummies(df["family_pair"], drop_first=True, dtype=float)
    y = df["max_degradation"].values

    if family_dummies.shape[1] > 0:
        X_fam = np.column_stack([np.ones(len(y)), family_dummies.values])
        beta, _, _, _ = np.linalg.lstsq(X_fam, y, rcond=None)
        y_hat_fam = X_fam @ beta
        ss_res_fam = np.sum((y - y_hat_fam)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2_family = 1 - ss_res_fam / ss_tot if ss_tot > 0 else 0.0
    else:
        r2_family = 0.0
        y_hat_fam = np.full_like(y, y.mean())

    print(f"\n  Family-only R² = {r2_family:.4f}")
    print(f"  Residual variance available: {1 - r2_family:.4f}")

    # Family residuals
    residuals = y - y_hat_fam

    results = {"r2_family_only": float(r2_family)}
    print(f"\n  {'Feature':<30} {'ΔR²':>8} {'resid ρ':>9} {'resid p':>9}")
    print("  " + "-" * 60)

    for col in sorted(feature_cols):
        vals = df[col].values
        if np.std(vals) < 1e-12:
            continue

        # Incremental R²: add feature to family model
        X_aug = np.column_stack([X_fam, vals]) if family_dummies.shape[1] > 0 else np.column_stack([np.ones(len(y)), vals])
        beta_aug, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat_aug = X_aug @ beta_aug
        ss_res_aug = np.sum((y - y_hat_aug)**2)
        r2_aug = 1 - ss_res_aug / ss_tot if ss_tot > 0 else 0.0
        delta_r2 = r2_aug - r2_family

        # Correlation with family residuals
        rho_resid, p_resid = stats.spearmanr(vals, residuals)

        sig = "*" if delta_r2 > 0.03 else " "
        print(f"  {col:<30} {delta_r2:+8.4f} {rho_resid:+9.3f} {p_resid:9.4f}{sig}")
        results[col] = {
            "delta_r2": float(delta_r2),
            "rho_residual": float(rho_resid),
            "p_residual": float(p_resid),
        }

    return results


def phase4_within_cluster(df: pd.DataFrame) -> dict:
    """Within-cluster ranking analysis."""
    print("\n=== Phase 4: Within-Cluster Resolution ===")

    feature_cols = [c for c in df.columns if c not in
                    ("pair", "task_a", "task_b", "family_pair",
                     "max_degradation", "min_degradation", "mean_degradation", "asymmetry")]

    clusters = df.groupby("family_pair")
    results = {}

    for cluster_name, cluster_df in clusters:
        n = len(cluster_df)
        if n < 4:
            print(f"\n  Cluster '{cluster_name}': n={n} (too small, skipping)")
            continue

        print(f"\n  Cluster '{cluster_name}': n={n}")
        deg_range = cluster_df["max_degradation"].max() - cluster_df["max_degradation"].min()
        print(f"    Degradation range: [{cluster_df['max_degradation'].min():.3f}, "
              f"{cluster_df['max_degradation'].max():.3f}] (span={deg_range:.3f})")

        cluster_results = {"n": n, "deg_range": float(deg_range)}
        best_rho = 0.0
        best_feature = ""

        for col in sorted(feature_cols):
            vals = cluster_df[col].values
            if np.std(vals) < 1e-12:
                continue
            rho, p = stats.spearmanr(vals, cluster_df["max_degradation"].values)
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_feature = col
            cluster_results[col] = {"rho": float(rho), "p": float(p)}

        print(f"    Best within-cluster predictor: {best_feature} (ρ={best_rho:+.3f})")
        interesting = abs(best_rho) > 0.5 and n >= 5
        if interesting:
            print(f"    *** INTERESTING: {best_feature} shows within-cluster signal ***")
        cluster_results["best_feature"] = best_feature
        cluster_results["best_rho"] = float(best_rho)
        cluster_results["interesting"] = interesting
        results[cluster_name] = cluster_results

    return results


def phase5_asymmetric_degradation(df: pd.DataFrame) -> dict:
    """Test features against asymmetric degradation patterns."""
    print("\n=== Phase 5: Asymmetric Degradation ===")

    results = {}

    # Which task degrades more?
    print("\n  Per-pair degradation asymmetry:")
    for _, row in df.iterrows():
        print(f"    {row['pair']}: max={row['max_degradation']:.3f}, "
              f"min={row['min_degradation']:.3f}, asym={row['asymmetry']:.3f}")

    # Test: does erank predict which task suffers?
    # Low-erank adapter's task should degrade more
    rho_erank_asym, p_erank_asym = stats.spearmanr(
        df["erank_ratio"].values, df["asymmetry"].values
    ) if len(df) > 2 else (0.0, 1.0)

    rho_frob_asym, p_frob_asym = stats.spearmanr(
        df["frob_ratio"].values, df["asymmetry"].values
    ) if len(df) > 2 else (0.0, 1.0)

    print(f"\n  Erank ratio vs asymmetry: ρ={rho_erank_asym:+.3f}, p={p_erank_asym:.4f}")
    print(f"  Frob ratio vs asymmetry: ρ={rho_frob_asym:+.3f}, p={p_frob_asym:.4f}")

    results["erank_ratio_vs_asymmetry"] = {"rho": float(rho_erank_asym), "p": float(p_erank_asym)}
    results["frob_ratio_vs_asymmetry"] = {"rho": float(rho_frob_asym), "p": float(p_frob_asym)}

    return results


def phase6_dynamic_range(df: pd.DataFrame) -> dict:
    """Characterize the dynamic range problem for each feature."""
    print("\n=== Phase 6: Dynamic Range Characterization ===")

    feature_cols = [c for c in df.columns if c not in
                    ("pair", "task_a", "task_b", "family_pair",
                     "max_degradation", "min_degradation", "mean_degradation", "asymmetry")]

    results = {}
    print(f"\n  {'Feature':<30} {'min':>9} {'max':>9} {'range':>9} {'CV':>8}")
    print("  " + "-" * 68)

    for col in sorted(feature_cols):
        vals = df[col].values
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        v_range = v_max - v_min
        v_mean = float(np.mean(vals))
        cv = float(np.std(vals) / v_mean) if abs(v_mean) > 1e-10 else 0.0
        print(f"  {col:<30} {v_min:9.4f} {v_max:9.4f} {v_range:9.4f} {cv:8.3f}")
        results[col] = {
            "min": v_min, "max": v_max, "range": v_range, "cv": cv,
        }

    return results


def phase7_verdict(
    global_corr: dict,
    family_resid: dict,
    within_cluster: dict,
) -> str:
    """Synthesize findings."""
    print("\n=== Phase 7: Verdict ===\n")

    # Check if any feature exceeds ΔR² > 0.03 after residualization
    exceeds_threshold = []
    for feat, res in family_resid.items():
        if feat == "r2_family_only":
            continue
        if isinstance(res, dict) and res.get("delta_r2", 0) > 0.03:
            exceeds_threshold.append((feat, res["delta_r2"]))

    # Check if any cluster has an interesting feature
    interesting_clusters = []
    for cluster_name, cluster_res in within_cluster.items():
        if isinstance(cluster_res, dict) and cluster_res.get("interesting", False):
            interesting_clusters.append(
                (cluster_name, cluster_res["best_feature"], cluster_res["best_rho"])
            )

    lines = []

    if not exceeds_threshold and not interesting_clusters:
        lines.append(
            "P-null CONFIRMED: No feature exceeds ΔR² > 0.03 after family "
            "residualization, and no feature shows within-cluster ρ > 0.5 in any "
            "cluster with n ≥ 5."
        )
        lines.append("")
        lines.append(
            "The task-family confound runs uniformly through all spectral-SV features "
            "computable from the N133 data, including N135-informed C_k-gated variants, "
            "spectral shape divergence (JSD), module-stratified depth-weighted metrics, "
            "and erank/Frobenius asymmetry features."
        )
        lines.append("")
        lines.append(
            "This strengthens the paper's negative claim and confirms N134's design "
            "constraint C2 (task-family non-partition) is necessary."
        )
    elif exceeds_threshold:
        lines.append("P-alt TRIGGERED: Feature(s) exceeding ΔR² > 0.03 after residualization:")
        for feat, dr2 in exceeds_threshold:
            lines.append(f"  - {feat}: ΔR² = {dr2:.4f}")
        lines.append("")
        lines.append(
            "CAUTION: With n=12, this may be overfitting. These features enter N134's "
            "exploratory analysis list but do not change pre-registered predictions."
        )
    else:
        lines.append("MIXED: No feature exceeds ΔR² threshold, but within-cluster signal found:")
        for cluster, feat, rho in interesting_clusters:
            lines.append(f"  - Cluster '{cluster}': {feat} (ρ={rho:+.3f})")
        lines.append("")
        lines.append(
            "Within-cluster resolution exists but doesn't survive family residualization. "
            "Worth monitoring in N134 but not strong evidence for a spectral-SV predictor."
        )

    verdict = "\n".join(lines)
    print(verdict)
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("N136: Bounded Reanalysis of N133 Decoder-Scale Merge Data")
    print("=" * 60)

    # Load data
    profiles = load_adapter_profiles()
    pairs = load_pair_alignments()
    w0 = load_w0_properties()
    outcomes = load_merge_outcomes()
    ck_lookup = build_ck_lookup(w0)

    print(f"  Loaded {len(profiles)} adapter profiles")
    print(f"  Loaded {len(pairs)} pair alignments")
    print(f"  Loaded {len(w0)} W_0 layer properties")
    print(f"  Loaded {len(outcomes)} merge outcomes")
    print(f"  C_k threshold: {CK_THRESHOLD}")

    # Phase 1: Extract features
    df = phase1_feature_extraction(pairs, profiles, ck_lookup, outcomes)

    if len(df) == 0:
        print("\nERROR: No evaluated cross-task pairs found. Check data paths.")
        return

    df.to_csv(OUTPUT_DIR / "n136_features.csv", index=False)

    # Phase 2: Global correlations
    global_corr = phase2_global_correlations(df)

    # Phase 3: Family residualization
    family_resid = phase3_family_residualization(df)

    # Phase 4: Within-cluster analysis
    within_cluster = phase4_within_cluster(df)

    # Phase 5: Asymmetric degradation
    asymmetric = phase5_asymmetric_degradation(df)

    # Phase 6: Dynamic range
    dynamic_range = phase6_dynamic_range(df)

    # Phase 7: Verdict
    verdict = phase7_verdict(global_corr, family_resid, within_cluster)

    # Save all results
    all_results = {
        "experiment": "N136",
        "description": "Bounded reanalysis of N133 with novel features",
        "ck_threshold": CK_THRESHOLD,
        "n_pairs": len(df),
        "global_correlations": global_corr,
        "family_residualization": family_resid,
        "within_cluster": within_cluster,
        "asymmetric_degradation": asymmetric,
        "dynamic_range": dynamic_range,
        "verdict": verdict,
        "elapsed_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "n136_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print(f"  Results saved to {OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
