#!/usr/bin/env python3
"""
N129: Portfolio Rank Collapse Probe

Does pairwise triage suffice at larger pool sizes, or does additive merging
of retained adapters suppress task-specific spectral structure?

Motivation (Technical Report S7.6, Skorobogat et al. 2025):
    When merging k task vectors by Task Arithmetic, the ratio of spectral
    energy in the common shared subspace to the task-specific subspace grows
    linearly with k.  A set of pairwise-retained adapters may therefore
    produce a merged ensemble that suppresses task-specific information,
    regardless of pairwise compatibility.

Method:
    1. Load retained adapter pools from field trial inventory preflight data
    2. For each pool, compute single-adapter baseline skewness per layer
    3. Sweep pool sizes k = 1..K_max, measuring relative skewness rho(k)
    4. Fit linear model rho(k) = beta_0 + beta_1 * k
    5. Test H_null (beta_1 <= 0) vs H_linear (beta_1 > 0)
    6. Test H_selection: retained pools vs full pools

Reference: sidecar/notes/N129_rank_collapse_probe.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_MAX_K = 10
DEFAULT_MAX_COMBOS = 20
NOISE_FLOOR_RATIO = 1e-10

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdapterInfo:
    adapter_id: str
    path: Path
    backbone: str
    task: str
    role: str          # "retained", "excluded", "full_pool"
    rank: int = 0
    alpha: float = 0.0
    target_modules: list[str] = field(default_factory=list)


@dataclass
class InventoryPool:
    inventory_id: str
    backbone: str
    pool_type: str              # "same_task", "mixed_task", "mixed"
    retained: list[AdapterInfo] = field(default_factory=list)
    full: list[AdapterInfo] = field(default_factory=list)
    h_selection_testable: bool = False


@dataclass
class CollapsePoint:
    k: int
    rho_mean: float
    rho_std: float
    epsilon_mean: float
    epsilon_std: float
    n_combos: int
    sampling_strategy: str      # "exhaustive" or "sampled_20"


@dataclass
class LinearFit:
    beta_0: float
    beta_1: float
    beta_1_se: float
    beta_1_ci_lower: float
    beta_1_ci_upper: float
    p_one_sided: float
    r_squared: float


@dataclass
class SelectionResult:
    delta_rho_by_k: dict[int, float]
    direction_positive_at_k3_plus: bool | None


@dataclass
class InventoryResult:
    inventory_id: str
    backbone: str
    n_retained: int
    n_full: int
    pool_type: str
    h_selection_testable: bool
    n_intersection_layers: int
    skewness_baseline: float
    collapse_curve: list[CollapsePoint]
    linear_fit: LinearFit | None
    k_collapse: int | None
    k_collapse_extrap: float | None
    h_selection: SelectionResult | None
    by_module_type: dict[str, dict]
    v_module_excess: bool


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load LoRA A and B matrices from a PEFT adapter directory.

    Handles both safetensors and bin formats.
    Returns dict mapping layer_key -> {"A": ndarray, "B": ndarray, "scaling": float}.
    """
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    config_path = adapter_dir / "adapter_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No adapter_config.json at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    alpha = config.get("lora_alpha", config.get("r", 8))
    r = config.get("r", 8)
    scaling = alpha / r if r > 0 else 1.0

    weights = {}

    if safetensors_path.exists():
        with safe_open(str(safetensors_path), framework="numpy") as f:
            weights = _extract_lora_pairs(f, scaling, use_safetensors=True)
    elif bin_path.exists():
        import torch
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        weights = _extract_lora_pairs(state_dict, scaling, use_safetensors=False)
    else:
        raise FileNotFoundError(
            f"No adapter weights at {safetensors_path} or {bin_path}"
        )

    return weights


def _extract_lora_pairs(
    source, scaling: float, use_safetensors: bool
) -> dict[str, dict[str, np.ndarray]]:
    """Extract matched LoRA A/B pairs from a weight source."""
    if use_safetensors:
        keys = list(source.keys())
        get_fn = lambda k: source.get_tensor(k).astype(np.float64)
    else:
        keys = list(source.keys())
        get_fn = lambda k: source[k].numpy().astype(np.float64)

    a_keys = sorted(k for k in keys if "lora_A" in k)
    b_keys = sorted(k for k in keys if "lora_B" in k)

    weights = {}
    for a_key in a_keys:
        layer_key = a_key.replace(".lora_A.weight", "").replace(
            ".lora_A.default.weight", ""
        )
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in keys:
            for bk in b_keys:
                bk_layer = bk.replace(".lora_B.weight", "").replace(
                    ".lora_B.default.weight", ""
                )
                if bk_layer == layer_key:
                    b_key = bk
                    break

        if b_key not in keys:
            continue

        A = get_fn(a_key)
        B = get_fn(b_key)

        short_key = layer_key
        for prefix in ["base_model.model.", "base_model."]:
            if short_key.startswith(prefix):
                short_key = short_key[len(prefix):]

        weights[short_key] = {"A": A, "B": B, "scaling": scaling}

    return weights


def detect_module_type(layer_name: str) -> str:
    """Detect Q/K/V/O/MLP module type from layer name."""
    name_lower = layer_name.lower()
    if "q_lin" in name_lower or ".q_proj" in name_lower or ".query" in name_lower:
        return "q"
    if "k_lin" in name_lower or ".k_proj" in name_lower or ".key" in name_lower:
        return "k"
    if "v_lin" in name_lower or ".v_proj" in name_lower or ".value" in name_lower:
        return "v"
    if "out_lin" in name_lower or ".o_proj" in name_lower:
        return "o"
    if ".dense" in name_lower and ".output" in name_lower:
        return "o"
    if any(x in name_lower for x in ["lin1", "lin2", "intermediate", "mlp", "pre_classifier", "classifier"]):
        return "mlp"
    return "other"


# ---------------------------------------------------------------------------
# Inventory discovery
# ---------------------------------------------------------------------------

def discover_inventories(results_dir: Path) -> list[InventoryPool]:
    """Discover field trial inventories with retained adapter pools."""
    field_trials = REPO_ROOT / "field_trials"
    pools = []

    for inv_dir in sorted(field_trials.glob("inventory_*")):
        pool = _load_inventory(inv_dir)
        if pool is not None:
            pools.append(pool)

    return pools


def _load_inventory(inv_dir: Path) -> InventoryPool | None:
    """Load a single inventory's retained and full pools."""
    # Read manifest for backbone
    manifest_path = inv_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  Warning: no manifest.json in {inv_dir.name}, skipping")
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    backbone = manifest.get("backbone", "unknown")
    inventory_id = inv_dir.name

    # Find latest preflight summary
    preflight_files = sorted(inv_dir.glob("*/bundle/preflight_summary.json"))
    preflight_files += sorted(inv_dir.glob("preflight/bundle/preflight_summary.json"))
    if not preflight_files:
        print(f"  Warning: no preflight_summary.json in {inventory_id}, skipping")
        return None

    with open(preflight_files[-1]) as f:
        preflight = json.load(f)

    # Parse retained pairs to get unique retained adapter IDs
    retained_pairs = preflight.get("same_task_priority_pairs", [])
    excluded_sources = preflight.get("excluded_sources", [])
    excluded_ids = set()
    for ex in excluded_sources:
        # Format: "adapter_id: reason"
        excluded_ids.add(ex.split(":")[0].strip())

    retained_ids = set()
    for pair_str in retained_pairs:
        parts = pair_str.split(" \u00d7 ")
        for a in parts:
            retained_ids.add(a.strip())

    if len(retained_ids) < 2:
        print(f"  {inventory_id}: |R| = {len(retained_ids)} < 2, excluding")
        return None

    # Build adapter info from adapter_cache
    adapter_cache = inv_dir / "adapter_cache"
    if not adapter_cache.exists():
        print(f"  Warning: no adapter_cache in {inventory_id}, skipping")
        return None

    retained_adapters = []
    full_adapters = []

    for adapter_dir in sorted(adapter_cache.iterdir()):
        if not adapter_dir.is_dir():
            continue

        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            continue

        # Check for weight files
        has_weights = (
            (adapter_dir / "adapter_model.safetensors").exists()
            or (adapter_dir / "adapter_model.bin").exists()
        )
        if not has_weights:
            continue

        with open(config_path) as f:
            config = json.load(f)

        adapter_id = adapter_dir.name
        task = _infer_task_from_adapter_id(adapter_id)

        info = AdapterInfo(
            adapter_id=adapter_id,
            path=adapter_dir,
            backbone=backbone,
            task=task,
            role="retained" if adapter_id in retained_ids else "full_pool",
            rank=config.get("r", 0),
            alpha=config.get("lora_alpha", 0),
            target_modules=config.get("target_modules", []),
        )

        full_adapters.append(info)
        if adapter_id in retained_ids:
            retained_adapters.append(info)

    if len(retained_adapters) < 2:
        print(f"  {inventory_id}: found only {len(retained_adapters)} retained adapters with weights, excluding")
        return None

    # Determine pool type
    tasks = set(a.task for a in retained_adapters)
    if len(tasks) == 1:
        pool_type = "same_task"
    elif all(t == "unknown" for t in tasks):
        pool_type = "mixed"
    else:
        pool_type = "mixed_task"

    h_selection_testable = len(full_adapters) >= len(retained_adapters) + 2

    pool = InventoryPool(
        inventory_id=inventory_id,
        backbone=backbone,
        pool_type=pool_type,
        retained=retained_adapters,
        full=full_adapters,
        h_selection_testable=h_selection_testable,
    )

    print(f"  {inventory_id}: |R|={len(retained_adapters)}, |F|={len(full_adapters)}, "
          f"type={pool_type}, H_sel={'yes' if h_selection_testable else 'no'}")

    return pool


def _infer_task_from_adapter_id(adapter_id: str) -> str:
    """Infer task from adapter directory name heuristic."""
    name = adapter_id.lower()
    if "sst2" in name or "sst-2" in name:
        return "sst2"
    if "ag_news" in name or "agnews" in name:
        return "ag_news"
    if "irony" in name:
        return "irony"
    if "emotion" in name:
        return "emotion"
    if "hate" in name:
        return "hate"
    if "imdb" in name:
        return "imdb"
    if "mnli" in name:
        return "mnli"
    if "qnli" in name:
        return "qnli"
    if "mrpc" in name:
        return "mrpc"
    if "rte" in name:
        return "rte"
    return "unknown"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_baseline(
    adapter_weights: dict[str, dict[str, dict[str, np.ndarray]]],
    intersection_layers: list[str],
) -> dict[str, float]:
    """Compute single-adapter baseline skewness per layer.

    Returns {layer_name: mean_skewness_across_adapters}.
    """
    baselines = {}
    for layer in intersection_layers:
        skews = []
        for aid, weights in adapter_weights.items():
            w = weights[layer]
            delta_W = w["scaling"] * (w["B"].astype(np.float64) @ w["A"].astype(np.float64))
            s = np.linalg.svd(delta_W, compute_uv=False)
            s = s[s >= NOISE_FLOOR_RATIO * s[0]] if s[0] > 0 else s
            if len(s) > 0 and np.mean(s) > 0:
                skews.append(s[0] / np.mean(s))
        baselines[layer] = float(np.mean(skews)) if skews else 1.0
    return baselines


def run_collapse_curve(
    adapter_weights: dict[str, dict[str, dict[str, np.ndarray]]],
    intersection_layers: list[str],
    baselines: dict[str, float],
    max_k: int,
    rng: np.random.Generator,
    max_combos: int = DEFAULT_MAX_COMBOS,
) -> list[CollapsePoint]:
    """Run collapse curve analysis for k = 1..K_max."""
    adapter_ids = sorted(adapter_weights.keys())
    n = len(adapter_ids)
    K_max = min(max_k, n)

    # Grand baseline = mean across layers
    grand_baseline = float(np.mean(list(baselines.values())))

    points = []

    # k = 1: by construction rho = 1.0
    # Compute epsilon at k=1
    eps_vals = []
    for layer in intersection_layers:
        for aid in adapter_ids:
            w = adapter_weights[aid][layer]
            delta_W = w["scaling"] * (w["B"].astype(np.float64) @ w["A"].astype(np.float64))
            s = np.linalg.svd(delta_W, compute_uv=False)
            s = s[s >= NOISE_FLOOR_RATIO * s[0]] if s[0] > 0 else s
            if len(s) > 0 and np.sum(s**2) > 0:
                eps_vals.append(s[0]**2 / np.sum(s**2))
    eps_1 = float(np.mean(eps_vals)) if eps_vals else 0.0

    points.append(CollapsePoint(
        k=1, rho_mean=1.0, rho_std=0.0,
        epsilon_mean=eps_1, epsilon_std=0.0,
        n_combos=n, sampling_strategy="individual",
    ))

    # k = 2..K_max
    for k in range(2, K_max + 1):
        combos = list(combinations(adapter_ids, k))

        if len(combos) > max_combos:
            indices = rng.choice(len(combos), size=max_combos, replace=False)
            combos = [combos[i] for i in sorted(indices)]
            sampling = "sampled_20"
        else:
            sampling = "exhaustive"

        skew_by_combo = []
        eps_by_combo = []

        for combo in combos:
            layer_skews = []
            layer_eps = []
            for layer in intersection_layers:
                # Get shape from first adapter
                w0 = adapter_weights[combo[0]][layer]
                d_out = w0["B"].shape[0]
                d_in = w0["A"].shape[1]

                # Accumulate sum (memory-efficient)
                delta_W_merged = np.zeros((d_out, d_in), dtype=np.float64)
                for aid in combo:
                    w = adapter_weights[aid][layer]
                    delta_W_merged += w["scaling"] * (
                        w["B"].astype(np.float64) @ w["A"].astype(np.float64)
                    )

                s = np.linalg.svd(delta_W_merged, compute_uv=False)
                s = s[s >= NOISE_FLOOR_RATIO * s[0]] if s[0] > 0 else s

                if len(s) > 0 and np.mean(s) > 0:
                    layer_skews.append(s[0] / np.mean(s))
                if len(s) > 0 and np.sum(s**2) > 0:
                    layer_eps.append(s[0]**2 / np.sum(s**2))

            if layer_skews:
                skew_by_combo.append(float(np.mean(layer_skews)))
            if layer_eps:
                eps_by_combo.append(float(np.mean(layer_eps)))

        skew_k = float(np.mean(skew_by_combo)) if skew_by_combo else 0.0
        skew_k_std = float(np.std(skew_by_combo, ddof=1)) if len(skew_by_combo) > 1 else 0.0
        eps_k = float(np.mean(eps_by_combo)) if eps_by_combo else 0.0
        eps_k_std = float(np.std(eps_by_combo, ddof=1)) if len(eps_by_combo) > 1 else 0.0

        rho_k = skew_k / grand_baseline if grand_baseline > 0 else 0.0
        rho_k_std = skew_k_std / grand_baseline if grand_baseline > 0 else 0.0

        points.append(CollapsePoint(
            k=k, rho_mean=rho_k, rho_std=rho_k_std,
            epsilon_mean=eps_k, epsilon_std=eps_k_std,
            n_combos=len(combos), sampling_strategy=sampling,
        ))

    return points


def fit_linear_model(points: list[CollapsePoint]) -> LinearFit | None:
    """Fit rho(k) = beta_0 + beta_1 * k and compute hypothesis statistics."""
    if len(points) < 2:
        return None

    k_arr = np.array([p.k for p in points], dtype=float)
    rho_arr = np.array([p.rho_mean for p in points], dtype=float)

    result = stats.linregress(k_arr, rho_arr)

    p_one_sided = result.pvalue / 2 if result.slope > 0 else 1.0 - result.pvalue / 2

    df = len(k_arr) - 2
    if df > 0:
        t_crit = stats.t.ppf(0.975, df=df)
        ci_lower = result.slope - t_crit * result.stderr
        ci_upper = result.slope + t_crit * result.stderr
    else:
        ci_lower = result.slope
        ci_upper = result.slope

    return LinearFit(
        beta_0=float(result.intercept),
        beta_1=float(result.slope),
        beta_1_se=float(result.stderr),
        beta_1_ci_lower=float(ci_lower),
        beta_1_ci_upper=float(ci_upper),
        p_one_sided=float(p_one_sided),
        r_squared=float(result.rvalue**2),
    )


def compute_k_collapse(
    points: list[CollapsePoint], fit: LinearFit | None, threshold: float = 2.0
) -> tuple[int | None, float | None]:
    """Compute k_collapse from observed data and extrapolation."""
    for p in points:
        if p.rho_mean > threshold:
            return int(p.k), None

    if fit is not None and fit.beta_1 > 0:
        k_extrap = (threshold - fit.beta_0) / fit.beta_1
        return None, float(k_extrap)

    return None, None


def run_h_selection(
    retained_points: list[CollapsePoint],
    full_adapter_weights: dict[str, dict[str, dict[str, np.ndarray]]],
    intersection_layers: list[str],
    baselines: dict[str, float],
    max_k: int,
    rng: np.random.Generator,
) -> SelectionResult | None:
    """Compare collapse in retained pool vs full pool."""
    full_points = run_collapse_curve(
        full_adapter_weights, intersection_layers, baselines,
        max_k=max_k, rng=rng,
    )

    delta_rho = {}
    for rp in retained_points:
        for fp in full_points:
            if rp.k == fp.k and rp.k >= 2:
                delta_rho[rp.k] = fp.rho_mean - rp.rho_mean
                break

    if not delta_rho:
        return SelectionResult(delta_rho_by_k={}, direction_positive_at_k3_plus=None)

    k3_plus = {k: v for k, v in delta_rho.items() if k >= 3}
    if k3_plus:
        direction = all(v > 0 for v in k3_plus.values())
    else:
        direction = None

    return SelectionResult(
        delta_rho_by_k=delta_rho,
        direction_positive_at_k3_plus=direction,
    )


def run_module_breakdown(
    adapter_weights: dict[str, dict[str, dict[str, np.ndarray]]],
    intersection_layers: list[str],
    max_k: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Run collapse analysis per module type."""
    by_module: dict[str, list[str]] = {}
    for layer in intersection_layers:
        mod = detect_module_type(layer)
        by_module.setdefault(mod, []).append(layer)

    results = {}
    for mod, layers in sorted(by_module.items()):
        if not layers:
            continue

        baselines = compute_baseline(adapter_weights, layers)
        points = run_collapse_curve(
            adapter_weights, layers, baselines,
            max_k=max_k, rng=rng,
        )
        fit = fit_linear_model(points)
        k_obs, k_ext = compute_k_collapse(points, fit)

        results[mod] = {
            "beta_1": fit.beta_1 if fit else None,
            "beta_1_se": fit.beta_1_se if fit else None,
            "k_collapse": k_obs,
            "k_collapse_extrap": k_ext,
            "n_layers": len(layers),
        }

    return results


# ---------------------------------------------------------------------------
# Layer intersection
# ---------------------------------------------------------------------------

def find_intersection_layers(
    adapter_weights: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[str]:
    """Find layers modified by ALL adapters in the pool."""
    if not adapter_weights:
        return []
    layer_sets = [set(w.keys()) for w in adapter_weights.values()]
    intersection = sorted(set.intersection(*layer_sets))
    return intersection


# ---------------------------------------------------------------------------
# Analyze one inventory
# ---------------------------------------------------------------------------

def analyze_inventory(
    pool: InventoryPool,
    max_k: int,
    rng: np.random.Generator,
) -> InventoryResult | None:
    """Run full collapse analysis on one inventory."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {pool.inventory_id}")
    print(f"  Backbone: {pool.backbone}, |R|={len(pool.retained)}, |F|={len(pool.full)}")

    # Load retained adapter weights
    retained_weights: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for adapter in pool.retained:
        print(f"  Loading {adapter.adapter_id} (r={adapter.rank})...", end=" ", flush=True)
        try:
            w = load_adapter_weights(adapter.path)
            retained_weights[adapter.adapter_id] = w
            print(f"{len(w)} layers")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(retained_weights) < 2:
        print(f"  Only {len(retained_weights)} adapters loaded, skipping")
        return None

    # Find intersection layers
    intersection = find_intersection_layers(retained_weights)
    print(f"  Intersection layers: {len(intersection)}")
    if not intersection:
        print(f"  No shared layers, skipping")
        return None

    # Verify layer shapes are compatible (same d_out, d_in)
    compatible_layers = []
    for layer in intersection:
        shapes = set()
        for aid, w in retained_weights.items():
            lw = w[layer]
            shapes.add((lw["B"].shape[0], lw["A"].shape[1]))
        if len(shapes) == 1:
            compatible_layers.append(layer)
        else:
            print(f"    Warning: shape mismatch on {layer}: {shapes}, excluding")

    intersection = compatible_layers
    print(f"  Compatible layers: {len(intersection)}")
    if not intersection:
        print(f"  No compatible layers, skipping")
        return None

    # Compute baseline
    print(f"  Computing baseline...")
    baselines = compute_baseline(retained_weights, intersection)
    grand_baseline = float(np.mean(list(baselines.values())))
    print(f"  Grand baseline skewness: {grand_baseline:.4f}")

    # Verification check 2: baseline range
    for layer, bsl in baselines.items():
        if bsl < 1.0 or bsl > 30.0:
            print(f"    Warning: unusual baseline skewness {bsl:.2f} at {layer}")

    # Run collapse curve
    print(f"  Running collapse curve (k=1..{min(max_k, len(retained_weights))})...")
    points = run_collapse_curve(
        retained_weights, intersection, baselines,
        max_k=max_k, rng=rng,
    )
    for p in points:
        print(f"    k={p.k}: rho={p.rho_mean:.4f} +/- {p.rho_std:.4f}, "
              f"eps={p.epsilon_mean:.4f}, combos={p.n_combos} ({p.sampling_strategy})")

    # Verification check 1: rho(1) = 1.0
    if points and abs(points[0].rho_mean - 1.0) > 0.001:
        print(f"  WARNING: rho(1) = {points[0].rho_mean:.6f} != 1.0 -- normalization error!")

    # Verification check 4: epsilon bounded
    for p in points:
        if p.epsilon_mean > 1.0 or p.epsilon_mean < 0.0:
            print(f"  WARNING: epsilon({p.k}) = {p.epsilon_mean:.4f} out of [0,1]!")

    # Fit linear model
    fit = fit_linear_model(points)
    if fit:
        print(f"  Linear fit: rho = {fit.beta_0:.4f} + {fit.beta_1:.4f} * k")
        print(f"    beta_1 SE={fit.beta_1_se:.4f}, p(one-sided)={fit.p_one_sided:.4f}")
        print(f"    95% CI: [{fit.beta_1_ci_lower:.4f}, {fit.beta_1_ci_upper:.4f}]")
        print(f"    R^2 = {fit.r_squared:.4f}")

        # Verification check 5: fit near (1, 1.0)
        fit_at_1 = fit.beta_0 + fit.beta_1
        if abs(fit_at_1 - 1.0) > 0.1:
            print(f"  WARNING: fit(k=1) = {fit_at_1:.4f}, expected ~1.0")

    # k_collapse
    k_obs, k_ext = compute_k_collapse(points, fit)
    if k_obs:
        print(f"  k_collapse (observed): {k_obs}")
    elif k_ext:
        print(f"  k_collapse (extrapolated): {k_ext:.1f}")
    else:
        print(f"  k_collapse: none predicted")

    # H_selection
    h_sel = None
    if pool.h_selection_testable:
        print(f"  Running H_selection comparison...")
        full_weights: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for adapter in pool.full:
            if adapter.adapter_id not in retained_weights:
                try:
                    w = load_adapter_weights(adapter.path)
                    full_weights[adapter.adapter_id] = w
                except Exception as e:
                    print(f"    Failed to load {adapter.adapter_id}: {e}")

        # Merge retained + non-retained for full pool
        all_weights = {**retained_weights, **full_weights}
        # Recalculate intersection for full pool
        full_intersection = find_intersection_layers(all_weights)
        # Use the retained intersection (stricter)
        shared_int = [l for l in intersection if l in full_intersection]

        if len(all_weights) > len(retained_weights) and shared_int:
            full_baselines = compute_baseline(all_weights, shared_int)
            h_sel = run_h_selection(
                points, all_weights, shared_int, full_baselines,
                max_k=max_k, rng=rng,
            )
            if h_sel:
                for k, dr in sorted(h_sel.delta_rho_by_k.items()):
                    print(f"    delta_rho(k={k}): {dr:+.4f}")
                print(f"    Direction positive at k>=3: {h_sel.direction_positive_at_k3_plus}")

    # Module breakdown
    print(f"  Running module-type breakdown...")
    mod_results = run_module_breakdown(retained_weights, intersection, max_k, rng)
    for mod, mr in sorted(mod_results.items()):
        b1 = mr["beta_1"]
        b1_str = f"{b1:.4f}" if b1 is not None else "n/a"
        print(f"    {mod}: beta_1={b1_str}, n_layers={mr['n_layers']}")

    # V-module excess
    agg_beta = fit.beta_1 if fit else 0.0
    v_beta = mod_results.get("v", {}).get("beta_1")
    v_excess = v_beta is not None and v_beta > agg_beta

    return InventoryResult(
        inventory_id=pool.inventory_id,
        backbone=pool.backbone,
        n_retained=len(retained_weights),
        n_full=len(pool.full),
        pool_type=pool.pool_type,
        h_selection_testable=pool.h_selection_testable,
        n_intersection_layers=len(intersection),
        skewness_baseline=grand_baseline,
        collapse_curve=points,
        linear_fit=fit,
        k_collapse=k_obs,
        k_collapse_extrap=k_ext,
        h_selection=h_sel,
        by_module_type=mod_results,
        v_module_excess=v_excess,
    )


# ---------------------------------------------------------------------------
# Combined backbone pools
# ---------------------------------------------------------------------------

def build_combined_pools(
    inventories: list[InventoryPool],
) -> dict[str, list[AdapterInfo]]:
    """Build backbone-specific combined pools from all inventories."""
    by_backbone: dict[str, list[AdapterInfo]] = {}
    seen: dict[str, set[str]] = {}

    for pool in inventories:
        bb = pool.backbone
        if bb not in by_backbone:
            by_backbone[bb] = []
            seen[bb] = set()

        for adapter in pool.retained:
            if adapter.adapter_id not in seen[bb]:
                by_backbone[bb].append(adapter)
                seen[bb].add(adapter.adapter_id)

    return by_backbone


def analyze_combined_pool(
    backbone: str,
    adapters: list[AdapterInfo],
    max_k: int,
    rng: np.random.Generator,
) -> dict | None:
    """Analyze a combined backbone pool."""
    print(f"\n{'='*60}")
    print(f"Combined pool: {backbone} ({len(adapters)} adapters)")

    if len(adapters) < 2:
        print(f"  Only {len(adapters)} adapters, skipping")
        return None

    weights: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for adapter in adapters:
        try:
            w = load_adapter_weights(adapter.path)
            weights[adapter.adapter_id] = w
        except Exception as e:
            print(f"  Failed to load {adapter.adapter_id}: {e}")

    if len(weights) < 2:
        return None

    intersection = find_intersection_layers(weights)
    compatible = []
    for layer in intersection:
        shapes = set()
        for w in weights.values():
            lw = w[layer]
            shapes.add((lw["B"].shape[0], lw["A"].shape[1]))
        if len(shapes) == 1:
            compatible.append(layer)
    intersection = compatible

    print(f"  Intersection layers: {len(intersection)}")
    if not intersection:
        return None

    baselines = compute_baseline(weights, intersection)
    grand_baseline = float(np.mean(list(baselines.values())))
    print(f"  Grand baseline: {grand_baseline:.4f}")

    points = run_collapse_curve(weights, intersection, baselines, max_k=max_k, rng=rng)
    for p in points:
        print(f"    k={p.k}: rho={p.rho_mean:.4f} +/- {p.rho_std:.4f}")

    fit = fit_linear_model(points)
    k_obs, k_ext = compute_k_collapse(points, fit)

    return {
        "pool_type": "combined_backbone",
        "backbone": backbone,
        "n_adapters": len(weights),
        "n_inventories_pooled": len(set(
            a.path.parent.parent.name for a in adapters if a.adapter_id in weights
        )),
        "n_intersection_layers": len(intersection),
        "skewness_baseline": round(grand_baseline, 6),
        "collapse_curve": {
            "k_values": [p.k for p in points],
            "rho_mean": [round(p.rho_mean, 6) for p in points],
            "rho_std": [round(p.rho_std, 6) for p in points],
            "epsilon_mean": [round(p.epsilon_mean, 6) for p in points],
            "epsilon_std": [round(p.epsilon_std, 6) for p in points],
            "n_combos_evaluated": [p.n_combos for p in points],
        },
        "linear_fit": {
            "beta_0": round(fit.beta_0, 6),
            "beta_1": round(fit.beta_1, 6),
            "beta_1_se": round(fit.beta_1_se, 6),
            "beta_1_ci_lower": round(fit.beta_1_ci_lower, 6),
            "beta_1_ci_upper": round(fit.beta_1_ci_upper, 6),
            "p_one_sided": round(fit.p_one_sided, 6),
            "r_squared": round(fit.r_squared, 6),
        } if fit else None,
        "k_collapse": k_obs,
        "k_collapse_extrap": round(k_ext, 2) if k_ext else None,
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _inventory_result_to_dict(r: InventoryResult) -> dict:
    """Serialize an InventoryResult."""
    return {
        "inventory_id": r.inventory_id,
        "backbone": r.backbone,
        "n_retained": r.n_retained,
        "n_full": r.n_full,
        "pool_type": r.pool_type,
        "h_selection_testable": r.h_selection_testable,
        "n_intersection_layers": r.n_intersection_layers,
        "skewness_baseline": round(r.skewness_baseline, 6),
        "collapse_curve": {
            "k_values": [p.k for p in r.collapse_curve],
            "rho_mean": [round(p.rho_mean, 6) for p in r.collapse_curve],
            "rho_std": [round(p.rho_std, 6) for p in r.collapse_curve],
            "epsilon_mean": [round(p.epsilon_mean, 6) for p in r.collapse_curve],
            "epsilon_std": [round(p.epsilon_std, 6) for p in r.collapse_curve],
            "n_combos_evaluated": [p.n_combos for p in r.collapse_curve],
        },
        "linear_fit": {
            "beta_0": round(r.linear_fit.beta_0, 6),
            "beta_1": round(r.linear_fit.beta_1, 6),
            "beta_1_se": round(r.linear_fit.beta_1_se, 6),
            "beta_1_ci_lower": round(r.linear_fit.beta_1_ci_lower, 6),
            "beta_1_ci_upper": round(r.linear_fit.beta_1_ci_upper, 6),
            "p_value_one_sided": round(r.linear_fit.p_one_sided, 6),
            "r_squared": round(r.linear_fit.r_squared, 6),
        } if r.linear_fit else None,
        "k_collapse": r.k_collapse,
        "k_collapse_extrap": round(r.k_collapse_extrap, 2) if r.k_collapse_extrap else None,
        "h_selection": {
            "delta_rho_by_k": {str(k): round(v, 6) for k, v in r.h_selection.delta_rho_by_k.items()},
            "direction_positive_at_k3_plus": r.h_selection.direction_positive_at_k3_plus,
        } if r.h_selection else None,
        "by_module_type": {
            mod: {
                "beta_1": round(d["beta_1"], 6) if d["beta_1"] is not None else None,
                "k_collapse": d.get("k_collapse"),
                "n_layers": d.get("n_layers"),
            }
            for mod, d in r.by_module_type.items()
        },
        "v_module_excess": r.v_module_excess,
    }


def write_json(
    results: list[InventoryResult],
    combined: list[dict],
    output_path: Path,
    seed: int,
) -> None:
    """Write complete results to JSON."""
    all_fits = [r.linear_fit for r in results if r.linear_fit is not None]
    all_betas = [f.beta_1 for f in all_fits]
    all_p = [f.p_one_sided for f in all_fits]

    h_null = "not_rejected"
    if all_betas and any(b > 0 and p < 0.05 for b, p in zip(all_betas, all_p)):
        h_null = "rejected"

    h_linear = "supported" if h_null == "rejected" else "not_supported"

    h_sel_results = [r.h_selection for r in results if r.h_selection is not None]
    if h_sel_results:
        directions = [s.direction_positive_at_k3_plus for s in h_sel_results if s.direction_positive_at_k3_plus is not None]
        if directions and all(directions):
            h_selection = "supported"
        elif directions:
            h_selection = "not_supported"
        else:
            h_selection = "insufficient_data"
    else:
        h_selection = "insufficient_data"

    if all_betas:
        mean_beta = float(np.mean(all_betas))
        p1 = "confirmed" if 0 < mean_beta < 1.0 else ("disconfirmed" if mean_beta >= 1.0 else "not_applicable")
    else:
        p1 = "not_applicable"

    mixed_betas = [r.linear_fit.beta_1 for r in results if r.pool_type == "mixed_task" and r.linear_fit]
    same_betas = [r.linear_fit.beta_1 for r in results if r.pool_type == "same_task" and r.linear_fit]
    if mixed_betas and same_betas:
        p2 = "confirmed" if np.mean(mixed_betas) > np.mean(same_betas) else "disconfirmed"
    else:
        p2 = "insufficient_data"

    k_collapses = [r.k_collapse for r in results if r.k_collapse is not None]
    if k_collapses:
        p3 = "confirmed" if all(k > 5 for k in k_collapses) else "disconfirmed"
    else:
        p3 = "not_observed"

    output = {
        "study_id": "N129",
        "date": str(np.datetime64("today")),
        "gradience_version": "v1.0.1",
        "rng_seed": seed,
        "hypothesis_result": {
            "H_null": h_null,
            "H_linear": h_linear,
            "H_selection": h_selection,
            "P1_beta_less_than_1": p1,
            "P2_mixed_faster_than_same_task": p2,
            "P3_k_collapse_gt_5": p3,
        },
        "inventories": [_inventory_result_to_dict(r) for r in results],
        "combined_pools": combined,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(
    results: list[InventoryResult],
    combined: list[dict],
    seed: int,
) -> None:
    """Print structured console report per protocol S4.7."""
    print(f"\n{'='*74}")
    print("N129 Portfolio Rank Collapse Probe -- Results Summary")
    print(f"{'='*74}")

    excluded = 5 - len(results)  # 5 total inventories
    h_sel_testable = sum(1 for r in results if r.h_selection_testable)

    all_fits = [r.linear_fit for r in results if r.linear_fit is not None]
    all_betas = [f.beta_1 for f in all_fits]
    all_p = [f.p_one_sided for f in all_fits]

    # Section A: Summary
    print(f"\nInventories analyzed:           {len(results)}  (excluded: {excluded} with |R| < 2)")
    print(f"H_selection testable:           {h_sel_testable} inventories")

    print(f"\nPRIMARY RESULT (retained pools, aggregate):")
    if all_betas:
        mean_beta = float(np.mean(all_betas))
        se_beta = float(np.std(all_betas, ddof=1) / np.sqrt(len(all_betas))) if len(all_betas) > 1 else 0.0
        print(f"  Mean beta_1 across inventories:  {mean_beta:.4f} +/- {se_beta:.4f}")

        any_sig = any(b > 0 and p < 0.05 for b, p in zip(all_betas, all_p))
        print(f"  H_null rejected (beta_1 > 0):    {'Yes' if any_sig else 'No'} "
              f"(min one-sided p = {min(all_p):.4f})")
        print(f"  Comparison to Skorobogat et al. (beta_1 ~ 1.0 for random pools):")
        print(f"    P1 (beta_1 < 1.0):             {'confirmed' if 0 < mean_beta < 1.0 else 'disconfirmed'}")
    else:
        print(f"  No valid linear fits")

    k_collapses = [r.k_collapse for r in results if r.k_collapse is not None]
    k_extraps = [r.k_collapse_extrap for r in results if r.k_collapse_extrap is not None]
    if k_collapses:
        print(f"  k_collapse (rho > 2.0):         {min(k_collapses)} (observed)")
        print(f"  P3 (k_collapse > 5):             {'confirmed' if all(k > 5 for k in k_collapses) else 'disconfirmed'}")
    elif k_extraps:
        print(f"  k_collapse (rho > 2.0):         {min(k_extraps):.1f} (extrapolated)")
        print(f"  P3 (k_collapse > 5):             {'confirmed' if all(k > 5 for k in k_extraps) else 'disconfirmed'}")
    else:
        print(f"  k_collapse (rho > 2.0):         not observed in corpus")
        print(f"  P3 (k_collapse > 5):             not observed")

    # H_selection
    print(f"\nH_SELECTION:")
    h_sel_results = [r.h_selection for r in results if r.h_selection is not None]
    if h_sel_results:
        directions = [s.direction_positive_at_k3_plus for s in h_sel_results if s.direction_positive_at_k3_plus is not None]
        if directions:
            print(f"  Retained pools show lower rho(k) than full pools at k >= 3:")
            print(f"    {'Yes' if all(directions) else 'No'}")
        else:
            print(f"  Insufficient data (no k >= 3 comparisons)")
    else:
        print(f"  Insufficient data")

    # P2
    mixed_betas = [r.linear_fit.beta_1 for r in results if r.pool_type == "mixed_task" and r.linear_fit]
    same_betas = [r.linear_fit.beta_1 for r in results if r.pool_type == "same_task" and r.linear_fit]
    if mixed_betas and same_betas:
        print(f"  P2 (mixed-task > same-task beta_1):")
        print(f"    mixed mean={np.mean(mixed_betas):.4f}, same mean={np.mean(same_betas):.4f}")
        print(f"    {'confirmed' if np.mean(mixed_betas) > np.mean(same_betas) else 'disconfirmed'}")
    else:
        print(f"  P2 (mixed-task > same-task beta_1): insufficient data")

    # Framing verdict
    print(f"\nFRAMING VERDICT:")
    if k_collapses and min(k_collapses) <= 5:
        print(f"  S7.6 framing as 'future concern':  near-term revision needed")
    else:
        print(f"  S7.6 framing as 'future concern':  accurate")

    # Section B: Per-inventory table
    print(f"\n--- B. Per-Inventory Collapse Table ---")
    print(f"  {'inventory':<40} {'n_ret':>5} {'beta_1':>7} {'SE':>7} {'p(1s)':>8} {'k_coll':>7} {'k_ext':>8} {'dH@k3':>8}")
    print(f"  {'_'*96}")

    for r in results:
        b1 = f"{r.linear_fit.beta_1:.4f}" if r.linear_fit else "   n/a"
        se = f"{r.linear_fit.beta_1_se:.4f}" if r.linear_fit else "   n/a"
        pv = f"{r.linear_fit.p_one_sided:.4f}" if r.linear_fit else "   n/a"
        kc = str(r.k_collapse) if r.k_collapse else "   n/a"
        ke = f"{r.k_collapse_extrap:.1f}" if r.k_collapse_extrap else "   n/a"
        dh = "   n/a"
        if r.h_selection and 3 in r.h_selection.delta_rho_by_k:
            dh = f"{r.h_selection.delta_rho_by_k[3]:+.4f}"

        short_id = r.inventory_id if len(r.inventory_id) <= 40 else r.inventory_id[:37] + "..."
        print(f"  {short_id:<40} {r.n_retained:>5} {b1:>7} {se:>7} {pv:>8} {kc:>7} {ke:>8} {dh:>8}")

    # Section C: Module-type breakdown
    print(f"\n--- C. Per-Module-Type Breakdown (aggregate) ---")
    all_mod_betas: dict[str, list[float]] = {}
    for r in results:
        for mod, d in r.by_module_type.items():
            if d["beta_1"] is not None:
                all_mod_betas.setdefault(mod, []).append(d["beta_1"])

    agg_beta = float(np.mean(all_betas)) if all_betas else 0.0
    print(f"  {'module':<8} {'mean beta_1':>11} {'SE':>8} {'k_collapse':>11} {'V-excess?':>10}")
    for mod in sorted(all_mod_betas.keys()):
        betas = all_mod_betas[mod]
        mb = float(np.mean(betas))
        se = float(np.std(betas, ddof=1) / np.sqrt(len(betas))) if len(betas) > 1 else 0.0
        v_ex = "  <-- yes" if mod == "v" and mb > agg_beta else ""
        print(f"  {mod:<8} {mb:>11.4f} {se:>8.4f} {'n/a':>11} {v_ex}")

    # Section D: Combined pools
    if combined:
        print(f"\n--- D. Combined Backbone Pools ---")
        for cp in combined:
            fit = cp.get("linear_fit")
            if fit and fit.get("beta_1") is not None:
                print(f"  {cp['backbone']}: n={cp['n_adapters']}, "
                      f"beta_1={fit['beta_1']:.4f} "
                      f"[{fit['beta_1_ci_lower']:.4f}, {fit['beta_1_ci_upper']:.4f}], "
                      f"k_coll_ext={cp.get('k_collapse_extrap', 'n/a')}")
            else:
                print(f"  {cp['backbone']}: n={cp['n_adapters']}, fit not available")
        print(f"  (Note: combined pools mix adapters from different inventories -- extended-range only)")

    print(f"\n{'='*74}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="N129: Portfolio Rank Collapse Probe")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "sidecar" / "results")
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "sidecar" / "results" / "N129_rank_collapse")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)

    print("N129: Portfolio Rank Collapse Probe")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Max k: {args.max_k}, Seed: {args.seed}")

    # Step 1: Discover inventories
    print(f"\n{'_'*40}")
    print("Step 1: Discovering inventory pools")
    pools = discover_inventories(args.results_dir)
    print(f"\nUsable inventories: {len(pools)}")

    if not pools:
        print("ERROR: No usable inventories found (need |R| >= 2)")
        return 1

    # Step 2: Analyze each inventory
    print(f"\n{'_'*40}")
    print("Step 2: Analyzing per-inventory collapse")
    results = []
    for pool in pools:
        result = analyze_inventory(pool, args.max_k, rng)
        if result is not None:
            results.append(result)

    if not results:
        print("ERROR: No inventories produced results")
        return 1

    # Step 3: Combined backbone pools
    print(f"\n{'_'*40}")
    print("Step 3: Combined backbone pools")
    combined_map = build_combined_pools(pools)
    combined_results = []
    for backbone, adapters in sorted(combined_map.items()):
        if len(adapters) >= 3:
            cr = analyze_combined_pool(backbone, adapters, args.max_k, rng)
            if cr is not None:
                combined_results.append(cr)
        else:
            print(f"  {backbone}: only {len(adapters)} adapters, skipping combined analysis")

    # Step 4: Write JSON
    print(f"\n{'_'*40}")
    print("Step 4: Writing results")
    output_path = args.output_dir / "results.json"
    write_json(results, combined_results, output_path, args.seed)
    print(f"  JSON: {output_path}")

    # Step 5: Console report
    print_report(results, combined_results, args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
