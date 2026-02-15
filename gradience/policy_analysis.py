"""Shared policy disagreement analysis logic.

Extracted from cli.py to eliminate duplication between
_analyze_policy_disagreements (JSON output) and
_print_policy_disagreement_summary (terminal output).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np


def compute_layer_importance_scores(
    layers: List[Any],
    name_mapping: Dict[str, str],
    importance_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Compute importance scores and disagreement metrics for all layers.

    Returns:
        (layer_analysis, importance_scores) where layer_analysis is a list of
        dicts with per-layer metrics and importance_scores is the raw list for
        quantile calculations.  Returns ([], []) when no layers have sufficient
        policy data.
    """
    if importance_config is None:
        importance_config = {}

    if not layers:
        return [], []

    importance_scores: List[float] = []
    layer_analysis: List[Dict[str, Any]] = []

    for layer in layers:
        if not hasattr(layer, 'rank_suggestions') or not layer.rank_suggestions:
            continue

        k_values: List[int] = []
        layer_policies: List[str] = []
        for policy_internal, suggestion in layer.rank_suggestions.items():
            if isinstance(suggestion, dict) and 'k' in suggestion:
                k = suggestion['k']
                if isinstance(k, (int, float)) and k >= 0:
                    k_values.append(int(k))
                    layer_policies.append(name_mapping.get(policy_internal, policy_internal))

        if len(k_values) >= 2:
            policy_spread = max(k_values) - min(k_values)
            max_k = max(k_values)
            min_k = min(k_values)

            frobenius_norm = (getattr(layer, 'frob_sq', 0.0) ** 0.5) if hasattr(layer, 'frob_sq') else 0.0
            param_count = getattr(layer, 'params', 0)
            utilization = getattr(layer, 'utilization', 0.0) if hasattr(layer, 'utilization') else 0.0
            energy_raw = getattr(layer, 'frob_sq', 0.0) if hasattr(layer, 'frob_sq') else 0.0

            importance_raw = (
                frobenius_norm * 0.6 +
                (param_count / 10000) * 0.3 +
                utilization * 100 * 0.1
            )

            importance_scores.append(importance_raw)
            layer_analysis.append({
                'layer': layer,
                'layer_name': getattr(layer, 'name', getattr(layer, 'layer_name', 'unknown')),
                'k_values': k_values,
                'policy_spread': policy_spread,
                'spread': policy_spread,  # alias for terminal output compat
                'max_k': max_k,
                'min_k': min_k,
                'policies': layer_policies,
                'energy_raw': energy_raw,
                'frobenius_norm': frobenius_norm,
                'param_count': param_count,
                'utilization': utilization,
                'importance_raw': importance_raw,
            })

    return layer_analysis, importance_scores


def compute_energy_distribution(
    layer_analysis: List[Dict[str, Any]],
    min_uniform_mult: float,
) -> Tuple[float, float, float, bool]:
    """Calculate energy shares, uniform multipliers, and flat-distribution gate.

    Mutates each dict in *layer_analysis* to add ``energy_share``,
    ``uniform_mult``, and ``importance`` keys.

    Returns:
        (total_energy, uniform_share, max_uniform_mult, distribution_is_flat)
    """
    n_layers = len(layer_analysis)
    total_energy = sum(a['energy_raw'] for a in layer_analysis)
    uniform_share = 1.0 / n_layers if n_layers > 0 else 0.0

    for analysis in layer_analysis:
        energy_share = analysis['energy_raw'] / total_energy if total_energy > 0 else uniform_share
        uniform_mult = energy_share / uniform_share if uniform_share > 0 else 1.0
        analysis['energy_share'] = energy_share
        analysis['uniform_mult'] = uniform_mult
        analysis['importance'] = analysis['importance_raw']

    max_uniform_mult = max(a['uniform_mult'] for a in layer_analysis) if layer_analysis else 0.0
    distribution_is_flat = max_uniform_mult < min_uniform_mult

    return total_energy, uniform_share, max_uniform_mult, distribution_is_flat


def filter_disagreement_layers(
    layer_analysis: List[Dict[str, Any]],
    importance_scores: List[float],
    quantile_threshold: float,
    min_uniform_mult: float,
    distribution_is_flat: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply spread + energy filtering to find high-impact disagreement layers.

    Returns:
        (flagged_layers, all_disagreement_layers) — both sorted by
        priority_score descending.
    """
    quantile_pct = quantile_threshold * 100
    flagged: List[Dict[str, Any]] = []
    all_disagreement: List[Dict[str, Any]] = []

    for analysis in layer_analysis:
        policy_spread = analysis['policy_spread']
        max_k = analysis['max_k']
        energy_share = analysis['energy_share']
        uniform_mult = analysis['uniform_mult']

        spread_threshold = max(3, int(0.5 * max_k)) if max_k > 0 else 3
        high_spread = policy_spread >= spread_threshold

        spread_norm = max(0.0, policy_spread / spread_threshold) if spread_threshold > 0 else 0.0
        priority_score = spread_norm * uniform_mult
        analysis['priority_score'] = priority_score
        analysis['spread_threshold'] = spread_threshold
        analysis['meets_spread_threshold'] = high_spread

        if high_spread:
            all_disagreement.append(analysis)

            if not distribution_is_flat:
                energy_shares = [a['energy_share'] for a in layer_analysis]
                energy_quantile = np.percentile(energy_shares, quantile_pct)
                analysis['energy_quantile_threshold'] = energy_quantile
                meets_quantile = energy_share >= energy_quantile
                passed_gate = meets_quantile and uniform_mult >= min_uniform_mult
                analysis['meets_quantile_threshold'] = meets_quantile
                analysis['passed_gate'] = passed_gate
                if passed_gate:
                    flagged.append(analysis)

    flagged.sort(key=lambda x: x['priority_score'], reverse=True)
    all_disagreement.sort(key=lambda x: x['priority_score'], reverse=True)

    return flagged, all_disagreement
