#!/usr/bin/env python3
"""
HTSR / spectral-edge-gap phase-probe add-on runner (CPU-only).

Re-analyzes existing checkpoint artifacts (no new training) and compares:
  - HTSR-style tail exponent alpha
  - Edge-gap probe sigma1/sigma2
against existing spectral observables (stable/effective rank, energy concentration).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from gradience.research.spectral_extended import compute_spectral_edge_gap, fit_htsr_tail_exponent, fit_spectral_decay
from gradience.vnext.audit.lora_audit import low_rank_singular_values


DEFAULT_RUN_ROOTS = [
    "bench_runs/smoke_test/probe_r16",
    "test_regular_undertrained/probe_r16",
    "bench_runs/multiseed_seed42/probe_r32",
    "bench_runs/cert_v0.1_seed42/probe_r32",
    "bench_runs/cert_v0.1_seed42/uniform_median_r2",
    "bench_runs/safety_uniform_r16_extended_seed42/probe_r32",
]


@dataclass
class LayerMetrics:
    stable_rank: float
    effective_rank: float
    top1_energy: float
    spectral_decay_alpha: float | None
    htsr_alpha: float | None
    htsr_alpha_valid: bool
    htsr_fit_r2: float | None
    edge_gap_12: float | None
    edge_gap_valid: bool


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    pos = (len(xs) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    den = den_x * den_y
    if den < 1e-12:
        return None
    return float(num / den)


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _parse_step(checkpoint_name: str) -> int:
    try:
        return int(checkpoint_name.split("-")[-1])
    except ValueError:
        return -1


def _discover_run_roots(repo_root: Path, max_runs: int) -> list[Path]:
    selected: list[Path] = []

    for rel in DEFAULT_RUN_ROOTS:
        p = repo_root / rel
        if p.exists():
            checkpoints = [d for d in p.glob("checkpoint-*") if d.is_dir()]
            if len(checkpoints) >= 3:
                selected.append(p)

    if len(selected) >= max_runs:
        return selected[:max_runs]

    candidates: list[Path] = []
    for base in [repo_root / "bench_runs", repo_root / "test_regular_undertrained"]:
        if not base.exists():
            continue
        for d in base.rglob("*"):
            if not d.is_dir():
                continue
            checkpoints = [cp for cp in d.glob("checkpoint-*") if cp.is_dir()]
            if len(checkpoints) < 3:
                continue
            if not all((cp / "adapter_model.safetensors").exists() for cp in checkpoints):
                continue
            candidates.append(d)

    for cand in sorted(candidates):
        if cand in selected:
            continue
        selected.append(cand)
        if len(selected) >= max_runs:
            break
    return selected


def _extract_lora_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    a_map: dict[str, torch.Tensor] = {}
    b_map: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            prefix = key[: -len(".lora_A.weight")]
            a_map[prefix] = tensor
        elif key.endswith(".lora_B.weight"):
            prefix = key[: -len(".lora_B.weight")]
            b_map[prefix] = tensor

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for prefix, a in a_map.items():
        b = b_map.get(prefix)
        if b is not None:
            pairs[prefix] = (a, b)
    return pairs


def _effective_rank_from_singular_values(values: list[float], eps: float = 1e-12) -> float:
    sv = [v for v in values if v > eps]
    if not sv:
        return 0.0
    total = sum(sv)
    probs = [v / total for v in sv]
    entropy = -sum(p * math.log(p + eps) for p in probs)
    return float(math.exp(entropy))


def _stable_rank_from_singular_values(values: list[float], eps: float = 1e-12) -> float:
    sv = [v for v in values if v > eps]
    if not sv:
        return 0.0
    sq = sum(v * v for v in sv)
    return float(sq / (sv[0] * sv[0] + eps))


def _top1_energy_from_singular_values(values: list[float], eps: float = 1e-12) -> float:
    sv = [v for v in values if v > eps]
    if not sv:
        return 0.0
    sq = sum(v * v for v in sv)
    return float((sv[0] * sv[0]) / (sq + eps))


def _compute_layer_metrics(singular_values: list[float]) -> LayerMetrics:
    decay_alpha, _decay_r2 = fit_spectral_decay(singular_values)
    tail = fit_htsr_tail_exponent(singular_values)
    edge = compute_spectral_edge_gap(singular_values)

    return LayerMetrics(
        stable_rank=_stable_rank_from_singular_values(singular_values),
        effective_rank=_effective_rank_from_singular_values(singular_values),
        top1_energy=_top1_energy_from_singular_values(singular_values),
        spectral_decay_alpha=decay_alpha,
        htsr_alpha=tail.alpha,
        htsr_alpha_valid=tail.valid,
        htsr_fit_r2=tail.fit_r2,
        edge_gap_12=edge.edge_gap_12,
        edge_gap_valid=edge.valid,
    )


def _read_checkpoint_metrics(checkpoint_dir: Path, step: int) -> dict[str, float | None]:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return {"train_loss": None, "eval_loss": None, "eval_accuracy": None}

    try:
        trainer_state = json.loads(trainer_state_path.read_text())
    except json.JSONDecodeError:
        return {"train_loss": None, "eval_loss": None, "eval_accuracy": None}

    log_history = trainer_state.get("log_history", [])
    if not isinstance(log_history, list):
        return {"train_loss": None, "eval_loss": None, "eval_accuracy": None}

    # Prefer exact-step eval entries, then nearest previous eval entries.
    eval_candidates = [e for e in log_history if isinstance(e, dict) and "step" in e and e.get("step", -1) <= step]
    eval_candidates = [e for e in eval_candidates if ("eval_loss" in e) or ("eval_accuracy" in e)]
    eval_entry = eval_candidates[-1] if eval_candidates else {}

    train_candidates = [e for e in log_history if isinstance(e, dict) and "loss" in e and e.get("step", -1) <= step]
    train_entry = train_candidates[-1] if train_candidates else {}

    return {
        "train_loss": float(train_entry["loss"]) if "loss" in train_entry else None,
        "eval_loss": float(eval_entry["eval_loss"]) if "eval_loss" in eval_entry else None,
        "eval_accuracy": float(eval_entry["eval_accuracy"]) if "eval_accuracy" in eval_entry else None,
    }


def _aggregate_checkpoint(layer_metrics: list[LayerMetrics]) -> dict[str, float | int | None]:
    stable = [m.stable_rank for m in layer_metrics]
    erank = [m.effective_rank for m in layer_metrics]
    top1 = [m.top1_energy for m in layer_metrics]
    decay = [m.spectral_decay_alpha for m in layer_metrics if m.spectral_decay_alpha is not None]
    htsr_alpha = [m.htsr_alpha for m in layer_metrics if m.htsr_alpha is not None]
    htsr_alpha_valid = [m.htsr_alpha for m in layer_metrics if m.htsr_alpha_valid and m.htsr_alpha is not None]
    htsr_r2 = [m.htsr_fit_r2 for m in layer_metrics if m.htsr_fit_r2 is not None]
    edge12 = [m.edge_gap_12 for m in layer_metrics if m.edge_gap_12 is not None]
    edge_valid_count = sum(1 for m in layer_metrics if m.edge_gap_valid)
    htsr_valid_count = sum(1 for m in layer_metrics if m.htsr_alpha_valid)

    n_layers = len(layer_metrics)
    return {
        "n_layers": n_layers,
        "stable_rank_mean": _mean(stable),
        "effective_rank_mean": _mean(erank),
        "top1_energy_mean": _mean(top1),
        "spectral_decay_alpha_mean": _mean(decay),
        "htsr_alpha_mean": _mean(htsr_alpha),
        "htsr_alpha_median": _median(htsr_alpha),
        "htsr_alpha_valid_mean": _mean(htsr_alpha_valid),
        "htsr_fit_r2_mean": _mean(htsr_r2),
        "htsr_valid_layer_count": htsr_valid_count,
        "htsr_valid_layer_fraction": (htsr_valid_count / n_layers) if n_layers else 0.0,
        "edge_gap_12_mean": _mean(edge12),
        "edge_gap_12_p90": _quantile([v for v in edge12 if v is not None], 0.9),
        "edge_valid_layer_count": edge_valid_count,
        "edge_valid_layer_fraction": (edge_valid_count / n_layers) if n_layers else 0.0,
    }


def _compute_run_timeseries(run_root: Path, repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints = sorted([cp for cp in run_root.glob("checkpoint-*") if cp.is_dir()], key=lambda p: _parse_step(p.name))
    rows: list[dict[str, Any]] = []

    for cp in checkpoints:
        step = _parse_step(cp.name)
        model_path = cp / "adapter_model.safetensors"
        if step < 0 or not model_path.exists():
            continue

        state = load_file(str(model_path))
        pairs = _extract_lora_pairs(state)
        per_layer: list[LayerMetrics] = []

        for layer_name in sorted(pairs):
            A, B = pairs[layer_name]
            try:
                s = low_rank_singular_values(A, B, compute_dtype=torch.float64)
                values = [float(v) for v in s.cpu().tolist() if float(v) > 1e-12]
                if not values:
                    continue
                per_layer.append(_compute_layer_metrics(values))
            except RuntimeError:
                continue

        if not per_layer:
            continue

        agg = _aggregate_checkpoint(per_layer)
        ck_metrics = _read_checkpoint_metrics(cp, step)

        row: dict[str, Any] = {
            "run_id": str(run_root.relative_to(repo_root)),
            "checkpoint": cp.name,
            "step": step,
            "policy_label": run_root.name,
            "family_label": run_root.parent.name if run_root.parent != repo_root else run_root.name,
            **agg,
            **ck_metrics,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: int(r["step"]))
    run_meta = {
        "run_id": str(run_root.relative_to(repo_root)),
        "run_path": str(run_root),
        "policy_label": run_root.name,
        "family_label": run_root.parent.name if run_root.parent != repo_root else run_root.name,
        "n_checkpoints": len(rows),
        "steps": [int(r["step"]) for r in rows],
        "baseline_observables_available": {
            "stable_rank_mean": all(r.get("stable_rank_mean") is not None for r in rows),
            "effective_rank_mean": all(r.get("effective_rank_mean") is not None for r in rows),
            "top1_energy_mean": all(r.get("top1_energy_mean") is not None for r in rows),
            "spectral_decay_alpha_mean": any(r.get("spectral_decay_alpha_mean") is not None for r in rows),
            "dfa_metrics": False,
        },
    }
    return rows, run_meta


def _peak_change_step(rows: list[dict[str, Any]], metric: str) -> int | None:
    if len(rows) < 2:
        return None
    best: tuple[float, int] | None = None
    for prev, curr in zip(rows[:-1], rows[1:]):
        a = prev.get(metric)
        b = curr.get(metric)
        if a is None or b is None:
            continue
        mag = abs(float(b) - float(a))
        if best is None or mag > best[0]:
            best = (mag, int(curr["step"]))
    return best[1] if best else None


def _assign_regimes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable_vals = [float(r["stable_rank_mean"]) for r in rows if r.get("stable_rank_mean") is not None]
    if len(stable_vals) < 3:
        for r in rows:
            r["regime_label"] = "unknown"
        return rows

    q1 = _quantile(stable_vals, 1 / 3) or min(stable_vals)
    q2 = _quantile(stable_vals, 2 / 3) or max(stable_vals)
    for r in rows:
        v = r.get("stable_rank_mean")
        if v is None:
            r["regime_label"] = "unknown"
        elif v <= q1:
            r["regime_label"] = "low_rank_regime"
        elif v <= q2:
            r["regime_label"] = "mid_rank_regime"
        else:
            r["regime_label"] = "high_rank_regime"
    return rows


def _build_comparative_analysis(timeseries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _assign_regimes([dict(r) for r in timeseries])
    n_rows = len(rows)

    probe_validity = {
        "htsr_valid_row_fraction": _mean([r.get("htsr_valid_layer_fraction", 0.0) for r in rows]) or 0.0,
        "edge_valid_row_fraction": _mean([r.get("edge_valid_layer_fraction", 0.0) for r in rows]) or 0.0,
    }

    regime_groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        regime_groups.setdefault(str(r.get("regime_label", "unknown")), []).append(r)

    regime_summary: dict[str, Any] = {}
    for regime, grp in regime_groups.items():
        regime_summary[regime] = {
            "count": len(grp),
            "htsr_alpha_mean": _mean([float(r["htsr_alpha_mean"]) for r in grp if r.get("htsr_alpha_mean") is not None]),
            "edge_gap_12_mean": _mean([float(r["edge_gap_12_mean"]) for r in grp if r.get("edge_gap_12_mean") is not None]),
            "stable_rank_mean": _mean([float(r["stable_rank_mean"]) for r in grp if r.get("stable_rank_mean") is not None]),
            "top1_energy_mean": _mean([float(r["top1_energy_mean"]) for r in grp if r.get("top1_energy_mean") is not None]),
        }

    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_run.setdefault(str(r["run_id"]), []).append(r)
    for run_id in by_run:
        by_run[run_id] = sorted(by_run[run_id], key=lambda x: int(x["step"]))

    transition_leads: dict[str, list[int]] = {"htsr_alpha_mean": [], "edge_gap_12_mean": []}
    per_run_transition: list[dict[str, Any]] = []
    for run_id, run_rows in by_run.items():
        base_step = _peak_change_step(run_rows, "stable_rank_mean")
        if base_step is None:
            continue
        row = {"run_id": run_id, "baseline_peak_step_stable_rank": base_step}
        for metric in ["htsr_alpha_mean", "edge_gap_12_mean", "effective_rank_mean", "top1_energy_mean"]:
            m_step = _peak_change_step(run_rows, metric)
            row[f"{metric}_peak_step"] = m_step
            if m_step is not None and metric in transition_leads:
                transition_leads[metric].append(base_step - m_step)
        per_run_transition.append(row)

    correlation_targets = {}
    for outcome in ["eval_accuracy", "eval_loss", "train_loss"]:
        pairs = [(r, r.get(outcome)) for r in rows if r.get(outcome) is not None]
        if len(pairs) < 4:
            continue
        outcome_vals = [float(v) for _, v in pairs]
        metric_corrs: dict[str, float | None] = {}
        for metric in [
            "stable_rank_mean",
            "effective_rank_mean",
            "top1_energy_mean",
            "spectral_decay_alpha_mean",
            "htsr_alpha_mean",
            "edge_gap_12_mean",
        ]:
            xs = [float(rr[metric]) for rr, _ in pairs if rr.get(metric) is not None]
            ys = [float(vv) for rr, vv in pairs if rr.get(metric) is not None]
            metric_corrs[metric] = _spearman(xs, ys) if len(xs) >= 4 else None
        correlation_targets[outcome] = metric_corrs

    # Redundancy snapshot across structural observables (all rows).
    observable_keys = [
        "stable_rank_mean",
        "effective_rank_mean",
        "top1_energy_mean",
        "spectral_decay_alpha_mean",
        "htsr_alpha_mean",
        "edge_gap_12_mean",
    ]
    redundancy_matrix: dict[str, dict[str, float | None]] = {}
    for a in observable_keys:
        redundancy_matrix[a] = {}
        for b in observable_keys:
            pairs = [(r.get(a), r.get(b)) for r in rows if r.get(a) is not None and r.get(b) is not None]
            if len(pairs) < 4:
                redundancy_matrix[a][b] = None
                continue
            xa = [float(x) for x, _ in pairs]
            xb = [float(y) for _, y in pairs]
            redundancy_matrix[a][b] = _spearman(xa, xb)

    baseline_metrics = ["stable_rank_mean", "effective_rank_mean", "top1_energy_mean", "spectral_decay_alpha_mean"]
    probe_metrics = ["htsr_alpha_mean", "edge_gap_12_mean"]
    added_value: dict[str, Any] = {}
    for outcome, corr_map in correlation_targets.items():
        base_corrs = [abs(corr_map[m]) for m in baseline_metrics if corr_map.get(m) is not None]
        probe_corrs = [abs(corr_map[m]) for m in probe_metrics if corr_map.get(m) is not None]
        if not base_corrs and not probe_corrs:
            continue
        added_value[outcome] = {
            "baseline_best_abs_spearman": max(base_corrs) if base_corrs else None,
            "probe_best_abs_spearman": max(probe_corrs) if probe_corrs else None,
            "best_probe_metric": max(probe_metrics, key=lambda m: abs(corr_map.get(m) or -1.0)),
        }

    def decide_probe(metric_name: str, valid_fraction: float) -> str:
        probe_corrs: list[float] = []
        base_corrs: list[float] = []
        for corr_map in correlation_targets.values():
            m = corr_map.get(metric_name)
            if m is not None:
                probe_corrs.append(abs(m))
            bvals = [abs(corr_map.get(k)) for k in baseline_metrics if corr_map.get(k) is not None]
            if bvals:
                base_corrs.append(max(bvals))

        probe_best = max(probe_corrs) if probe_corrs else 0.0
        baseline_best = max(base_corrs) if base_corrs else 0.0
        lead_median = _median([float(v) for v in transition_leads.get(metric_name, [])]) or 0.0

        if valid_fraction < 0.30:
            return "discard"
        # Keep requires stronger evidence: better than baseline by a clear margin,
        # positive median lead, and enough timepoints.
        if probe_best >= baseline_best + 0.10 and lead_median > 0 and n_rows >= 40:
            return "keep"
        if probe_best <= baseline_best and valid_fraction < 0.5:
            return "discard"
        return "bounded_keep"

    decision = {
        "htsr_alpha": decide_probe("htsr_alpha_mean", probe_validity["htsr_valid_row_fraction"]),
        "edge_gap_12": decide_probe("edge_gap_12_mean", probe_validity["edge_valid_row_fraction"]),
    }

    return {
        "n_timepoints": n_rows,
        "probe_validity": probe_validity,
        "regime_discrimination": regime_summary,
        "transition_sensitivity": {
            "per_run_peak_steps": per_run_transition,
            "lead_vs_stable_rank": {
                metric: {
                    "count": len(vals),
                    "median_lead_steps": _median([float(v) for v in vals]),
                    "mean_lead_steps": _mean([float(v) for v in vals]),
                }
                for metric, vals in transition_leads.items()
            },
        },
        "correlation_vs_outcomes": correlation_targets,
        "redundancy_matrix_spearman": redundancy_matrix,
        "added_value": added_value,
        "probe_decision": decision,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_timeseries_md(path: Path, rows: list[dict[str, Any]], run_meta: list[dict[str, Any]]) -> None:
    summary_rows = []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_run.setdefault(str(r["run_id"]), []).append(r)
    for run_id, run_rows in sorted(by_run.items()):
        summary_rows.append(
            {
                "run_id": run_id,
                "n_points": len(run_rows),
                "step_min": min(int(rr["step"]) for rr in run_rows),
                "step_max": max(int(rr["step"]) for rr in run_rows),
                "htsr_alpha_mean": _mean([float(rr["htsr_alpha_mean"]) for rr in run_rows if rr.get("htsr_alpha_mean") is not None]),
                "edge_gap_12_mean": _mean([float(rr["edge_gap_12_mean"]) for rr in run_rows if rr.get("edge_gap_12_mean") is not None]),
                "stable_rank_mean": _mean([float(rr["stable_rank_mean"]) for rr in run_rows if rr.get("stable_rank_mean") is not None]),
            }
        )

    md = []
    md.append("# Phase Probe Timeseries")
    md.append("")
    md.append(f"- Runs analyzed: {len(by_run)}")
    md.append(f"- Timepoints analyzed: {len(rows)}")
    md.append("- New probes: `htsr_alpha` and `edge_gap_12`")
    md.append("- Baseline observables: `stable_rank_mean`, `effective_rank_mean`, `top1_energy_mean`, `spectral_decay_alpha_mean`")
    md.append("")
    md.append("## Run Summary")
    md.append("")
    md.append(
        _markdown_table(
            summary_rows,
            ["run_id", "n_points", "step_min", "step_max", "stable_rank_mean", "htsr_alpha_mean", "edge_gap_12_mean"],
        )
    )
    md.append("")
    md.append("## Baseline Run Metadata")
    md.append("")
    md.append(
        _markdown_table(
            run_meta,
            ["run_id", "policy_label", "family_label", "n_checkpoints", "steps"],
        )
    )
    path.write_text("\n".join(md) + "\n")


def _write_comparative_md(path: Path, analysis: dict[str, Any]) -> None:
    decision = analysis.get("probe_decision", {})
    lead = analysis.get("transition_sensitivity", {}).get("lead_vs_stable_rank", {})
    md = []
    md.append("# Comparative Analysis")
    md.append("")
    md.append(f"- Timepoints: {analysis.get('n_timepoints')}")
    md.append(
        f"- Probe validity: HTSR={analysis.get('probe_validity', {}).get('htsr_valid_row_fraction', 0.0):.3f}, "
        f"Edge-gap={analysis.get('probe_validity', {}).get('edge_valid_row_fraction', 0.0):.3f}"
    )
    md.append(
        f"- Transition lead median (vs stable-rank peaks): HTSR={lead.get('htsr_alpha_mean', {}).get('median_lead_steps')}, "
        f"Edge-gap={lead.get('edge_gap_12_mean', {}).get('median_lead_steps')}"
    )
    md.append("")
    md.append("## Probe Decisions")
    md.append("")
    md.append(f"- HTSR alpha: `{decision.get('htsr_alpha', 'bounded_keep')}`")
    md.append(f"- Edge-gap (sigma1/sigma2): `{decision.get('edge_gap_12', 'bounded_keep')}`")
    md.append("")
    md.append("## Regime Discrimination Snapshot")
    md.append("")
    regime_rows = []
    for regime, vals in sorted((analysis.get("regime_discrimination") or {}).items()):
        row = {"regime": regime, **vals}
        regime_rows.append(row)
    if regime_rows:
        md.append(
            _markdown_table(
                regime_rows,
                ["regime", "count", "stable_rank_mean", "htsr_alpha_mean", "edge_gap_12_mean", "top1_energy_mean"],
            )
        )
    path.write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HTSR/edge-gap phase-probe add-on reanalysis on saved checkpoints.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=6,
        help="Maximum run roots to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sidecar/results/phase_probe_addon"),
        help="Output directory for JSON/MD artifacts.",
    )
    parser.add_argument(
        "--write-baseline-notes",
        action="store_true",
        help="If set, also writes a small baseline note scaffold under sidecar/notes.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_roots = _discover_run_roots(repo_root, max_runs=args.max_runs)
    if not run_roots:
        raise SystemExit("No suitable run roots with checkpoint-* adapters were found.")

    all_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    for run_root in run_roots:
        rows, meta = _compute_run_timeseries(run_root, repo_root)
        if rows:
            all_rows.extend(rows)
            run_meta.append(meta)

    all_rows = sorted(all_rows, key=lambda r: (str(r["run_id"]), int(r["step"])))
    if not all_rows:
        raise SystemExit("No checkpoint rows could be analyzed.")

    baseline_snapshot = {
        "study": "htsr_and_spectral_edge_gap_phase_probe_addon",
        "scope": "cpu_only_reanalysis_existing_checkpoint_artifacts",
        "run_count": len(run_meta),
        "timepoint_count": len(all_rows),
        "runs": run_meta,
        "current_toolkit_baseline": [
            "stable_rank_mean",
            "effective_rank_mean",
            "top1_energy_mean",
            "spectral_decay_alpha_mean",
        ],
        "dfa_available_in_this_slice": False,
        "candidate_transition_reference": "peak absolute change in stable_rank_mean by run",
    }

    analysis = _build_comparative_analysis(all_rows)
    decision_summary = {
        "probe_decisions": analysis.get("probe_decision", {}),
        "decision_categories": ["keep", "bounded_keep", "discard"],
        "notes": "Decision is bounded to this CPU-only reanalysis slice and should not be over-generalized.",
    }

    out = args.output_dir
    _write_json(out / "baseline_snapshot.json", baseline_snapshot)
    _write_json(out / "phase_probe_timeseries.json", all_rows)
    _write_json(out / "comparative_analysis.json", analysis)
    _write_json(out / "phase_probe_decision_summary.json", decision_summary)
    _write_timeseries_md(out / "phase_probe_timeseries.md", all_rows, run_meta)
    _write_comparative_md(out / "comparative_analysis.md", analysis)

    if args.write_baseline_notes:
        note_path = repo_root / "sidecar/notes/n125_phase_probe_baseline.md"
        note_lines = [
            "# n125 - Phase Probe Add-On Baseline Freeze",
            "",
            "- Scope: CPU-only reanalysis over existing checkpoint artifacts.",
            f"- Runs frozen: {len(run_meta)}",
            f"- Timepoints frozen: {len(all_rows)}",
            "- Baseline observables: stable/effective rank, top1 energy, spectral decay alpha.",
            "- New probes to compare: HTSR tail alpha and edge-gap (sigma1/sigma2).",
            "",
            "See `sidecar/results/phase_probe_addon/baseline_snapshot.json` for the canonical machine-readable baseline.",
        ]
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("\n".join(note_lines) + "\n")

    print(f"Wrote outputs to: {out}")
    print(f"Runs analyzed: {len(run_meta)}")
    print(f"Timepoints analyzed: {len(all_rows)}")
    print(f"Probe decision: {analysis.get('probe_decision')}")


if __name__ == "__main__":
    main()
