#!/usr/bin/env python3
"""Phase-2/3 synthetic comparison: OA-v1 heuristic vs OA-v2 rank-r interaction line.

This runner performs higher-rank controlled sweeps and writes:
- heuristic comparison artifact (v1 vs v2)
- synthetic validation artifact (coverage + score sanity)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gradience.vnext.merge.over_accumulation import (
    estimate_over_accumulation,
    estimate_over_accumulation_v2,
)
from gradience.vnext.merge.spectral_compat import compute_subspace_metrics
from gradience.vnext.merge.spectral_theory_test_utils import sigma1


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None

    def _rank(vals: list[float]) -> list[float]:
        ordered = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            rank_v = ((i + j - 1) / 2.0) + 1.0
            for k in range(i, j):
                out[ordered[k][1]] = rank_v
            i = j
        return out

    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(den_x * den_y)
    if den <= 0.0:
        return None
    return num / den


def _classification(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    n = max(len(rows), 1)
    tp = fp = fn = tn = 0
    for r in rows:
        truth = bool(r["true_over"])
        pred = bool(r[pred_key])
        if truth and pred:
            tp += 1
        elif (not truth) and pred:
            fp += 1
        elif truth and (not pred):
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / n
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    margins = [float(r["true_margin"]) for r in rows]
    v1_scores = [float(r["oa_v1_score"]) for r in rows]
    v2_scores = [float(r["oa_v2_score"]) for r in rows]

    return {
        "n": len(rows),
        "true_over_fraction": sum(1 for r in rows if r["true_over"]) / len(rows),
        "v1_watch_or_high_fraction": sum(1 for r in rows if r["oa_v1_watch_or_high"]) / len(rows),
        "v2_watch_or_high_fraction": sum(1 for r in rows if r["oa_v2_watch_or_high"]) / len(rows),
        "spearman_v1_score_vs_true_margin": _spearman(v1_scores, margins),
        "spearman_v2_score_vs_true_margin": _spearman(v2_scores, margins),
        "spearman_v2_vs_v1": _spearman(v2_scores, v1_scores),
        "classification": {
            "v1": _classification(rows, "oa_v1_watch_or_high"),
            "v2": _classification(rows, "oa_v2_watch_or_high"),
        },
    }


def _qr_orthonormal(n: int, cols: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, cols, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(x)
    return q[:, :cols]


def _cos_profile(kind: str, r: int) -> list[float]:
    if kind == "low":
        return [0.15 - 0.08 * (i / max(r - 1, 1)) for i in range(r)]
    if kind == "mixed":
        return [0.85 - 0.65 * (i / max(r - 1, 1)) for i in range(r)]
    if kind == "high":
        return [0.97 - 0.20 * (i / max(r - 1, 1)) for i in range(r)]
    raise ValueError(f"unknown cos profile: {kind}")


def _sigma_profile(kind: str, r: int) -> list[float]:
    if kind == "flat":
        return [1.0 for _ in range(r)]
    if kind == "exp":
        return [math.exp(-0.22 * i) for i in range(r)]
    if kind == "power":
        return [1.0 / ((i + 1) ** 0.8) for i in range(r)]
    raise ValueError(f"unknown sigma profile: {kind}")


def _make_basis_pair(n: int, cosines: list[float], g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    r = len(cosines)
    if n < 2 * r:
        raise ValueError(f"need n >= 2r for controlled basis construction, got n={n}, r={r}")
    q = _qr_orthonormal(n, 2 * r, g)
    a = q[:, :r]
    orth = q[:, r : 2 * r]
    cols = []
    for i in range(r):
        c = max(0.0, min(1.0, float(cosines[i])))
        s = math.sqrt(max(0.0, 1.0 - c * c))
        cols.append(c * a[:, i] + s * orth[:, i])
    b = torch.stack(cols, dim=1)
    return a, b


def _delta_from_factors(u: torch.Tensor, s: list[float], v: torch.Tensor) -> torch.Tensor:
    s_t = torch.tensor(s, dtype=torch.float64)
    return u @ torch.diag(s_t) @ v.T


def _factorize_delta_to_lora(delta: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(delta.to(torch.float64), full_matrices=False)
    u = u[:, :r]
    s = s[:r]
    vh = vh[:r, :]
    sqrt_s = torch.sqrt(torch.clamp(s, min=0.0))
    a = (sqrt_s.unsqueeze(1) * vh).to(torch.float32).contiguous()
    b = (u * sqrt_s.unsqueeze(0)).to(torch.float32).contiguous()
    return a, b


def run_rankr_sweep(*, seeds_per_setting: int, d_out: int, d_in: int) -> dict[str, Any]:
    ranks = [4, 8, 16, 32]
    coefficients = [(0.5, 0.5), (0.7, 0.3)]
    signs = [1.0, -1.0]
    angle_profiles = ["low", "mixed", "high"]
    sigma_profiles = [("flat", "flat"), ("exp", "exp"), ("power", "power"), ("exp", "power")]

    rows: list[dict[str, Any]] = []
    run_idx = 0
    for r in ranks:
        for left_kind in angle_profiles:
            left_cos = _cos_profile(left_kind, r)
            for right_kind in angle_profiles:
                right_cos = _cos_profile(right_kind, r)
                for s_kind_a, s_kind_b in sigma_profiles:
                    s_a = _sigma_profile(s_kind_a, r)
                    s_b = _sigma_profile(s_kind_b, r)
                    for sign in signs:
                        for alpha, beta in coefficients:
                            for seed in range(seeds_per_setting):
                                run_idx += 1
                                g = torch.Generator(device="cpu")
                                g.manual_seed(37_000 + run_idx * 17 + seed)

                                u_a, u_b = _make_basis_pair(d_out, left_cos, g)
                                v_a, v_b = _make_basis_pair(d_in, right_cos, g)
                                delta_a = _delta_from_factors(u_a, s_a, v_a)
                                delta_b = _delta_from_factors(u_b, s_b, v_b)
                                if sign < 0:
                                    delta_b = -delta_b

                                a_a, b_a = _factorize_delta_to_lora(delta_a, r)
                                a_b, b_b = _factorize_delta_to_lora(delta_b, r)
                                metrics = compute_subspace_metrics(
                                    a_a,
                                    b_a,
                                    float(r),
                                    r,
                                    a_b,
                                    b_b,
                                    float(r),
                                    r,
                                    compute_dtype=torch.float64,
                                )

                                s1_a = sigma1(delta_a)
                                s1_b = sigma1(delta_b)
                                s1_m = sigma1(alpha * delta_a + beta * delta_b)
                                baseline = max(abs(alpha) * s1_a, abs(beta) * s1_b)
                                margin = s1_m - baseline
                                truth = bool(margin > 1e-8)

                                v1 = estimate_over_accumulation(metrics, assumed_coefficients=(alpha, beta))
                                v2 = estimate_over_accumulation_v2(metrics, assumed_coefficients=(alpha, beta))

                                right_mean = 0.0
                                if metrics.right_principal_angle_cosines:
                                    right_mean = sum(metrics.right_principal_angle_cosines) / len(
                                        metrics.right_principal_angle_cosines
                                    )

                                rows.append(
                                    {
                                        "rank": r,
                                        "left_profile": left_kind,
                                        "right_profile": right_kind,
                                        "sigma_profile_a": s_kind_a,
                                        "sigma_profile_b": s_kind_b,
                                        "directional_sign": sign,
                                        "alpha": alpha,
                                        "beta": beta,
                                        "mean_left_overlap": float(metrics.mean_overlap),
                                        "mean_right_overlap": float(right_mean),
                                        "directional_agreement": float(metrics.directional_agreement),
                                        "true_margin": margin,
                                        "true_over": truth,
                                        "oa_v1_score": float(v1.score),
                                        "oa_v1_band": v1.band,
                                        "oa_v1_watch_or_high": v1.band in {"watch", "high"},
                                        "oa_v2_score": float(v2.score),
                                        "oa_v2_band": v2.band,
                                        "oa_v2_watch_or_high": v2.band in {"watch", "high"},
                                        "oa_v2_interaction_primary": float(v2.factors.interaction_primary),
                                        "oa_v2_interaction_top3_mean": float(v2.factors.interaction_top3_mean),
                                        "oa_v2_degraded_input": bool(v2.factors.degraded_input),
                                    }
                                )

    slices = {
        "all": rows,
        "positive_alignment": [r for r in rows if r["directional_sign"] > 0],
        "negative_alignment": [r for r in rows if r["directional_sign"] < 0],
        "high_overlap": [r for r in rows if r["mean_left_overlap"] >= 0.65 and r["mean_right_overlap"] >= 0.65],
        "high_overlap_positive": [
            r
            for r in rows
            if r["directional_sign"] > 0 and r["mean_left_overlap"] >= 0.65 and r["mean_right_overlap"] >= 0.65
        ],
    }
    summary = {name: _summary(subset) for name, subset in slices.items()}
    summary["config"] = {
        "ranks": ranks,
        "coefficients": coefficients,
        "signs": signs,
        "angle_profiles": angle_profiles,
        "sigma_profiles": sigma_profiles,
        "seeds_per_setting": seeds_per_setting,
        "d_out": d_out,
        "d_in": d_in,
    }
    summary["degraded_input_fraction"] = (
        sum(1 for r in rows if bool(r.get("oa_v2_degraded_input", False))) / len(rows) if rows else 0.0
    )
    summary["n_total"] = len(rows)
    summary["sample_row"] = rows[0] if rows else {}
    return summary


def _write_markdown(heur_json: dict[str, Any], syn_json: dict[str, Any], heur_md_path: Path, syn_md_path: Path) -> None:
    s = heur_json["summary"]["rank_r"]
    all_s = s["all"]
    high_s = s["high_overlap_positive"]
    heur_md = f"""# OA-v1 vs OA-v2 Heuristic Comparison (Higher-Rank Sweep)

## Scope

- Controlled rank-r synthetic sweep using full left/right angle spectra.
- Ranks: `{s['config']['ranks']}`
- Coefficients: `{s['config']['coefficients']}`
- Seeds per setting: `{s['config']['seeds_per_setting']}`
- Cases: `{s['n_total']}`

## Key Results

- Overall Spearman:
  - OA-v1 vs true margin: `{all_s['spearman_v1_score_vs_true_margin']:.4f}`
  - OA-v2 vs true margin: `{all_s['spearman_v2_score_vs_true_margin']:.4f}`
- High-overlap positive-alignment Spearman:
  - OA-v1: `{high_s['spearman_v1_score_vs_true_margin']:.4f}`
  - OA-v2: `{high_s['spearman_v2_score_vs_true_margin']:.4f}`

## Interpretation

OA-v2 uses rank-r cross-term geometry and should be read as experimental.
OA-v1 remains the production advisory path until strict-naive field cross-check passes.
"""
    heur_md_path.write_text(heur_md)

    syn = syn_json["metrics"]
    syn_md = f"""# Synthetic Validation Results (Higher-Rank OA-v2)

- n_total: `{syn['n_total']}`
- high_overlap_positive Spearman gain (v2 - v1): `{syn['high_overlap_positive_spearman_gain']:.4f}`
- overall Spearman gain (v2 - v1): `{syn['overall_spearman_gain']:.4f}`
- degraded-input fraction: `{syn['degraded_input_fraction']:.4f}`
"""
    syn_md_path.write_text(syn_md)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds-per-setting", type=int, default=12)
    p.add_argument("--d-out", type=int, default=128)
    p.add_argument("--d-in", type=int, default=128)
    p.add_argument(
        "--heuristic-json",
        type=Path,
        default=Path("field_trials/analytical_spectral_geometry/heuristic_comparison.json"),
    )
    p.add_argument(
        "--synthetic-json",
        type=Path,
        default=Path("field_trials/analytical_spectral_geometry/synthetic_validation_results.json"),
    )
    p.add_argument(
        "--heuristic-md",
        type=Path,
        default=Path("field_trials/analytical_spectral_geometry/heuristic_comparison.md"),
    )
    p.add_argument(
        "--synthetic-md",
        type=Path,
        default=Path("field_trials/analytical_spectral_geometry/synthetic_validation_results.md"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary_rankr = run_rankr_sweep(
        seeds_per_setting=max(1, int(args.seeds_per_setting)),
        d_out=max(64, int(args.d_out)),
        d_in=max(64, int(args.d_in)),
    )

    payload_heur = {
        "status": "phase2_rankr_v2_comparison_completed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {
            "rank_r": summary_rankr,
        },
    }
    args.heuristic_json.parent.mkdir(parents=True, exist_ok=True)
    args.heuristic_json.write_text(json.dumps(payload_heur, indent=2))

    all_s = summary_rankr["all"]
    high_s = summary_rankr["high_overlap_positive"]
    degraded_frac = float(summary_rankr.get("degraded_input_fraction", 0.0))

    payload_syn = {
        "status": "rankr_synthetic_validation_completed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": {
            "n_total": summary_rankr["n_total"],
            "n_parameter_settings": len(summary_rankr["config"]["ranks"])
            * len(summary_rankr["config"]["angle_profiles"])
            * len(summary_rankr["config"]["angle_profiles"])
            * len(summary_rankr["config"]["sigma_profiles"])
            * len(summary_rankr["config"]["coefficients"])
            * len(summary_rankr["config"]["signs"]),
            "n_samples_per_setting": summary_rankr["config"]["seeds_per_setting"],
            "overall_spearman_v1": all_s["spearman_v1_score_vs_true_margin"],
            "overall_spearman_v2": all_s["spearman_v2_score_vs_true_margin"],
            "overall_spearman_gain": (
                (all_s["spearman_v2_score_vs_true_margin"] or 0.0)
                - (all_s["spearman_v1_score_vs_true_margin"] or 0.0)
            ),
            "high_overlap_positive_spearman_v1": high_s["spearman_v1_score_vs_true_margin"],
            "high_overlap_positive_spearman_v2": high_s["spearman_v2_score_vs_true_margin"],
            "high_overlap_positive_spearman_gain": (
                (high_s["spearman_v2_score_vs_true_margin"] or 0.0)
                - (high_s["spearman_v1_score_vs_true_margin"] or 0.0)
            ),
            "degraded_input_fraction": degraded_frac,
        },
    }
    args.synthetic_json.parent.mkdir(parents=True, exist_ok=True)
    args.synthetic_json.write_text(json.dumps(payload_syn, indent=2))

    _write_markdown(payload_heur, payload_syn, args.heuristic_md, args.synthetic_md)

    print(
        json.dumps(
            {
                "heuristic_json": str(args.heuristic_json),
                "synthetic_json": str(args.synthetic_json),
                "n_total": summary_rankr["n_total"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
