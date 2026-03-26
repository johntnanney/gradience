"""Run a repeatable core-space benchmark on fixed pair archetype fixtures.

This harness is diagnostic-first. It does not alter merge recommendation logic.
It provides a reproducible bundle to support feature-status decisions:
`retain_optional`, `promote_advanced`, or `shelve`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from gradience.vnext.audit import find_peft_files
from gradience.vnext.merge.core_space import (
    aggregate_core_space_diagnostics,
    compute_core_space_layer_diagnostic,
)
from gradience.vnext.merge.io import extract_factors, load_adapter, match_layers

_VALID_STATUS = frozenset({"compatible", "marginal", "incompatible", "not_applicable"})
_VALID_EVIDENCE_TIERS = frozenset({"synthetic", "realistic"})
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    category: str
    description: str
    evidence_tier: str
    seeds: tuple[int, ...]
    n_layers: int
    d_in: int
    d_out: int
    r_a: int
    r_b: int
    alpha_a: float
    alpha_b: float
    generator: dict[str, Any]
    expected_dominant_statuses: tuple[str, ...]
    expected_trial_statuses: tuple[str, ...]


def _load_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _require_non_empty_str(raw: Any, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Field '{field}' must be a non-empty string")
    return raw.strip()


def _require_positive_int(raw: Any, field: str) -> int:
    if not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"Field '{field}' must be a positive integer")
    return raw


def _require_positive_float(raw: Any, field: str) -> float:
    if not isinstance(raw, (int, float)) or float(raw) <= 0.0:
        raise ValueError(f"Field '{field}' must be positive numeric")
    return float(raw)


def _require_status_list(raw: Any, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Field '{field}' must be a non-empty list")
    values: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"Field '{field}[{i}]' must be a string")
        if item not in _VALID_STATUS:
            raise ValueError(f"Field '{field}[{i}]' has invalid status '{item}'")
        values.append(item)
    return tuple(values)


def _require_seed_list(raw: Any, field: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Field '{field}' must be a non-empty list of integers")
    seeds: list[int] = []
    for i, item in enumerate(raw):
        if not isinstance(item, int):
            raise ValueError(f"Field '{field}[{i}]' must be an integer")
        seeds.append(item)
    return tuple(seeds)


def _load_fixture(path: Path) -> FixtureSpec:
    d = _load_json_object(path)

    fixture_id = _require_non_empty_str(d.get("fixture_id"), "fixture_id")
    category = _require_non_empty_str(d.get("category"), "category")
    description = _require_non_empty_str(d.get("description"), "description")
    evidence_tier = _require_non_empty_str(d.get("evidence_tier"), "evidence_tier")
    if evidence_tier not in _VALID_EVIDENCE_TIERS:
        raise ValueError(f"Field 'evidence_tier' must be one of {sorted(_VALID_EVIDENCE_TIERS)}")

    seeds = _require_seed_list(d.get("seeds"), "seeds")
    n_layers = _require_positive_int(d.get("n_layers"), "n_layers")
    d_in = _require_positive_int(d.get("d_in"), "d_in")
    d_out = _require_positive_int(d.get("d_out"), "d_out")
    r_a = _require_positive_int(d.get("r_a"), "r_a")
    r_b = _require_positive_int(d.get("r_b"), "r_b")
    alpha_a = _require_positive_float(d.get("alpha_a"), "alpha_a")
    alpha_b = _require_positive_float(d.get("alpha_b"), "alpha_b")

    generator = d.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("Field 'generator' must be an object")
    mode = generator.get("mode")
    if mode not in {
        "same_domain_noise",
        "blend_random",
        "random_independent",
        "rank_mismatch_random",
        "real_adapter_pair",
    }:
        raise ValueError(
            "Field 'generator.mode' must be one of "
            "['same_domain_noise', 'blend_random', 'random_independent', "
            "'rank_mismatch_random', 'real_adapter_pair']"
        )

    expected_dominant = _require_status_list(d.get("expected_dominant_statuses"), "expected_dominant_statuses")
    expected_trial = _require_status_list(d.get("expected_trial_statuses"), "expected_trial_statuses")

    return FixtureSpec(
        fixture_id=fixture_id,
        category=category,
        description=description,
        evidence_tier=evidence_tier,
        seeds=seeds,
        n_layers=n_layers,
        d_in=d_in,
        d_out=d_out,
        r_a=r_a,
        r_b=r_b,
        alpha_a=alpha_a,
        alpha_b=alpha_b,
        generator=generator,
        expected_dominant_statuses=expected_dominant,
        expected_trial_statuses=expected_trial,
    )


def _discover_fixtures(fixtures_root: Path, selected: list[str] | None) -> list[Path]:
    all_files = sorted(fixtures_root.glob("*.json"))
    if not selected:
        return all_files
    wanted = set(selected)
    chosen = [p for p in all_files if p.stem in wanted]
    missing = sorted(wanted - {p.stem for p in chosen})
    if missing:
        raise FileNotFoundError(f"Unknown fixture(s): {', '.join(missing)}")
    return chosen


def _rand_factors(r: int, d_in: int, d_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(r, d_in), torch.randn(d_out, r)


def _generate_pair_factors(
    spec: FixtureSpec,
    *,
    A_a: torch.Tensor,
    B_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    mode = str(spec.generator["mode"])
    if mode == "same_domain_noise":
        noise_scale = float(spec.generator.get("noise_scale", 0.03))
        A_b = A_a + noise_scale * torch.randn_like(A_a)
        B_b = B_a + noise_scale * torch.randn_like(B_a)
        return A_b, B_b, spec.r_b, spec.alpha_b

    if mode == "blend_random":
        blend_weight = float(spec.generator.get("blend_weight", 0.5))
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}")
        A_r, B_r = _rand_factors(spec.r_b, spec.d_in, spec.d_out)
        A_b = blend_weight * A_a + (1.0 - blend_weight) * A_r
        B_b = blend_weight * B_a + (1.0 - blend_weight) * B_r
        return A_b, B_b, spec.r_b, spec.alpha_b

    if mode == "random_independent":
        A_b, B_b = _rand_factors(spec.r_b, spec.d_in, spec.d_out)
        scale = float(spec.generator.get("scale", 1.0))
        if scale != 1.0:
            A_b = A_b * scale
            B_b = B_b * scale
        return A_b, B_b, spec.r_b, spec.alpha_b

    if mode == "rank_mismatch_random":
        A_b, B_b = _rand_factors(spec.r_b, spec.d_in, spec.d_out)
        scale = float(spec.generator.get("scale", 1.0))
        if scale != 1.0:
            A_b = A_b * scale
            B_b = B_b * scale
        return A_b, B_b, spec.r_b, spec.alpha_b

    raise ValueError(f"Unsupported generator mode: {mode}")


def _resolve_repo_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _check_real_adapter_fixture_availability(spec: FixtureSpec) -> tuple[bool, str | None]:
    if str(spec.generator["mode"]) != "real_adapter_pair":
        return True, None

    adapter_a_raw = spec.generator.get("adapter_a_dir")
    adapter_b_raw = spec.generator.get("adapter_b_dir")
    if not isinstance(adapter_a_raw, str) or not adapter_a_raw.strip():
        return False, "missing generator.adapter_a_dir"
    if not isinstance(adapter_b_raw, str) or not adapter_b_raw.strip():
        return False, "missing generator.adapter_b_dir"

    problems: list[str] = []
    for label, raw in (("adapter_a_dir", adapter_a_raw), ("adapter_b_dir", adapter_b_raw)):
        path = _resolve_repo_path(raw.strip())
        if not path.is_dir():
            problems.append(f"{label} not found: {path}")
            continue
        cfg, weights, _issues = find_peft_files(path)
        if cfg is None:
            problems.append(f"{label} missing adapter_config.*: {path}")
        if weights is None:
            problems.append(f"{label} missing adapter_model.*: {path}")

    if problems:
        return False, "; ".join(problems)
    return True, None


def _run_real_adapter_trial(spec: FixtureSpec, seed: int) -> dict[str, Any]:
    adapter_a_raw = spec.generator.get("adapter_a_dir")
    adapter_b_raw = spec.generator.get("adapter_b_dir")
    if not isinstance(adapter_a_raw, str) or not adapter_a_raw.strip():
        raise ValueError("real_adapter_pair fixtures require generator.adapter_a_dir")
    if not isinstance(adapter_b_raw, str) or not adapter_b_raw.strip():
        raise ValueError("real_adapter_pair fixtures require generator.adapter_b_dir")

    adapter_a_dir = _resolve_repo_path(adapter_a_raw.strip())
    adapter_b_dir = _resolve_repo_path(adapter_b_raw.strip())
    if not adapter_a_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_a_dir}")
    if not adapter_b_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_b_dir}")

    info_a = load_adapter(adapter_a_dir)
    info_b = load_adapter(adapter_b_dir)

    # Keep fixture metadata honest relative to the source adapters.
    if spec.r_a != info_a.rank:
        raise ValueError(f"{spec.fixture_id}: fixture r_a={spec.r_a} does not match adapter A rank={info_a.rank}")
    if spec.r_b != info_b.rank:
        raise ValueError(f"{spec.fixture_id}: fixture r_b={spec.r_b} does not match adapter B rank={info_b.rank}")
    if abs(spec.alpha_a - float(info_a.alpha)) > 1e-6:
        raise ValueError(
            f"{spec.fixture_id}: fixture alpha_a={spec.alpha_a} does not match adapter A alpha={float(info_a.alpha)}"
        )
    if abs(spec.alpha_b - float(info_b.alpha)) > 1e-6:
        raise ValueError(
            f"{spec.fixture_id}: fixture alpha_b={spec.alpha_b} does not match adapter B alpha={float(info_b.alpha)}"
        )

    shared, _, _ = match_layers(info_a, info_b)
    if not shared:
        raise ValueError(f"{spec.fixture_id}: adapters have no shared LoRA layers")

    max_layers_raw = spec.generator.get("max_layers", len(shared))
    if not isinstance(max_layers_raw, int) or max_layers_raw <= 0:
        raise ValueError(f"{spec.fixture_id}: generator.max_layers must be a positive integer")
    max_layers = min(max_layers_raw, len(shared))

    if max_layers < len(shared):
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(shared), generator=gen).tolist()
        selected = [shared[i] for i in perm[:max_layers]]
    else:
        selected = list(shared)

    layer_diagnostics = []
    start = perf_counter()
    for module_prefix in selected:
        A_a, B_a, r_a = extract_factors(info_a, module_prefix)
        A_b, B_b, r_b = extract_factors(info_b, module_prefix)
        diag = compute_core_space_layer_diagnostic(
            layer_name=module_prefix,
            A_a=A_a,
            B_a=B_a,
            alpha_a=info_a.alpha,
            r_a=r_a,
            A_b=A_b,
            B_b=B_b,
            alpha_b=info_b.alpha,
            r_b=r_b,
            compute_dtype=torch.float64,
        )
        layer_diagnostics.append(diag)
    elapsed_ms = (perf_counter() - start) * 1000.0

    pair_diag = aggregate_core_space_diagnostics(layer_diagnostics)
    layer_status_counts = Counter(ld.status for ld in layer_diagnostics)

    return {
        "seed": seed,
        "status": pair_diag.status,
        "shared_basis_score_mean": pair_diag.shared_basis_score_mean,
        "basis_distortion_mean": pair_diag.basis_distortion_mean,
        "effective_shared_rank_median": pair_diag.effective_shared_rank_median,
        "n_layers_scored": pair_diag.n_layers_scored,
        "n_layers_incompatible": pair_diag.n_layers_incompatible,
        "layer_status_counts": dict(layer_status_counts),
        "runtime_ms": elapsed_ms,
        "runtime_ms_per_layer": elapsed_ms / float(max_layers),
    }


def _run_trial(spec: FixtureSpec, seed: int) -> dict[str, Any]:
    if str(spec.generator["mode"]) == "real_adapter_pair":
        return _run_real_adapter_trial(spec, seed)

    torch.manual_seed(seed)
    layer_diagnostics = []

    start = perf_counter()
    for i in range(spec.n_layers):
        A_a, B_a = _rand_factors(spec.r_a, spec.d_in, spec.d_out)
        A_b, B_b, r_b, alpha_b = _generate_pair_factors(spec, A_a=A_a, B_a=B_a)
        diag = compute_core_space_layer_diagnostic(
            layer_name=f"layer_{i}",
            A_a=A_a,
            B_a=B_a,
            alpha_a=spec.alpha_a,
            r_a=spec.r_a,
            A_b=A_b,
            B_b=B_b,
            alpha_b=alpha_b,
            r_b=r_b,
            compute_dtype=torch.float64,
        )
        layer_diagnostics.append(diag)
    elapsed_ms = (perf_counter() - start) * 1000.0

    pair_diag = aggregate_core_space_diagnostics(layer_diagnostics)
    layer_status_counts = Counter(ld.status for ld in layer_diagnostics)

    return {
        "seed": seed,
        "status": pair_diag.status,
        "shared_basis_score_mean": pair_diag.shared_basis_score_mean,
        "basis_distortion_mean": pair_diag.basis_distortion_mean,
        "effective_shared_rank_median": pair_diag.effective_shared_rank_median,
        "n_layers_scored": pair_diag.n_layers_scored,
        "n_layers_incompatible": pair_diag.n_layers_incompatible,
        "layer_status_counts": dict(layer_status_counts),
        "runtime_ms": elapsed_ms,
        "runtime_ms_per_layer": elapsed_ms / float(spec.n_layers),
    }


def _dominant_status(counter: Counter[str]) -> str:
    if not counter:
        return "not_applicable"
    top = max(counter.values())
    tied = sorted([status for status, count in counter.items() if count == top])
    return tied[0]


def _summarize_fixture(spec: FixtureSpec, trial_rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counter = Counter(str(row["status"]) for row in trial_rows)
    dominant_status = _dominant_status(status_counter)
    trial_statuses = sorted(status_counter.keys())

    mean_score = sum(float(row["shared_basis_score_mean"]) for row in trial_rows) / len(trial_rows)
    mean_distortion = sum(float(row["basis_distortion_mean"]) for row in trial_rows) / len(trial_rows)
    mean_rank = sum(float(row["effective_shared_rank_median"]) for row in trial_rows) / len(trial_rows)
    mean_runtime_per_layer = sum(float(row["runtime_ms_per_layer"]) for row in trial_rows) / len(trial_rows)

    dominant_match = dominant_status in set(spec.expected_dominant_statuses)
    trial_status_match = all(status in set(spec.expected_trial_statuses) for status in trial_statuses)

    if dominant_match and trial_status_match:
        verdict = "passed"
    elif dominant_match or trial_status_match:
        verdict = "partially passed"
    else:
        verdict = "unexpected_status_pattern"

    return {
        "fixture_id": spec.fixture_id,
        "category": spec.category,
        "description": spec.description,
        "evidence_tier": spec.evidence_tier,
        "expected_dominant_statuses": list(spec.expected_dominant_statuses),
        "expected_trial_statuses": list(spec.expected_trial_statuses),
        "status_counts": dict(status_counter),
        "dominant_status": dominant_status,
        "trial_statuses": trial_statuses,
        "trial_count": len(trial_rows),
        "shared_basis_score_mean": mean_score,
        "basis_distortion_mean": mean_distortion,
        "effective_shared_rank_mean": mean_rank,
        "runtime_ms_per_layer_mean": mean_runtime_per_layer,
        "dominant_match": dominant_match,
        "trial_status_match": trial_status_match,
        "verdict": verdict,
        "available": True,
        "decision_eligible": True,
        "unavailable_reason": None,
    }


def _unavailable_fixture_summary(spec: FixtureSpec, reason: str) -> dict[str, Any]:
    return {
        "fixture_id": spec.fixture_id,
        "category": spec.category,
        "description": spec.description,
        "evidence_tier": spec.evidence_tier,
        "expected_dominant_statuses": list(spec.expected_dominant_statuses),
        "expected_trial_statuses": list(spec.expected_trial_statuses),
        "status_counts": {},
        "dominant_status": "not_applicable",
        "trial_statuses": ["not_applicable"],
        "trial_count": 0,
        "shared_basis_score_mean": 0.0,
        "basis_distortion_mean": 0.0,
        "effective_shared_rank_mean": 0.0,
        "runtime_ms_per_layer_mean": 0.0,
        "dominant_match": False,
        "trial_status_match": False,
        "verdict": "unavailable",
        "available": False,
        "decision_eligible": False,
        "unavailable_reason": reason,
    }


def _evaluate_promotion_rubric(fixture_rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [row for row in fixture_rows if bool(row.get("decision_eligible", True))]
    if not decision_rows:
        raise ValueError("No decision-eligible fixtures are available for rubric evaluation")

    by_id = {row["fixture_id"]: row for row in decision_rows}

    same_domain = by_id.get("balanced_same_domain")
    cross_domain = by_id.get("balanced_cross_domain")
    moderate = by_id.get("moderate_risk_pairs")
    distant = by_id.get("same_rank_semantically_distant")
    mismatch = by_id.get("deliberately_mismatched_random")

    c1_same_domain = same_domain is not None and same_domain["dominant_status"] == "compatible"
    c2_distant_separation = (
        distant is not None
        and mismatch is not None
        and distant["dominant_status"] in {"marginal", "incompatible"}
        and mismatch["dominant_status"] in {"marginal", "incompatible"}
        and (
            distant["dominant_status"] == "incompatible"
            or mismatch["dominant_status"] == "incompatible"
        )
    )
    c3_ambiguous_band = (
        cross_domain is not None
        and moderate is not None
        and cross_domain["dominant_status"] in {"compatible", "marginal"}
        and moderate["dominant_status"] in {"marginal", "incompatible"}
    )
    dominant_statuses = {row["dominant_status"] for row in decision_rows}
    c4_status_spread = len(dominant_statuses) >= 2
    max_runtime_per_layer = max(float(row["runtime_ms_per_layer_mean"]) for row in decision_rows)
    c5_runtime = max_runtime_per_layer <= 2.5

    realistic_fixture_count = sum(1 for row in decision_rows if row["evidence_tier"] == "realistic")
    total_realistic_fixture_count = sum(1 for row in fixture_rows if row["evidence_tier"] == "realistic")
    unavailable_realistic_count = sum(
        1 for row in fixture_rows if row["evidence_tier"] == "realistic" and not bool(row.get("available", True))
    )
    all_gate = c1_same_domain and c2_distant_separation and c3_ambiguous_band and c4_status_spread and c5_runtime

    if not (c1_same_domain and c2_distant_separation and c4_status_spread):
        decision = "shelve"
    elif all_gate and realistic_fixture_count >= 2:
        decision = "promote_advanced"
    else:
        decision = "retain_optional"

    return {
        "decision": decision,
        "criteria": {
            "same_domain_compatible": c1_same_domain,
            "distant_and_mismatch_downgraded": c2_distant_separation,
            "ambiguous_band_behavior": c3_ambiguous_band,
            "status_spread_nontrivial": c4_status_spread,
            "runtime_within_budget": c5_runtime,
            "realistic_fixture_count_available": realistic_fixture_count,
            "realistic_fixture_count_total": total_realistic_fixture_count,
            "realistic_fixture_count_unavailable": unavailable_realistic_count,
        },
        "dominant_statuses": sorted(dominant_statuses),
        "max_runtime_ms_per_layer": max_runtime_per_layer,
    }


def _write_markdown_summary(out_path: Path, run_label: str, fixture_rows: list[dict[str, Any]], rubric: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Core-Space Benchmark Summary")
    lines.append("")
    lines.append(f"- Run label: `{run_label}`")
    lines.append(f"- Decision: **{rubric['decision']}**")
    lines.append("")
    lines.append(
        "| Fixture | Category | Availability | Dominant Status | Trial Statuses | "
        "Score Mean | Distortion Mean | Rank Mean | Runtime ms/layer | Verdict |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|")
    for row in fixture_rows:
        availability = "available" if row.get("available", True) else "unavailable"
        lines.append(
            f"| {row['fixture_id']} | {row['category']} | {availability} | {row['dominant_status']} | "
            f"{', '.join(row['trial_statuses'])} | "
            f"{row['shared_basis_score_mean']:.3f} | {row['basis_distortion_mean']:.3f} | "
            f"{row['effective_shared_rank_mean']:.2f} | {row['runtime_ms_per_layer_mean']:.3f} | {row['verdict']} |"
        )
    lines.append("")

    unavailable_rows = [row for row in fixture_rows if not bool(row.get("available", True))]
    if unavailable_rows:
        lines.append("## Unavailable Fixtures")
        lines.append("")
        for row in unavailable_rows:
            lines.append(f"- `{row['fixture_id']}`: {row.get('unavailable_reason', 'missing inputs')}")
        lines.append("")

    lines.append("## Rubric Criteria")
    lines.append("")
    for key, passed in rubric["criteria"].items():
        lines.append(f"- `{key}`: {'pass' if passed else 'fail'}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed core-space benchmark fixtures and emit a decision bundle.")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("examples/core_space_benchmark"),
        help="Root directory containing core-space fixture JSON files (default: examples/core_space_benchmark)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/core_space_benchmark"),
        help="Root directory for benchmark output bundles (default: results/core_space_benchmark)",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=None,
        help="Specific fixture stem to run. Repeatable. Default: run all fixtures.",
    )
    parser.add_argument(
        "--require-realistic-fixtures",
        action="store_true",
        help=(
            "Fail if any realistic fixture inputs are missing. Default behavior is to emit "
            "explicit 'unavailable' fixture rows and continue."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures_root: Path = args.fixtures_root
    out_root: Path = args.out_root

    if not fixtures_root.is_dir():
        raise FileNotFoundError(f"Fixtures root does not exist: {fixtures_root}")

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = out_root / run_label
    run_out.mkdir(parents=True, exist_ok=True)

    fixture_files = _discover_fixtures(fixtures_root, args.fixture)
    if not fixture_files:
        raise ValueError(f"No fixtures found under {fixtures_root}")

    fixture_rows: list[dict[str, Any]] = []
    for fixture_file in fixture_files:
        spec = _load_fixture(fixture_file)
        available, unavailable_reason = _check_real_adapter_fixture_availability(spec)
        if not available:
            if bool(getattr(args, "require_realistic_fixtures", False)) and spec.evidence_tier == "realistic":
                raise FileNotFoundError(
                    f"Required realistic fixture '{spec.fixture_id}' unavailable: {unavailable_reason}"
                )
            trial_rows = []
            fixture_summary = _unavailable_fixture_summary(
                spec,
                unavailable_reason or "missing realistic fixture inputs",
            )
        else:
            trial_rows = [_run_trial(spec, seed) for seed in spec.seeds]
            fixture_summary = _summarize_fixture(spec, trial_rows)
        fixture_rows.append(fixture_summary)

        fixture_out = run_out / spec.fixture_id
        fixture_out.mkdir(parents=True, exist_ok=True)
        with open(fixture_out / "fixture_spec.json", "w", encoding="utf-8") as f:
            json.dump(_load_json_object(fixture_file), f, indent=2)
            f.write("\n")
        with open(fixture_out / "trial_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fixture_id": spec.fixture_id,
                    "available": available,
                    "unavailable_reason": unavailable_reason,
                    "trials": trial_rows,
                },
                f,
                indent=2,
            )
            f.write("\n")
        with open(fixture_out / "fixture_summary.json", "w", encoding="utf-8") as f:
            json.dump(fixture_summary, f, indent=2)
            f.write("\n")

    fixture_rows = sorted(fixture_rows, key=lambda row: row["fixture_id"])
    rubric = _evaluate_promotion_rubric(fixture_rows)

    summary = {
        "run_label": run_label,
        "fixtures": fixture_rows,
        "promotion_rubric": rubric,
    }
    with open(run_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    with open(run_out / "decision.json", "w", encoding="utf-8") as f:
        json.dump(rubric, f, indent=2)
        f.write("\n")

    _write_markdown_summary(run_out / "summary.md", run_label, fixture_rows, rubric)

    print(f"Core-space benchmark run: {run_label}")
    print(f"Output directory: {run_out}")
    for row in fixture_rows:
        if row.get("available", True):
            detail = f"dominant={row['dominant_status']}, statuses={','.join(row['trial_statuses'])}"
        else:
            detail = f"unavailable ({row.get('unavailable_reason', 'missing inputs')})"
        print(
            f"- {row['fixture_id']}: {detail}, verdict={row['verdict']}"
        )
    print(f"Decision: {rubric['decision']}")


if __name__ == "__main__":
    main()
