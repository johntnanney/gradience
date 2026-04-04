#!/usr/bin/env python3
"""
Run a tiny CPU-only reference-choice demo for the merge-aware monitor.

This script keeps the training trajectory fixed and only changes merge_target:
  1) same_task
  2) same_family
  3) cross_task

Outputs are written under:
  field_trials/merge_aware_monitor_reference_choice/runs/<reference_type>/
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn as nn

from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


def _is_lora_key(key: str) -> bool:
    return (
        (".lora_A." in key or ".lora_B." in key or key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"))
        and key.endswith(".weight")
    )


class _LoRAProj(nn.Module):
    def __init__(self, d_in: int = 8, d_out: int = 8, r: int = 4):
        super().__init__()
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)


class _SelfAttn(nn.Module):
    def __init__(self, d_in: int = 8, d_out: int = 8, r: int = 4):
        super().__init__()
        self.q_proj = _LoRAProj(d_in=d_in, d_out=d_out, r=r)


class _Layer(nn.Module):
    def __init__(self, d_in: int = 8, d_out: int = 8, r: int = 4):
        super().__init__()
        self.self_attn = _SelfAttn(d_in=d_in, d_out=d_out, r=r)


class _TinyPeftModel(nn.Module):
    def __init__(self, *, r: int = 4, alpha: float = 16.0):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.layers = nn.ModuleList([_Layer(r=r), _Layer(r=r)])
        self.peft_config = {"default": SimpleNamespace(r=r, lora_alpha=alpha)}
        self.config = SimpleNamespace(name_or_path="tiny-peft-monitor-demo")


def _mock_hf_args(output_dir: str) -> Mock:
    mock_args = Mock()
    mock_args.output_dir = output_dir
    mock_args.seed = 42
    mock_args.per_device_train_batch_size = 2
    mock_args.gradient_accumulation_steps = 1
    mock_args.max_steps = 12
    mock_args.num_train_epochs = 1.0
    mock_args.learning_rate = 5e-4
    mock_args.weight_decay = 0.01
    mock_args.adam_beta1 = 0.9
    mock_args.adam_beta2 = 0.999
    mock_args.adam_epsilon = 1e-8
    mock_args.optim = "adamw_torch"
    mock_args.fp16 = False
    mock_args.bf16 = False
    return mock_args


@dataclass(frozen=True)
class ReferenceSpec:
    name: str
    relation: str
    # reference = base + c1*d_primary + c2*d_secondary
    c1: float
    c2: float
    rationale: str
    pseudo_task: str


def _collect_base_and_directions(
    model: nn.Module,
    *,
    seed_primary: int = 11,
    seed_secondary: int = 37,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    state = model.state_dict()
    lora_keys = sorted([k for k in state.keys() if _is_lora_key(k)])
    base = {k: state[k].detach().clone() for k in lora_keys}

    gen_primary = torch.Generator(device="cpu")
    gen_primary.manual_seed(seed_primary)
    gen_secondary = torch.Generator(device="cpu")
    gen_secondary.manual_seed(seed_secondary)

    d_primary: dict[str, torch.Tensor] = {}
    d_secondary: dict[str, torch.Tensor] = {}
    for k in lora_keys:
        t = base[k]
        p = torch.randn(t.shape, generator=gen_primary, dtype=t.dtype)
        s = torch.randn(t.shape, generator=gen_secondary, dtype=t.dtype)
        p = p / (p.norm() + 1e-8)
        s = s / (s.norm() + 1e-8)
        d_primary[k] = p
        d_secondary[k] = s
    return base, d_primary, d_secondary, lora_keys


def _apply_trajectory_state(
    model: nn.Module,
    *,
    base: dict[str, torch.Tensor],
    d_primary: dict[str, torch.Tensor],
    d_secondary: dict[str, torch.Tensor],
    lora_keys: list[str],
    t: float,
) -> None:
    # fixed trajectory across all reference choices
    # c1 increases steadily (toward same-task reference)
    # c2 oscillates mildly to create realistic non-monotone perturbation
    c1 = -0.40 + 1.50 * t
    c2 = 0.25 * math.sin(2.0 * math.pi * t)

    with torch.no_grad():
        sd = model.state_dict()
        for k in lora_keys:
            sd[k].copy_(base[k] + c1 * d_primary[k] + c2 * d_secondary[k])


def _write_reference_adapter(
    out_dir: Path,
    *,
    base: dict[str, torch.Tensor],
    d_primary: dict[str, torch.Tensor],
    d_secondary: dict[str, torch.Tensor],
    lora_keys: list[str],
    spec: ReferenceSpec,
    rank: int,
    alpha: float,
) -> Path:
    ref_dir = out_dir / "reference_adapter"
    ref_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "r": int(rank),
        "lora_alpha": float(alpha),
        "target_modules": ["q_proj"],
    }
    (ref_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    weights: dict[str, torch.Tensor] = {}
    for k in lora_keys:
        weights[k] = base[k] + spec.c1 * d_primary[k] + spec.c2 * d_secondary[k]
    torch.save(weights, ref_dir / "adapter_model.bin")
    return ref_dir


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_reference_summary(reference_type: str, telemetry_rows: list[dict], relation: str, pseudo_task: str) -> dict:
    metrics_rows = [r for r in telemetry_rows if r.get("event") == "metrics"]
    init_rows = [r for r in metrics_rows if r.get("kind") == "merge_aware_monitor"]
    snap_rows = [r for r in metrics_rows if r.get("kind") == "merge_aware_compatibility"]
    summary_rows = [r for r in metrics_rows if r.get("kind") == "merge_aware_summary"]
    summary_metrics = summary_rows[-1].get("metrics", {}) if summary_rows else {}
    per_metric = summary_metrics.get("per_metric_trend", {})

    run_level = str(summary_metrics.get("run_level_trend", "inconclusive"))
    if run_level in {"toward_compatibility", "away_from_compatibility"}:
        interpretability = "interpretable"
        score = 2
    elif run_level == "mixed":
        interpretability = "mixed"
        score = 1
    else:
        interpretability = "inconclusive"
        score = 0

    return {
        "reference_type": reference_type,
        "relation": relation,
        "pseudo_task": pseudo_task,
        "snapshot_count": len(snap_rows),
        "init_event_present": bool(init_rows),
        "summary_event_present": bool(summary_rows),
        "overlap_trend": per_metric.get("mean_overlap", "inconclusive"),
        "conflict_trend": per_metric.get("conflict_fraction", "inconclusive"),
        "imbalance_trend": per_metric.get("imbalance_fraction", "inconclusive"),
        "score_trend": per_metric.get("compatibility_score", "inconclusive"),
        "run_level_summary_label": run_level,
        "trajectory_interpretability": interpretability,
        "interpretability_score": score,
    }


def main() -> int:
    torch.manual_seed(1234)
    root = Path(__file__).resolve().parent
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    training_run_id = "tiny_encoder_reference_choice_demo_v1"
    training_task = "sst2_like_sentiment_train_stub"
    base_model_name = "tiny-peft-monitor-demo"
    rank = 4
    alpha = 16.0
    trajectory_steps = [1, 2, 3, 4, 5, 6]
    t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    refs = [
        ReferenceSpec(
            name="same_task",
            relation="same_task",
            c1=1.05,
            c2=0.05,
            rationale="Reference aligned with the primary training trajectory direction.",
            pseudo_task="sst2_like_sentiment_reference",
        ),
        ReferenceSpec(
            name="same_family",
            relation="same_family",
            c1=0.55,
            c2=0.55,
            rationale="Reference mixes primary and secondary directions, emulating family-level partial overlap.",
            pseudo_task="imdb_like_sentiment_reference",
        ),
        ReferenceSpec(
            name="cross_task",
            relation="cross_task",
            c1=-0.75,
            c2=0.10,
            rationale="Reference opposes the primary trajectory direction, emulating cross-task mismatch.",
            pseudo_task="ag_news_like_topic_reference",
        ),
    ]

    model_for_basis = _TinyPeftModel(r=rank, alpha=alpha)
    base, d_primary, d_secondary, lora_keys = _collect_base_and_directions(model_for_basis)

    design_json = {
        "training_run_id": training_run_id,
        "training_task": training_task,
        "base_model": base_model_name,
        "trajectory_steps": trajectory_steps,
        "reference_same_task": refs[0].__dict__,
        "reference_same_family": refs[1].__dict__,
        "reference_cross_task": refs[2].__dict__,
        "notes": "Synthetic relation-labeled references for bounded interpretability demo only.",
    }
    (root / "demo_design.json").write_text(json.dumps(design_json, indent=2), encoding="utf-8")

    run_manifest: dict[str, object] = {
        "study": "merge_aware_monitor_reference_choice",
        "training_run_id": training_run_id,
        "training_task": training_task,
        "base_model": base_model_name,
        "runs": [],
    }
    summary_rows: list[dict] = []

    for spec in refs:
        run_dir = runs_root / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = run_dir / "run.jsonl"

        model = _TinyPeftModel(r=rank, alpha=alpha)
        ref_dir = _write_reference_adapter(
            run_dir,
            base=base,
            d_primary=d_primary,
            d_secondary=d_secondary,
            lora_keys=lora_keys,
            spec=spec,
            rank=rank,
            alpha=alpha,
        )

        callback = GradienceCallback(
            GradienceCallbackConfig(
                output_dir=str(run_dir),
                filename=telemetry_path.name,
                merge_target=str(ref_dir),
                merge_monitor_every=1,
                enable_guard=False,
                enable_monitor=False,
            )
        )

        args = _mock_hf_args(str(run_dir))
        state = Mock()
        control = Mock()
        state.global_step = 0
        callback.on_train_begin(args, state, control, model=model)

        for step, t in zip(trajectory_steps, t_values):
            _apply_trajectory_state(
                model,
                base=base,
                d_primary=d_primary,
                d_secondary=d_secondary,
                lora_keys=lora_keys,
                t=t,
            )
            state.global_step = int(step)
            callback.on_log(
                args,
                state,
                control,
                logs={"loss": float(1.2 - 0.05 * step), "learning_rate": 5e-4, "grad_norm": float(1.0 + 0.1 * step)},
                model=model,
            )

        callback.on_train_end(args, state, control, model=model)

        rows = _load_jsonl(telemetry_path)
        summary = _build_reference_summary(
            reference_type=spec.name,
            telemetry_rows=rows,
            relation=spec.relation,
            pseudo_task=spec.pseudo_task,
        )
        summary_rows.append(summary)

        run_manifest["runs"].append(
            {
                "reference_type": spec.name,
                "relation": spec.relation,
                "reference_adapter": str(ref_dir.relative_to(root)),
                "telemetry_path": str(telemetry_path.relative_to(root)),
                "snapshot_count": summary["snapshot_count"],
                "run_level_summary_label": summary["run_level_summary_label"],
                "trajectory_interpretability": summary["trajectory_interpretability"],
            }
        )

    # Stable ordering by reference type.
    summary_rows = sorted(summary_rows, key=lambda r: str(r.get("reference_type", "")))
    (root / "runs_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    (root / "reference_choice_summary.json").write_text(
        json.dumps(
            {
                "study": "merge_aware_monitor_reference_choice",
                "training_run_id": training_run_id,
                "summary_rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Lightweight comparison outcome
    by_type = {row["reference_type"]: row for row in summary_rows}
    expected_order = ["same_task", "same_family", "cross_task"]
    observed_scores = [int(by_type.get(k, {}).get("interpretability_score", -1)) for k in expected_order]
    monotone_desc = observed_scores == sorted(observed_scores, reverse=True)

    decision = {
        "study": "merge_aware_monitor_reference_choice",
        "outcome": "same_task_preferred" if monotone_desc and observed_scores[0] >= observed_scores[-1] else "mixed",
        "observed_interpretability_scores": {
            k: int(by_type.get(k, {}).get("interpretability_score", -1)) for k in expected_order
        },
        "monotone_same_task_to_cross_task": monotone_desc,
        "notes": "Tiny synthetic demo; interpretability guidance only.",
    }
    (root / "decision_summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
