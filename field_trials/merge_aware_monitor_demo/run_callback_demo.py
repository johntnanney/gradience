#!/usr/bin/env python3
"""
Generate a bounded diagnostic-only merge-aware monitor demo run.

This is intentionally lightweight and CPU-friendly:
- no optimizer interaction
- no loss/gradient intervention
- callback lifecycle only
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn as nn

from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


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
        self.config = SimpleNamespace(name_or_path="tiny-peft-model")


def _mock_hf_args(output_dir: str) -> Mock:
    mock_args = Mock()
    mock_args.output_dir = output_dir
    mock_args.seed = 42
    mock_args.per_device_train_batch_size = 2
    mock_args.gradient_accumulation_steps = 1
    mock_args.max_steps = 10
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


def _write_reference_adapter(path: Path, *, state_dict: dict[str, torch.Tensor], r: int, alpha: float) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "r": int(r),
        "lora_alpha": float(alpha),
        "target_modules": ["q_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    ref_weights: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if ".lora_A." in k or ".lora_B." in k:
            ref_weights[k] = (v.detach().cpu() * 0.92).clone()
    torch.save(ref_weights, path / "adapter_model.bin")
    return path


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _TinyPeftModel(r=4, alpha=16.0)
    ref_dir = _write_reference_adapter(out_dir / "reference_adapter", state_dict=model.state_dict(), r=4, alpha=16.0)
    telemetry_path = out_dir / "demo_run.jsonl"

    callback = GradienceCallback(
        GradienceCallbackConfig(
            output_dir=str(out_dir),
            filename=telemetry_path.name,
            merge_target=str(ref_dir),
            merge_monitor_every=1,
        )
    )

    args = _mock_hf_args(str(out_dir))
    state = Mock()
    control = Mock()

    state.global_step = 0
    callback.on_train_begin(args, state, control, model=model)

    log_steps = (1, 2, 3, 4)
    for step in log_steps:
        state.global_step = step
        callback.on_log(
            args,
            state,
            control,
            logs={"loss": 1.0 / step, "learning_rate": 5e-4, "grad_norm": 1.0 + 0.1 * step},
            model=model,
        )

    callback.on_train_end(args, state, control, model=model)

    events = _load_jsonl(telemetry_path)
    compat = [e for e in events if e.get("event") == "metrics" and e.get("kind") == "merge_aware_compatibility"]
    summary = [e for e in events if e.get("event") == "metrics" and e.get("kind") == "merge_aware_summary"]
    summary_metrics = summary[-1]["metrics"] if summary else {}

    try:
        telemetry_ref = str(telemetry_path.relative_to(Path.cwd()))
    except ValueError:
        telemetry_ref = str(telemetry_path)
    try:
        reference_ref = str(ref_dir.relative_to(Path.cwd()))
    except ValueError:
        reference_ref = str(ref_dir)

    manifest = {
        "study": "merge_aware_training_monitor_demo",
        "prototype_mode": "diagnostic_only",
        "runs": [
            {
                "run_id": events[0].get("run_id") if events else None,
                "telemetry_path": telemetry_ref,
                "reference_adapter": reference_ref,
                "log_steps": list(log_steps),
                "compatibility_snapshot_count": len(compat),
                "run_level_trend": summary_metrics.get("run_level_trend"),
                "status": "completed",
            }
        ],
    }
    (root / "demo_runs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    decision_summary = {
        "status": "bounded_keep",
        "basis": "prototype demonstrates callback-compatible telemetry drift traces with conservative trend labels",
        "n_runs": 1,
        "n_snapshots_total": len(compat),
        "run_level_trends": [summary_metrics.get("run_level_trend")] if summary_metrics else [],
    }
    (root / "decision_summary.json").write_text(json.dumps(decision_summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
