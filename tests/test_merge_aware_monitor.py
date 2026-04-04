from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from gradience.vnext.integrations.merge_aware_monitor import (
    MergeAwareTrainingMonitor,
    summarize_merge_aware_trends,
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


def _write_reference_adapter(path: Path, *, source_state_dict: dict[str, torch.Tensor], r: int, alpha: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "r": int(r),
        "lora_alpha": float(alpha),
        "target_modules": ["q_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    ref_weights: dict[str, torch.Tensor] = {}
    for k, v in source_state_dict.items():
        if ".lora_A." in k or ".lora_B." in k:
            # Nudge weights slightly so comparison is non-degenerate.
            ref_weights[k] = (v.detach().cpu() * 0.95).clone()
    torch.save(ref_weights, path / "adapter_model.bin")


def test_snapshot_from_model_emits_shared_layer_summary() -> None:
    model = _TinyPeftModel(r=4, alpha=16.0)
    with tempfile.TemporaryDirectory() as td:
        ref_dir = Path(td) / "ref_adapter"
        _write_reference_adapter(ref_dir, source_state_dict=model.state_dict(), r=4, alpha=16.0)

        monitor = MergeAwareTrainingMonitor(ref_dir)
        snapshot = monitor.snapshot_from_model(model)

    assert snapshot.status == "ok"
    assert snapshot.n_shared_layers >= 1
    assert 0.0 <= snapshot.mean_overlap <= 1.0
    assert 0.0 <= snapshot.conflict_fraction <= 1.0
    assert 0.0 <= snapshot.imbalance_fraction <= 1.0
    assert snapshot.overall_verdict


def test_snapshot_no_shared_layers_is_bounded() -> None:
    model = _TinyPeftModel(r=4, alpha=16.0)
    with tempfile.TemporaryDirectory() as td:
        ref_dir = Path(td) / "ref_adapter"
        _write_reference_adapter(ref_dir, source_state_dict=model.state_dict(), r=4, alpha=16.0)

        monitor = MergeAwareTrainingMonitor(ref_dir)
        unrelated_state = {
            "other.module.lora_A.weight": torch.randn(4, 8),
            "other.module.lora_B.weight": torch.randn(8, 4),
        }
        snapshot = monitor.snapshot_from_state_dict(unrelated_state)

    assert snapshot.status == "no_shared_layers"
    assert snapshot.n_shared_layers == 0
    assert snapshot.compatibility_score == 0.0
    assert snapshot.overall_verdict == "inconclusive"


def test_summarize_merge_aware_trends_toward_compatibility() -> None:
    snaps = [
        {
            "step": 10,
            "mean_overlap": 0.20,
            "conflict_fraction": 0.60,
            "imbalance_fraction": 0.40,
            "compatibility_score": 0.10,
        },
        {
            "step": 20,
            "mean_overlap": 0.35,
            "conflict_fraction": 0.35,
            "imbalance_fraction": 0.30,
            "compatibility_score": 0.40,
        },
        {
            "step": 30,
            "mean_overlap": 0.50,
            "conflict_fraction": 0.20,
            "imbalance_fraction": 0.15,
            "compatibility_score": 0.70,
        },
    ]
    summary = summarize_merge_aware_trends(snaps)

    assert summary["run_level_trend"] == "toward_compatibility"
    assert summary["n_snapshots"] == 3
    assert summary["start_step"] == 10
    assert summary["end_step"] == 30
