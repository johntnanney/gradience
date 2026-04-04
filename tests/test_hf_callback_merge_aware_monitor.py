from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    transformers = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    torch is None or nn is None or transformers is None,
    reason="Requires torch + transformers",
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
        self.config = SimpleNamespace(name_or_path="tiny-peft-model")


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
            ref_weights[k] = (v.detach().cpu() * 0.9).clone()
    torch.save(ref_weights, path / "adapter_model.bin")


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


def test_hf_callback_emits_merge_aware_compatibility_and_summary() -> None:
    from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        model = _TinyPeftModel(r=4, alpha=16.0)
        ref_dir = Path(td) / "ref_adapter"
        _write_reference_adapter(ref_dir, source_state_dict=model.state_dict(), r=4, alpha=16.0)

        config = GradienceCallbackConfig(
            output_dir=str(output_dir),
            filename="run.jsonl",
            merge_target=str(ref_dir),
            merge_monitor_every=1,
        )
        callback = GradienceCallback(config)

        mock_args = _mock_hf_args(str(output_dir))
        mock_state = Mock()
        mock_control = Mock()

        mock_state.global_step = 0
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)

        for step in (1, 2, 3):
            mock_state.global_step = step
            callback.on_log(
                mock_args,
                mock_state,
                mock_control,
                logs={"loss": 1.0 / step, "learning_rate": 5e-4, "grad_norm": 1.0 + step},
                model=model,
            )

        callback.on_train_end(mock_args, mock_state, mock_control, model=model)

        events = []
        with (output_dir / "run.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

    kinds = [e.get("kind") for e in events if e.get("event") == "metrics"]
    assert "merge_aware_monitor" in kinds
    assert "merge_aware_compatibility" in kinds
    assert "merge_aware_summary" in kinds

    compatibility_events = [
        e for e in events if e.get("event") == "metrics" and e.get("kind") == "merge_aware_compatibility"
    ]
    assert len(compatibility_events) >= 1
    assert all(e.get("metrics", {}).get("reference_adapter") for e in compatibility_events)

    summary_events = [e for e in events if e.get("event") == "metrics" and e.get("kind") == "merge_aware_summary"]
    assert len(summary_events) == 1
    summary = summary_events[0].get("metrics", {})
    assert summary.get("n_snapshots", 0) >= 1
    assert summary.get("run_level_trend") in {"toward_compatibility", "away_from_compatibility", "mixed", "inconclusive"}


def test_hf_callback_merge_aware_noop_when_unset() -> None:
    from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        model = _TinyPeftModel(r=4, alpha=16.0)

        config = GradienceCallbackConfig(
            output_dir=str(output_dir),
            filename="run.jsonl",
            merge_target=None,
        )
        callback = GradienceCallback(config)

        mock_args = _mock_hf_args(str(output_dir))
        mock_state = Mock()
        mock_control = Mock()

        mock_state.global_step = 0
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        mock_state.global_step = 1
        callback.on_log(
            mock_args,
            mock_state,
            mock_control,
            logs={"loss": 0.9, "learning_rate": 5e-4, "grad_norm": 2.0},
            model=model,
        )
        callback.on_train_end(mock_args, mock_state, mock_control, model=model)

        events = []
        with (output_dir / "run.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

    merge_kinds = {
        e.get("kind")
        for e in events
        if e.get("event") == "metrics" and str(e.get("kind", "")).startswith("merge_aware_")
    }
    assert not merge_kinds
