from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..telemetry import TelemetryWriter
from ..types import (
    ConfigSnapshot,
    LoRAConfigSnapshot,
    OptimizerConfigSnapshot,
    TaskProfile,
    TrainingConfigSnapshot,
)

# Optional import so the base package doesn't require transformers unless used.
try:
    from transformers import TrainerCallback  # type: ignore
    from transformers.trainer_callback import TrainerControl, TrainerState  # type: ignore
    from transformers.training_args import TrainingArguments  # type: ignore
except Exception as _e:  # pragma: no cover
    TrainerCallback = object  # type: ignore
    TrainerControl = Any  # type: ignore
    TrainerState = Any  # type: ignore
    TrainingArguments = Any  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR = _e
else:
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass
class GradienceCallbackConfig:
    """
    Conservative, low-overhead HF integration config.

    - Output defaults to: training_args.output_dir/run.jsonl
    - dataset_name / task_profile are optional and not required.
    """
    output_dir: Optional[Union[str, Path]] = None
    filename: str = "run.jsonl"

    # Optional metadata (NOT required)
    dataset_name: Optional[str] = None
    task_profile: Optional[Union[str, TaskProfile]] = None
    notes: Optional[str] = None

    # Telemetry privacy knobs (if your TelemetryWriter supports them)
    # Defaults match your current stance: redact long strings unless opted in.
    telemetry_allow_text: bool = False
    telemetry_max_str_len: int = 256


def _coerce_task_profile(tp: Optional[Union[str, TaskProfile]]) -> TaskProfile:
    if tp is None:
        return TaskProfile.UNKNOWN
    if isinstance(tp, TaskProfile):
        return tp
    # Accept strings like "easy_classification"
    try:
        return TaskProfile(str(tp))
    except Exception:
        return TaskProfile.UNKNOWN


def _best_effort_model_name(model: Any) -> str:
    """
    Try hard to get a stable model identifier without touching anything expensive.
    """
    # HuggingFace models usually have model.config.name_or_path or _name_or_path
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("name_or_path", "_name_or_path", "model_type"):
                val = getattr(cfg, attr, None)
                if val:
                    return str(val)
    except Exception:
        pass

    # Fallbacks
    try:
        name = getattr(model, "name_or_path", None)
        if name:
            return str(name)
    except Exception:
        pass

    return model.__class__.__name__ if model is not None else "unknown"


def _best_effort_lora_snapshot(model: Any) -> LoRAConfigSnapshot:
    """
    Conservative LoRA detection:
    - If PEFT is present and model exposes peft_config, read it.
    - Otherwise leave LoRA snapshot mostly empty (that's fine).
    """
    # Default (empty-ish) snapshot
    snap = LoRAConfigSnapshot(
        r=None,
        alpha=None,
        target_modules=[],
        dropout=None,
        bias=None,
        extras={},
    )

    if model is None:
        return snap

    # Try PEFT config (no heavy ops; just attribute reads)
    peft_cfg = None
    try:
        pc = getattr(model, "peft_config", None)
        if isinstance(pc, dict) and pc:
            peft_cfg = pc.get("default") or next(iter(pc.values()))
        elif pc is not None:
            peft_cfg = pc
    except Exception:
        peft_cfg = None

    if peft_cfg is None:
        return snap

    def _as_list(x: Any) -> list[str]:
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            return [str(v) for v in x]
        return [str(x)]

    try:
        r = getattr(peft_cfg, "r", None)
        alpha = getattr(peft_cfg, "lora_alpha", None)
        dropout = getattr(peft_cfg, "lora_dropout", None)
        bias = getattr(peft_cfg, "bias", None)
        targets = _as_list(getattr(peft_cfg, "target_modules", None))

        return LoRAConfigSnapshot(
            r=int(r) if r is not None else None,
            alpha=float(alpha) if alpha is not None else None,
            target_modules=targets,
            dropout=float(dropout) if dropout is not None else None,
            bias=str(bias) if bias is not None else None,
            extras={},
        )
    except Exception:
        # If anything goes sideways, we just don't populate LoRA snapshot.
        return snap


def build_conservative_config_snapshot(
    args: Any,
    model: Any,
    *,
    dataset_name: Optional[str] = None,
    task_profile: Optional[Union[str, TaskProfile]] = None,
    notes: Optional[str] = None,
) -> ConfigSnapshot:
    """
    Build a conservative ConfigSnapshot.

    - No dataset/task required.
    - Avoids dumping huge HF args dicts.
    - Avoids paths and raw text.
    """
    # Training basics (safe, stable)
    seed = getattr(args, "seed", None)
    bs = getattr(args, "per_device_train_batch_size", None)
    gas = getattr(args, "gradient_accumulation_steps", None)
    max_steps = getattr(args, "max_steps", None)
    num_epochs = getattr(args, "num_train_epochs", None)

    # Mixed precision flags
    fp16 = bool(getattr(args, "fp16", False))
    bf16 = bool(getattr(args, "bf16", False))

    # Optimizer basics
    optim_name = getattr(args, "optim", None)  # e.g. "adamw_torch"
    lr = getattr(args, "learning_rate", None)
    weight_decay = getattr(args, "weight_decay", None)

    # Optional optimizer details (common HF args)
    beta1 = getattr(args, "adam_beta1", None)
    beta2 = getattr(args, "adam_beta2", None)
    eps = getattr(args, "adam_epsilon", None)

    opt = OptimizerConfigSnapshot(
        name=str(optim_name) if optim_name is not None else "adamw",
        lr=float(lr) if lr is not None else None,
        weight_decay=float(weight_decay) if weight_decay is not None else None,
        betas=(float(beta1), float(beta2)) if beta1 is not None and beta2 is not None else None,
        eps=float(eps) if eps is not None else None,
        extras={},
    )

    tr = TrainingConfigSnapshot(
        seed=int(seed) if seed is not None else None,
        batch_size=int(bs) if bs is not None else None,
        gradient_accumulation=int(gas) if gas is not None else None,
        max_steps=int(max_steps) if isinstance(max_steps, int) and max_steps > 0 else None,
        epochs=float(num_epochs) if num_epochs is not None else None,
        dtype="fp16" if fp16 else ("bf16" if bf16 else None),
        extras={},
    )

    lora = _best_effort_lora_snapshot(model)

    snap = ConfigSnapshot(
        model_name=_best_effort_model_name(model),
        dataset_name=dataset_name,  # optional; can be None
        task_profile=_coerce_task_profile(task_profile),
        optimizer=opt,
        lora=lora,
        training=tr,
        notes=notes,
        extras={},  # keep conservative; avoid paths/host/user by default
    )
    return snap


class GradienceCallback(TrainerCallback):
    """
    Minimal HuggingFace Trainer callback that emits Gradience telemetry.

    Conservative defaults:
    - Output: output_dir/run.jsonl
    - No expensive metrics at runtime (no SVD, no per-layer stuff)
    - dataset_name/task_profile are optional
    """

    def __init__(self, config: Optional[GradienceCallbackConfig] = None):
        if _TRANSFORMERS_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "Gradience HF integration requires transformers. "
                "Install with `pip install transformers` (or your project's hf extra)."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.config = config or GradienceCallbackConfig()
        self.writer: Optional[TelemetryWriter] = None
        self._run_id: Optional[str] = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)

        out_dir = Path(self.config.output_dir or getattr(args, "output_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / self.config.filename  # default: run.jsonl

        # Build writer kwargs defensively (works across TelemetryWriter versions)
        writer_kwargs: Dict[str, Any] = {}
        sig = inspect.signature(TelemetryWriter.__init__)
        if "allow_text" in sig.parameters:
            writer_kwargs["allow_text"] = self.config.telemetry_allow_text
        if "max_str_len" in sig.parameters:
            writer_kwargs["max_str_len"] = self.config.telemetry_max_str_len

        self.writer = TelemetryWriter(str(path), run_id=None, **writer_kwargs)

        snap = build_conservative_config_snapshot(
            args,
            model,
            dataset_name=self.config.dataset_name,
            task_profile=self.config.task_profile,
            notes=self.config.notes,
        )

        self._run_id = self.writer.run_start(snap, meta={"framework": "huggingface"})

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, Any]] = None, **kwargs):
        # Cheap logging only: whatever HF already computed (loss, lr, grad_norm, etc.)
        if self.writer is None or logs is None:
            return
        step = int(getattr(state, "global_step", 0))

        # Don't spam: only log when HF logs (controlled by logging_steps)
        payload = {}
        for k in ("loss", "learning_rate", "grad_norm", "epoch"):
            if k in logs:
                # map learning_rate -> lr for our schema's train_step helper
                if k == "learning_rate":
                    payload["lr"] = logs[k]
                else:
                    payload[k] = logs[k]

        if payload:
            self.writer.train_step(step, **payload)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        if self.writer is None or metrics is None:
            return
        step = int(getattr(state, "global_step", 0))

        # HF uses eval_* keys. We'll normalize by stripping "eval_" prefix.
        clean = {}
        for k, v in metrics.items():
            kk = str(k)
            if kk.startswith("eval_"):
                kk = kk[len("eval_") :]
            clean[kk] = v

        self.writer.eval(step, split="eval", metrics=clean)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.writer is None:
            return
        self.writer.run_end(status="ok")
        self.writer.close()
        self.writer = None