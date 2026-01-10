from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from transformers import TrainerCallback  # type: ignore
except Exception as e:
    raise ImportError("transformers is required for GradienceVNextCallback. pip install transformers") from e

from gradience.vnext.telemetry import TelemetryWriter, TELEMETRY_SCHEMA_VERSION


def _maybe_asdict(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


def _guess_task_profile(dataset_name: Optional[str]) -> str:
    if not dataset_name:
        return "unknown"
    d = dataset_name.lower()
    if "gsm8k" in d or "svamp" in d or "math" in d:
        return "hard_reasoning"
    if "sst2" in d or "sst-2" in d or "glue" in d:
        return "easy_classification"
    return "unknown"


class GradienceVNextCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that emits vNext telemetry JSONL.

    Writes events:
      - run_start (with best-effort ConfigSnapshot-like dict if config not provided)
      - train_step (loss + lr)
      - eval (split="val" by default; configurable via eval_split)
      - run_end

    Usage:
      cb = GradienceVNextCallback(
            telemetry_path="runs/my_run/run.jsonl",
            dataset_name="gsm8k",
            model_name="mistral-7b",
            lora_config={...},   # optional
            task_profile="hard_reasoning",  # optional
            eval_split="test",   # optional
          )
      trainer = Trainer(..., callbacks=[cb])
    """

    def __init__(
        self,
        telemetry_path: str,
        *,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        task_profile: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        eval_split: str = "val",
        log_every_n_steps: int = 10,
        strict_schema: bool = False,
        telemetry_allow_text: bool = False,
        telemetry_max_str_len: int = 256,
        telemetry_on_text_violation: str = "redact",
    ) -> None:
        # Telemetry privacy defaults: redact long strings unless explicitly opted in.
        self._telemetry_writer_kwargs = dict(
            max_str_len=telemetry_max_str_len,
            allow_text=telemetry_allow_text,
            on_text_violation=telemetry_on_text_violation,
        )
        if telemetry_allow_text:
            print("[gradience] WARNING: --telemetry-allow-text enabled; telemetry JSONL may include raw prompts/examples. Treat it as sensitive.")

        self.telemetry_path = str(telemetry_path)
        self.run_id = run_id or f"run_{int(time.time())}"
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task_profile = task_profile or _guess_task_profile(dataset_name)
        self.lora_config = lora_config or {}
        self.meta = meta or {}
        self.eval_split = eval_split
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.strict_schema = strict_schema

        self._writer: Optional[TelemetryWriter] = None
        self._started = False

    def _ensure_writer(self) -> TelemetryWriter:
        if self._writer is None:
            Path(self.telemetry_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = TelemetryWriter(self.telemetry_path, run_id=self.run_id, **self._telemetry_writer_kwargs)
        return self._writer

    def _build_config(self, args: Any) -> Dict[str, Any]:
        # Best-effort ConfigSnapshot-like dict (works with vNext TelemetryReader)
        # We keep it minimal and shove unknowns into extras.
        opt = {
            "name": getattr(args, "optim", None) or "AdamW",
            "lr": float(getattr(args, "learning_rate", 0.0) or 0.0),
            "weight_decay": float(getattr(args, "weight_decay", 0.0) or 0.0),
        }
        training = {
            "seed": getattr(args, "seed", None),
            "per_device_train_batch_size": getattr(args, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
            "max_steps": getattr(args, "max_steps", None),
            "num_train_epochs": getattr(args, "num_train_epochs", None),
            "bf16": getattr(args, "bf16", None),
            "fp16": getattr(args, "fp16", None),
        }
        cfg = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "task_profile": self.task_profile,
            "optimizer": opt,
            "lora": self.lora_config,
            "training": training,
            "extras": {
                "schema": TELEMETRY_SCHEMA_VERSION,
                "output_dir": getattr(args, "output_dir", None),
            },
        }
        return cfg

    def on_train_begin(self, args, state, control, **kwargs):
        w = self._ensure_writer()
        if not self._started:
            cfg = self._build_config(args)
            w.run_start(cfg, meta=self.meta)
            self._started = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = getattr(state, "global_step", None)
        if step is None:
            return
        if step % self.log_every_n_steps != 0:
            return

        w = self._ensure_writer()
        loss = logs.get("loss")
        lr = logs.get("learning_rate") or logs.get("lr")
        # best-effort
        try:
            loss_f = float(loss) if loss is not None else None
        except Exception:
            loss_f = None
        try:
            lr_f = float(lr) if lr is not None else None
        except Exception:
            lr_f = None

        if loss_f is not None:
            w.train_step(int(step), loss=loss_f, lr=lr_f)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        w = self._ensure_writer()
        step = getattr(state, "global_step", None)
        step_i = int(step) if step is not None else None

        m = metrics or {}

        # Infer split from metric prefixes when using Trainer.evaluate(metric_key_prefix=...)
        split = self.eval_split
        if any(k.startswith("train_") for k in m.keys()):
            split = "train"
        elif any(k.startswith("test_") for k in m.keys()):
            split = "test"
        elif any(k.startswith("eval_") for k in m.keys()):
            # default eval split
            split = self.eval_split

        # Prefix-aware metric extraction
        def _get(pref: str, key: str):
            return m.get(f"{pref}_{key}") if pref else m.get(key)

        pref = "eval"
        if split == "train":
            pref = "train"
        elif split == "test":
            pref = "test"


        # HF conventions: eval_loss, eval_accuracy, etc.
        loss = _get(pref, 'loss')
        acc = _get(pref, 'accuracy')
        ppl = _get(pref, 'ppl')

        metrics_out: Dict[str, Any] = {}
        if loss is not None:
            metrics_out["loss"] = float(loss)
            # If ppl not provided, for classification runs ppl=exp(loss) is a reasonable monotonic proxy.
            if ppl is None:
                try:
                    import math
                    metrics_out["ppl"] = float(math.exp(float(loss)))
                except Exception:
                    pass
        if ppl is not None:
            metrics_out["ppl"] = float(ppl)
        if acc is not None:
            metrics_out["accuracy"] = float(acc)

        # n not always known; allow it to be missing
        w.eval(step_i, split=split, metrics=metrics_out)

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer is not None:
            self._writer.run_end(status="ok")
            self._writer.close()
