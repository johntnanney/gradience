"""
Gradience vNext telemetry

Canonical JSONL schema (v1) - PUBLIC API
-----------------------------------------

⚠️  This telemetry schema is part of the public API with stability guarantees.
    The schema version and core event structure will remain backward compatible.

Each line is a single JSON object with the following required keys:

  schema   : str   (must equal TELEMETRY_SCHEMA_VERSION)
  ts       : float (unix timestamp, seconds)
  run_id   : str
  event    : str   (event type)
  step     : int | null

Event types and payloads (stable set):
  - run_start:
      config: ConfigSnapshot dict
      meta  : dict (optional; environment, git hash, etc.)

  - train_step:
      loss: float (optional)
      lr  : float | list[float] (optional; per param-group)
      ... extras allowed

  - eval:
      split  : str ("train"|"val"|"test"|custom)
      metrics: dict (e.g., {"ppl": 2.3, "accuracy": 0.35, "n": 100})

  - metrics:
      kind   : str (e.g., "spectral"|"structural"|"lora_audit")
      metrics: dict

  - alert:
      severity: str ("info"|"warning"|"error"|"critical")
      code    : str (stable identifier, e.g. "memorization_gap")
      message : str
      context : dict (optional)

  - recommendation:
      recommendations: list[Recommendation dict]

  - run_end:
      status: str ("ok"|"aborted"|"error")
      reason: str (optional)

Notes
-----
- Extra keys are allowed on any event (forward-compatible).
- Do NOT change meanings of existing keys within a schema version.
  If you must, bump TELEMETRY_SCHEMA_VERSION.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .types import (
    TELEMETRY_SCHEMA_VERSION,
    Severity,
    ConfigSnapshot,
    Recommendation,
)

# NOTE: The reader lives in a separate module so writer-only users don't
# accidentally pull in heavier summarization logic.
from .telemetry_reader import TelemetryReader


Jsonable = Union[None, bool, int, float, str, Dict[str, Any], List[Any]]


def _to_jsonable(obj: Any) -> Jsonable:
    """Convert common python objects (dataclasses, enums) into JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # Enums: use `.value` if present
    if hasattr(obj, "value") and isinstance(getattr(obj, "value"), (str, int, float)):
        return getattr(obj, "value")
    if is_dataclass(obj):
        # Prefer explicit to_dict() if available to keep schema stable.
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return _to_jsonable(to_dict())
        return _to_jsonable(asdict(obj))
    # Fallback: string repr (avoid breaking telemetry on unexpected objects)
    return str(obj)


class TelemetryWriter:
    """Write vNext telemetry events to JSONL."""

    def __init__(
        self,
        path: Union[str, Path],
        run_id: Optional[str] = None,
        *,
        max_str_len: int = 256,
        allow_text: bool = False,
        on_text_violation: str = "redact",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.run_id = run_id or str(uuid.uuid4())
        # Privacy guardrails: prevent accidental logging of long raw text.
        self.max_str_len = int(max_str_len)
        self.allow_text = bool(allow_text)
        self.on_text_violation = str(on_text_violation)
        if self.on_text_violation not in ("redact", "raise"):
            raise ValueError("on_text_violation must be 'redact' or 'raise'")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._f.close()
        finally:
            self._closed = True

    def _write(self, record: Dict[str, Any]) -> None:
        if self._closed:
            return
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def _sanitize(self, obj: Any, *, _path: str = "") -> Any:
        """Redact long strings in telemetry payloads by default.

        This prevents accidental logging of prompts/examples into JSONL.
        Set allow_text=True to permit long strings.
        """
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            if self.allow_text:
                return obj
            if self.max_str_len is not None and len(obj) > self.max_str_len:
                if self.on_text_violation == "raise":
                    where = _path or "<root>"
                    raise ValueError(f"Telemetry text too long ({len(obj)} chars) at {where}.")
                return f"[REDACTED len={len(obj)}]"
            return obj
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                key = str(k)
                child = f"{_path}.{key}" if _path else key
                out[key] = self._sanitize(v, _path=child)
            return out
        if isinstance(obj, (list, tuple)):
            out_list: List[Any] = []
            for i, v in enumerate(obj):
                child = f"{_path}[{i}]"
                out_list.append(self._sanitize(v, _path=child))
            return out_list
        # Fallback: stringify then sanitize length
        return self._sanitize(str(obj), _path=_path)

    def log(self, event: str, *, step: Optional[int] = None, **payload: Any) -> None:
        record: Dict[str, Any] = {
            "schema": TELEMETRY_SCHEMA_VERSION,
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event,
            "step": step,
        }
        for k, v in payload.items():
            record[k] = self._sanitize(_to_jsonable(v), _path=str(k))
        record = self._sanitize(record, _path="<record>")
        self._write(record)

    # ---- convenience helpers ----

    def run_start(self, config: ConfigSnapshot, *, meta: Optional[Dict[str, Any]] = None) -> str:
        self.log("run_start", step=None, config=config, meta=meta or {})
        return self.run_id

    def train_step(self, step: int, *, loss: Optional[float] = None, lr: Any = None, **extras: Any) -> None:
        payload: Dict[str, Any] = {"loss": loss, "lr": lr}
        payload.update(extras)
        # Remove explicit None keys to keep logs tidy (optional).
        payload = {k: v for k, v in payload.items() if v is not None}
        self.log("train_step", step=step, **payload)

    def eval(self, step: int, *, split: str, metrics: Dict[str, Any], **extras: Any) -> None:
        payload: Dict[str, Any] = {"split": split, "metrics": metrics}
        payload.update(extras)
        self.log("eval", step=step, **payload)

    def metrics(self, step: int, *, kind: str, metrics: Dict[str, Any], **extras: Any) -> None:
        payload: Dict[str, Any] = {"kind": kind, "metrics": metrics}
        payload.update(extras)
        self.log("metrics", step=step, **payload)

    def alert(
        self,
        *,
        severity: Severity,
        code: str,
        message: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        **extras: Any,
    ) -> None:
        payload: Dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
            "context": context or {},
        }
        payload.update(extras)
        self.log("alert", step=step, **payload)

    def recommendation(self, recommendations: List[Recommendation], *, step: Optional[int] = None, **extras: Any) -> None:
        payload: Dict[str, Any] = {"recommendations": recommendations}
        payload.update(extras)
        self.log("recommendation", step=step, **payload)

    def run_end(self, *, status: str = "ok", reason: Optional[str] = None, **extras: Any) -> None:
        payload: Dict[str, Any] = {"status": status, "reason": reason}
        payload.update(extras)
        # Keep reason only if provided.
        payload = {k: v for k, v in payload.items() if v is not None}
        self.log("run_end", step=None, **payload)

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            # If an exception occurs, annotate the log.
            self.run_end(status="error", reason=str(exc))
        self.close()
